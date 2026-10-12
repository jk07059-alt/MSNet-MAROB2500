# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the MSNet / MAROB-2500 authors. See the LICENSE file.
# ------------------------------------------------------------------------
# MaritimeDataset -- paired maritime deblurring with on-the-fly severity augmentation.
# It also emits exact, free degradation labels (code_gt / sigma_field); the default training
# model (ImageRestorationModel) ignores them, and they are left in for anyone extending the
# method with auxiliary supervision.
#
# WHY THIS EXISTS. The stored train blur sits at sigma_bg ~ 3.9 while the v3 eval spans
# sigma_bg in [3, 10] -- disjoint kernel-size and pass supports. A net trained only on the
# stored blur is evaluated largely out of its training distribution; closing that gap is worth
# more than any architecture change (measured -1.94 dB of extra degradation on the eval
# regime). So in the default (train) mode this dataset IGNORES the stored blur and re-blurs the
# sharp crop on the fly, drawing (sigma_bg, Delta) across BOTH regimes. Because we synthesise
# the blur, we know its second moment exactly -- the degradation labels are free and
# noise-free.
#
# THE theta-UNDER-AUGMENTATION TRAP, SIDESTEPPED. flip/rot90 change the blur axis: a naive
# pipeline must map theta*-> -theta* under flip and theta*+90 under rot90, and getting it wrong
# trains BMC on noise while the loss looks fine. This dataset applies the geometric
# augmentation to the SHARP crop and mask FIRST, then blurs in that already-transformed frame,
# so the label is simply the measured angle of the kernel in the final frame. No theta
# transform, no bug.
#
# BIT-CONSISTENCY WITH THE EVAL. The blur is composed with the exact functions the v3 eval
# generator used (convolve_image_multi_pass, blend_masked_region, feather_mask, dilate_mask)
# and the same continuous-target kernel loader (maritime_kernel), so augmented train blur and eval
# blur share kernel character. NOTE cv2.filter2D is CORRELATION, so the realized operator is
# flip(k) applied N times; second moments are invariant under that point reflection, so the
# MEASURED labels are exact for the realized blur.
# ------------------------------------------------------------------------

import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

from basicsr.data.transforms import augment
from basicsr.utils import img2tensor

# The kernel-bank utilities (maritime_kernel.py) and the compositing helpers
# (ship_mask_blur_pipeline.py) are shipped alongside this file in the release. setup.sh copies
# all three into basicsr/data/, so adding this file's own directory to sys.path lets the bare
# imports below resolve wherever the release is dropped into a BasicSR tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from maritime_kernel import (  # noqa: E402
    KERNEL_ROOT,
    SRC_DIR,
    build_sigma_grid,
    cov_after_passes,
    cov_to_stats,
    kernel_cov,
    load_kernel_continuous,
)
from ship_mask_blur_pipeline import (  # noqa: E402
    blend_masked_region,
    convolve_image_multi_pass,
    dilate_mask,
    feather_mask,
)

TARGETS = range(6, 61)
PASSES = (2, 3, 4)
CROP_SRC = 384          # source crop; blurred then centre-cropped to gt_size, leaving margin
                        # for the ~40px blur-support radius at the top of the sigma range
SB_RANGE = (2.5, 11.0)  # slightly wider than the eval box [3, 10] so training brackets it
DELTA_RANGE = (0.0, 9.5)
MASK_DILATE, FEATHER = 5, 11  # match the eval generator exactly


def _rotate_kernel(k, angle_deg):
    """Rotate a kernel by an arbitrary angle without clipping, then renormalise.

    The kernel is padded onto a larger square so an elongated PSF cannot lose mass at the
    corners, rotated about its centre, clipped non-negative (a PSF cannot be negative; cubic
    interpolation undershoots), and re-summed to 1. Labels are always MEASURED after this, so
    interpolation error in the rotation shows up in the (exact) label, never as label noise.
    """
    h = k.shape[0]
    pad = int(h * 0.5) + 2
    kp = np.pad(k, pad)
    hh = kp.shape[0]
    m = cv2.getRotationMatrix2D((hh / 2.0, hh / 2.0), angle_deg, 1.0)
    kr = cv2.warpAffine(kp, m, (hh, hh), flags=cv2.INTER_CUBIC, borderValue=0.0)
    kr = np.clip(kr, 0.0, None)
    s = float(kr.sum())
    return kr / s if s > 1e-12 else k


def _solve(grid, want, passes, floor=None):
    sub = [g for g in grid if g[2] == passes and (floor is None or g[0] > floor + 1e-9)]
    if not sub:
        sub = [max((g for g in grid if g[2] == passes), key=lambda g: g[0])]
    arr = np.fromiter((g[0] for g in sub), dtype=np.float64, count=len(sub))
    return sub[int(np.argmin(np.abs(arr - want)))]


class MaritimeDataset(data.Dataset):
    """Modes (from opt):
      phase == 'train', severity_aug True  (default): reblur sharp on the fly, exact labels.
      phase == 'train', severity_aug False           : stored blur, labels from the manifest.
      phase == 'val'                                  : stored blur+sharp, no labels.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt["phase"]
        self.gt_size = opt.get("gt_size", 256)
        self.severity_aug = bool(opt.get("severity_aug", True)) and self.phase == "train"
        self.ship_crop_prob = float(opt.get("ship_crop_prob", 0.5))
        self.use_flip = opt.get("use_flip", True)
        self.use_rot = opt.get("use_rot", True)

        self.gt_root = Path(opt["dataroot_gt"])
        self.lq_root = Path(opt["dataroot_lq"])

        # Join disk files to manifest rows by filename stem. group_id is a GROUP key (it
        # collides across images), so it must NOT be used -- the blur_dst basename is the
        # per-row key that joins exactly.
        manifest = opt.get("manifest")
        self.meta = {}
        if manifest and os.path.exists(manifest):
            split = opt.get("manifest_split", "train")
            for r in csv.DictReader(open(manifest, newline="", encoding="utf-8")):
                if r.get("split") != split:
                    continue
                self.meta[Path(r["blur_dst"]).name] = r

        exts = (".png", ".jpg", ".jpeg")
        self.names = sorted(f for f in os.listdir(self.gt_root) if f.lower().endswith(exts))
        if not self.names:
            raise ValueError(f"No images under {self.gt_root}")

        # Precompute the measured-sigma grid per kernel once (used only in aug mode).
        self.kernel_names = sorted(p.name for p in (KERNEL_ROOT / SRC_DIR).glob("*.png"))
        self.grids = (
            {n: build_sigma_grid(n, TARGETS, PASSES) for n in self.kernel_names}
            if self.severity_aug else {}
        )

    def __len__(self):
        return len(self.names)

    # ---- helpers -------------------------------------------------------------------------

    def _load_mask(self, name):
        r = self.meta.get(name)
        if r is None or r.get("has_ship") != "1" or not r.get("mask_path") or not os.path.exists(r["mask_path"]):
            return None
        m = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        return (m > 0).astype(np.uint8) if m is not None else None

    def _crop_src(self, sharp, mask):
        """Ship-biased CROP_SRC x CROP_SRC crop of the sharp image (+ aligned mask)."""
        h, w = sharp.shape[:2]
        cs = min(CROP_SRC, h, w)
        top = left = 0
        if mask is not None and mask.any() and random.random() < self.ship_crop_prob:
            ys, xs = np.where(mask > 0)
            cy, cx = int(ys[random.randrange(len(ys))]), int(xs[random.randrange(len(xs))])
            top = int(np.clip(cy - cs // 2, 0, h - cs))
            left = int(np.clip(cx - cs // 2, 0, w - cs))
        elif h > cs or w > cs:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        s = sharp[top:top + cs, left:left + cs]
        m = mask[top:top + cs, left:left + cs] if mask is not None else None
        return s, m

    def _reblur(self, sharp_u8, mask_u8):
        """Compose a fresh two-scale blur on the (already flip/rot'd) sharp crop.

        Returns (lq_u8, code[5], sigma_field HxW). code = [cos2t, sin2t, rho, sigma_bg, delta].
        """
        sb_want = random.uniform(*SB_RANGE)
        delta_want = random.uniform(*DELTA_RANGE)
        kname = random.choice(self.kernel_names)
        angle = random.uniform(0.0, 180.0)   # rotation augmentation: 16 kernels -> continuous axis
        n_bg = random.choice(PASSES)
        n_ship = random.choice(PASSES)
        grid = self.grids[kname]
        kpath = KERNEL_ROOT / SRC_DIR / kname

        _, t_bg, _ = _solve(grid, sb_want, n_bg)
        k_bg = _rotate_kernel(load_kernel_continuous(kpath, t_bg), angle)
        sigma_bg, theta, rho = cov_to_stats(cov_after_passes(kernel_cov(k_bg), n_bg))
        bg_blur = convolve_image_multi_pass(sharp_u8, k_bg.astype(np.float32), n_bg)

        has_ship = mask_u8 is not None and mask_u8.any() and delta_want > 0
        if has_ship:
            _, t_ship, _ = _solve(grid, float(np.sqrt(sigma_bg**2 + delta_want**2)), n_ship, floor=sigma_bg)
            k_ship = _rotate_kernel(load_kernel_continuous(kpath, t_ship), angle)  # SAME angle: coaxial
            sigma_ship, _, _ = cov_to_stats(cov_after_passes(kernel_cov(k_ship), n_ship))
            delta = float(np.sqrt(max(sigma_ship**2 - sigma_bg**2, 0.0)))
            ship_blur = convolve_image_multi_pass(sharp_u8, k_ship.astype(np.float32), n_ship)
            alpha = feather_mask(dilate_mask(mask_u8, MASK_DILATE), FEATHER)          # HxW float32
            lq = blend_masked_region(bg_blur, ship_blur, alpha)
            # variance-linear blend -- verified correct (sigma-linear errs up to -14.5% at the
            # v3 extreme). Getting this wrong biases the field target everywhere the ship is.
            sfield = np.sqrt(alpha * sigma_ship**2 + (1.0 - alpha) * sigma_bg**2).astype(np.float32)
        else:
            delta = 0.0
            lq = bg_blur
            sfield = np.full(sharp_u8.shape[:2], sigma_bg, np.float32)

        code = np.array(
            [np.cos(2 * np.radians(theta)), np.sin(2 * np.radians(theta)), rho, sigma_bg, delta],
            dtype=np.float32,
        )
        return lq, code, sfield

    def _manifest_code(self, name):
        """Labels for stored-blur train mode (severity_aug False): measure from the manifest."""
        r = self.meta.get(name)
        if r is None:
            return None, None
        kpath = KERNEL_ROOT / SRC_DIR / r["shared_kernel_name"]

        def stat(size_label, passes):
            t = int(str(size_label).lower().split("by")[0])
            return cov_to_stats(cov_after_passes(kernel_cov(load_kernel_continuous(kpath, t)), int(passes)))

        sb, theta, rho = stat(r["background_kernel_size"], r["background_passes"])
        if r.get("has_ship") == "1":
            ss, _, _ = stat(r["ship_kernel_size"], r["ship_passes"])
        else:
            ss = sb
        delta = float(np.sqrt(max(ss**2 - sb**2, 0.0)))
        code = np.array(
            [np.cos(2 * np.radians(theta)), np.sin(2 * np.radians(theta)), rho, sb, delta], np.float32
        )
        return code, sb

    # ---- item ----------------------------------------------------------------------------

    def __getitem__(self, index):
        name = self.names[index]
        gt_path = str(self.gt_root / name)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)  # BGR uint8, to match the eval generator
        if gt is None:
            raise IOError(f"Failed to read {gt_path}")

        gs = self.gt_size

        if self.phase != "train":
            # Validation / test: use the stored blur; only image PSNR is measured, no labels.
            lq = cv2.imread(str(self.lq_root / name), cv2.IMREAD_COLOR)
            gt_t, lq_t = img2tensor([gt[:, :, ::-1] / 255.0, lq[:, :, ::-1] / 255.0], bgr2rgb=False, float32=True)
            return {"lq": lq_t, "gt": gt_t, "lq_path": str(self.lq_root / name), "gt_path": gt_path}

        mask = self._load_mask(name)

        if self.severity_aug:
            s_src, m_src = self._crop_src(gt, mask)
            # Geometric aug BEFORE blur -> theta needs no transform (see module docstring).
            if m_src is None:
                s_src = augment([s_src], self.use_flip, self.use_rot)
                m_aug = None
            else:
                s_src, m_aug = augment([s_src, m_src[..., None]], self.use_flip, self.use_rot)
                m_aug = m_aug[..., 0].astype(np.uint8)
            lq_src, code, sfield = self._reblur(s_src, m_aug)

            # centre-crop CROP_SRC -> gt_size (drops the blur-contaminated margin)
            off = (s_src.shape[0] - gs) // 2
            sl = slice(off, off + gs)
            gt_c, lq_c, sf_c = s_src[sl, sl], lq_src[sl, sl], sfield[sl, sl]
            mask_c = m_aug[sl, sl] if m_aug is not None else np.zeros((gs, gs), np.uint8)

            gt_t, lq_t = img2tensor([gt_c[:, :, ::-1] / 255.0, lq_c[:, :, ::-1] / 255.0], bgr2rgb=False, float32=True)
            return {
                "lq": lq_t, "gt": gt_t, "gt_path": gt_path, "lq_path": gt_path,
                "ship_mask": torch.from_numpy(mask_c[None].astype(np.float32)),
                "code_gt": torch.from_numpy(code),
                "sigma_field": torch.from_numpy(sf_c[None]),
            }

        # severity_aug False: stored blur + manifest labels (R9 ablation).
        lq = cv2.imread(str(self.lq_root / name), cv2.IMREAD_COLOR)
        from basicsr.data.transforms import paired_random_crop
        gt_f, lq_f = gt[:, :, ::-1] / 255.0, lq[:, :, ::-1] / 255.0
        gt_c, lq_c = paired_random_crop(gt_f, lq_f, gs, 1, gt_path)
        gt_c, lq_c = augment([gt_c, lq_c], self.use_flip, self.use_rot)
        gt_t, lq_t = img2tensor([gt_c, lq_c], bgr2rgb=False, float32=True)
        code, sb = self._manifest_code(name)
        out = {"lq": lq_t, "gt": gt_t, "gt_path": gt_path, "lq_path": str(self.lq_root / name)}
        if code is not None:
            out["code_gt"] = torch.from_numpy(code)
            out["sigma_field"] = torch.full((1, gs, gs), float(sb))
            out["ship_mask"] = torch.zeros(1, gs, gs)
        return out
