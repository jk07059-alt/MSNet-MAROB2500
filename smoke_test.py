#!/usr/bin/env python3
"""Offline smoke test -- proves the released code runs WITHOUT the proprietary data.

Part A  (no BasicSR, no data): synthesize an anisotropic Gaussian PSF and run the kernel
        half of the blur model (maritime_kernel.py) -- resample, covariance, sigma/theta/rho.
Part B  (needs the BasicSR/NAFNet tree): instantiate MSNet and NAFNet from the released
        arch files and run a forward pass on a random tensor; check param counts + shape.

Run (after setup.sh, from the repo root):
    python smoke_test.py
    # if BasicSR is not auto-found, point at your NAFNet tree:
    NAFNET_ROOT=/path/to/NAFNet python smoke_test.py
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def part_a_kernel_model():
    print("== Part A: blur-model kernel math (maritime_kernel.py) ==")
    sys.path.insert(0, os.path.join(HERE, "code", "data"))
    import cv2
    import maritime_kernel as ok

    # synthesize a known anisotropic Gaussian PSF (3:1, major axis ~30 deg) at 21x21,
    # write it where maritime_kernel expects a bank entry, and recover its statistics.
    with tempfile.TemporaryDirectory() as td:
        bank = os.path.join(td, ok.SRC_DIR)
        os.makedirs(bank)
        n = 21
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) - (n - 1) / 2.0
        th = np.deg2rad(30.0)
        c, s = np.cos(th), np.sin(th)
        u = c * xx + s * yy
        v = -s * xx + c * yy
        sx, sy = 4.0, 4.0 / 3.0
        psf = np.exp(-(u ** 2) / (2 * sx ** 2) - (v ** 2) / (2 * sy ** 2))
        cv2.imwrite(os.path.join(bank, "synthetic.png"),
                    (psf / psf.max() * 255).astype(np.uint8))

        k = ok.load_kernel_continuous(os.path.join(bank, "synthetic.png"), target=42)
        assert abs(k.sum() - 1.0) < 1e-6, "kernel must be sum-normalised"
        sigma, theta, rho = ok.cov_to_stats(ok.kernel_cov(k))
        # passes add covariance: sigma(k*^3) should be ~sqrt(3)x sigma(k)
        sig3, _, _ = ok.cov_to_stats(ok.cov_after_passes(ok.kernel_cov(k), 3))
        ratio = sig3 / sigma
        print(f"  sigma={sigma:.2f}px  theta={theta:.1f}deg  rho={rho:.2f}  "
              f"(sigma x3-pass /x1 = {ratio:.3f}, expect ~{np.sqrt(3):.3f})")
        assert rho > 1.5, "anisotropy not recovered"
        assert abs(ratio - np.sqrt(3)) < 0.05, "pass-scaling law broken"
    print("  Part A PASS\n")


def part_b_networks():
    print("== Part B: MSNet / NAFNet forward pass ==")
    root = os.environ.get("NAFNET_ROOT")
    if root:
        sys.path.insert(0, root)
    try:
        import torch
        from basicsr.models.archs import define_network
    except Exception as e:
        print(f"  SKIPPED: BasicSR/NAFNet not importable ({type(e).__name__}: {e}).")
        print("  Run setup.sh first, or set NAFNET_ROOT=/path/to/NAFNet. See INSTALL.md.")
        return False

    cfgs = {
        "MSNet": dict(type="MSNet", img_channel=3, width=32, middle_blk_num=1,
                      enc_blk_nums=[1, 1, 1, 26], dec_blk_nums=[1, 1, 1, 1],
                      mar_stages=[0, 1, 2], strip_len=65, deep_every=6),
        "NAFNet": dict(type="NAFNet", img_channel=3, width=32, middle_blk_num=1,
                       enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]),
    }
    x = torch.randn(1, 3, 256, 256)
    for name, cfg in cfgs.items():
        net = define_network(dict(cfg)).eval()
        params = sum(p.numel() for p in net.parameters()) / 1e6
        with torch.no_grad():
            y = net(x)
            y = y[-1] if isinstance(y, (list, tuple)) else y
        assert tuple(y.shape) == (1, 3, 256, 256), f"{name} output shape wrong"
        print(f"  {name:7s} OK  params={params:.2f}M  out={tuple(y.shape)}")
    print("  Part B PASS\n")
    return True


if __name__ == "__main__":
    part_a_kernel_model()
    ok_b = part_b_networks()
    print("SMOKE TEST PASS" if ok_b else "SMOKE TEST: Part A passed, Part B skipped (see above)")
