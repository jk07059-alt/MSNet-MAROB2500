#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the MSNet / MAROB-2500 authors. See the LICENSE file.
"""Continuous-scale vibration-kernel utilities for the MAROB-2500 blur model.

This file is the *kernel half* of the blur-formation model (the compositing half lives in
``ship_mask_blur_pipeline.py``). It is pure algorithm: it loads a measured platform-vibration
PSF, resamples it to an arbitrary scale, and derives the second-order statistics (sigma, angle,
anisotropy) that drive the severity-controllable blur. Nothing here is proprietary.

The *measured kernel bank itself* (the PNG files under ``KERNEL_ROOT / SRC_DIR``) is the
defense-related, non-releasable data (see README "Data availability"). To reproduce the
pipeline on your own imagery, drop a bank of square, single-channel PSF PNGs into the directory
below (or set the ``MAROB_KERNEL_ROOT`` environment variable to point at it) and the dataset in
``maritime_dataset.py`` will synthesize blur from them exactly as the paper describes.

Conventions fixed here, once, deliberately:

    sigma := sqrt(trace(Cov) / 2)        # per-axis RMS extent, in pixels

    There are two conventions in circulation for this quantity and they differ by sqrt(2).
    Design notes quoting "sigma in [3, 16]" are in the sqrt(trace(Cov)) convention; every
    number produced by this module is in the sqrt(trace(Cov)/2) convention above. Mixing them
    silently mis-scales every downstream fit. ``SIGMA_CONVENTION`` names the choice explicitly.

    Interpolation: INTER_CUBIC when upsampling, INTER_AREA when downsampling. INTER_AREA does
    not degenerate to NEAREST on general upsampling (measured NCC(AREA, LINEAR) = 0.999), but it
    DOES at integer scale factors -- at target 42 from a 21px source it is 100% NEAREST. Cubic
    removes that special case. Cubic undershoots produce negative lobes (up to ~3.9% of mass); a
    PSF cannot be negative, so we clip before renormalising.

CAUTION -- cv2.filter2D computes CORRELATION, not convolution. The operator the dataset actually
applies is therefore flip(k) applied N times, i.e. flip(k*^N). Second-order statistics are blind
to this (a point reflection maps Cov to itself, so sigma, theta and rho are all unchanged), which
is why the measurements here are exact for the realized degradation. It matters the moment anyone
reimplements the blur with a true convolution (e.g. scipy.signal.fftconvolve): that path must
flip the kernel to reproduce the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

SIGMA_CONVENTION = "sqrt(trace(Cov)/2)"

# Directory holding the measured PSF bank. The bank itself is NOT distributed (proprietary,
# defense-related); point this at your own bank of square single-channel PSF PNGs. Override
# without editing the source by exporting MAROB_KERNEL_ROOT.
KERNEL_ROOT = Path(os.environ.get("MAROB_KERNEL_ROOT", "data/vibration_kernel"))
SRC_DIR = "21by21"  # sub-dir holding the largest bank entries; every kernel is resampled from it
SRC_PX = 21


def load_kernel_continuous(path: Path, target: int) -> np.ndarray:
    """Load a bank kernel PNG resampled to an arbitrary square target, sum-normalised."""
    k = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if k is None:
        raise FileNotFoundError(f"Failed to read kernel image: {path}")
    if k.ndim == 3:
        if k.shape[2] == 4:
            k = cv2.cvtColor(k, cv2.COLOR_BGRA2BGR)
        k = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
    k = k.astype(np.float64)
    if k.max() > 1.0:
        k = k / 255.0
    if k.shape[0] != target or k.shape[1] != target:
        interp = cv2.INTER_CUBIC if target > k.shape[0] else cv2.INTER_AREA
        k = cv2.resize(k, (target, target), interpolation=interp)
    k = np.clip(k, 0.0, None)  # cubic undershoot: a PSF cannot be negative
    total = float(k.sum())
    if total <= 1e-12:
        raise ValueError(f"Degenerate kernel after resample: {path} -> {target}")
    return k / total


def interp_name(target: int, src_px: int = SRC_PX) -> str:
    return "CUBIC" if target > src_px else "AREA"


def kernel_cov(k: np.ndarray) -> np.ndarray:
    """Second-moment (covariance) matrix of a sum-normalised kernel, in px^2."""
    h, w = k.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy = float((k * yy).sum())
    cx = float((k * xx).sum())
    dy, dx = yy - cy, xx - cx
    return np.array(
        [
            [float((k * dx * dx).sum()), float((k * dx * dy).sum())],
            [float((k * dx * dy).sum()), float((k * dy * dy).sum())],
        ]
    )


def cov_after_passes(cov: np.ndarray, passes: int) -> np.ndarray:
    """Covariance of k*^N.

    Self-convolution adds covariances: Cov(k*^N) = N * Cov(k). This is exact, not an
    approximation -- verified to 7.5e-15 relative error for N = 1..6. So the realized second
    moment never needs the N-fold convolution to be materialised.
    """
    return passes * cov


def cov_to_stats(cov: np.ndarray) -> Tuple[float, float, float]:
    """(sigma_px, theta_deg mod 180, rho) from a covariance matrix.

    theta is the major-eigenvector angle. It is AXIAL: a blur along 10 deg and one along 190 deg
    are the same degradation, so the angle is reported mod 180.
    """
    sigma = float(np.sqrt(np.trace(cov) / 2.0))
    evals, evecs = np.linalg.eigh(cov)
    lam_min, lam_max = float(evals[0]), float(evals[1])
    major = evecs[:, 1]
    theta = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    rho = float(np.sqrt(lam_max / max(lam_min, 1e-12)))
    return sigma, theta, rho


def realized_stats(path: Path, target: int, passes: int) -> Dict[str, float]:
    """Measured second-order statistics of the effective kernel k*^N actually applied."""
    k = load_kernel_continuous(path, target)
    cov = cov_after_passes(kernel_cov(k), passes)
    sigma, theta, rho = cov_to_stats(cov)
    return {"sigma": sigma, "theta": theta, "rho": rho}


def build_sigma_grid(
    kernel_name: str,
    targets: range,
    passes_choices: Tuple[int, ...],
    kernel_root: Path = KERNEL_ROOT,
) -> list:
    """Every reachable (sigma, target, passes) for one kernel, sorted by sigma.

    sigma is MEASURED per (target, passes), never predicted by c_k * t * sqrt(N) -- that closed
    form carries up to 1.94% error because rms is only approximately linear in the resample
    target, and the strata here are 1.4px wide.
    """
    path = kernel_root / SRC_DIR / kernel_name
    grid = []
    for t in targets:
        base_cov = kernel_cov(load_kernel_continuous(path, t))
        for n in passes_choices:
            sigma, _, _ = cov_to_stats(cov_after_passes(base_cov, n))
            grid.append((sigma, int(t), int(n)))
    grid.sort()
    return grid
