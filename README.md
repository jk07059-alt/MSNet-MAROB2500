# Maritime-Aware Image Deblurring: Benchmark, Information Limit, and a Lightweight Directional Context Module

Official **code** release for our IEEE Access submission. (The manuscript itself is not in this
repository; it will be linked here — DOI / preprint — upon publication.)

**Wonjong Lee¹, Ilhan Sin¹, Eunjae Ha², Heejo Woo², Sungho Kim¹** (corresponding: Sungho Kim)
¹ Department of Robotics and AI Convergence, Yeungnam University
² AI Team, Hanwha Systems Co., Ltd.

---

## TL;DR

Motion blur in maritime electro-optical (EO) surveillance comes from platform vibration (global) and
fast-moving vessels (local). We contribute:

1. **MAROB-2500** — a maritime deblurring benchmark whose blur is synthesized from **measured**
   platform-vibration kernels, with a severity-controllable object-vs-background blur model and a dedicated
   **object-region (vessel) metric**.
2. **Severity augmentation** — an on-the-fly re-blurring regularizer that lifts a strong baseline by
   **+0.94 dB overall / +1.61 dB on the vessel** (same-eval).
3. **An information limit** — heavy object motion pushes ~**74%** of the vessel's spatial frequencies below
   the 8-bit quantization floor, so the best oracle non-blind deconvolution (18.5 dB, after a regularization sweep)
   scores below both the blurred input (19.9 dB) and a blind network (23.3 dB).
5. **MSNet** (Maritime Scene Network) — a lightweight directional-context module that matches the strong
   backbone and surpasses every Transformer/multi-scale baseline on overall & background PSNR at ~10× fewer
   FLOPs. On the object region it, like every method, cannot exceed the physical ceiling (no robust gain
   across 5 seeds) — exactly as the information limit predicts.

## Main result (held-out MAROB-2500 test split)

| Method | Params (M) | Overall | SSIM | Background | Vessel |
|---|---|---|---|---|---|
| Blurred input | — | 28.45 | — | 29.95 | 19.93 |
| Restormer | 26.1 | 29.24 | 0.8547 | 30.48 | 21.84 |
| MIMO-UNet+ | 16.1 | 30.13 | 0.8662 | 31.17 | **23.83** |
| Stripformer | 19.7 | 30.03 | 0.8652 | 31.21 | 23.05 |
| NAFNet + aug (baseline) | 17.1 | 30.92 | 0.8753 | 32.21 | 23.34 |
| **MSNet (ours)** | 16.8 | **30.98** | **0.8760** | **32.24** | 23.45 |

Differences among the leading convolutional entries are within run-to-run noise (≈0.05 dB, 5 seeds) and
should be read as ties. All learned methods sit in a narrow 22–24 dB band on the vessel — the recoverability
ceiling.

---

## Repository structure

```
.
├── setup.sh                    # one-shot: fetch NAFNet, install deps, wire files in
├── requirements.txt            # pinned, verified env (Python 3.10 / CUDA 12.1)
├── smoke_test.py               # offline check: blur math + MSNet/NAFNet forward (no data)
├── code/
│   ├── archs/msnet_arch.py     # MSNet architecture (NAFNet backbone + MAR blocks)
│   ├── archs/NAFNet_arch.py    # backbone dependency
│   ├── data/maritime_dataset.py      # blur formation + on-the-fly severity augmentation
│   ├── data/maritime_kernel.py       # measured-kernel loader + covariance/sigma statistics
│   └── ship_mask_blur_pipeline.py    # masked bg/vessel composite + kernel helpers
├── configs/                    # training configs (BasicSR YAML) — see mapping below
├── scripts/
│   ├── eval_testsplit.py       # 3-region eval on the held-out test split
│   ├── seed_eval_5seed.py      # 5-seed robustness eval (baseline vs MSNet)
│   └── eval_lpips_detection.py # LPIPS (AlexNet) + YOLOv8 vessel-detection rate
├── results/
│   ├── seed_eval_5seed_result.txt      # 5-seed means ± std, GAIN (MSNet − baseline)
│   ├── eval_lpips_detection_result.txt # LPIPS + detection rate (test split)
│   └── severity_data_test.csv          # per-severity PSNR behind the severity-curve figure
└── README.md
```

> The code is built on **BasicSR / NAFNet**. The files here are the paper-specific additions
> (architecture, blur/augmentation dataset, region-metric model wrapper, eval scripts), which are
> auto-discovered once dropped into a BasicSR/NAFNet tree.

## Installation

The architecture, dataset, and model wrapper plug into the BasicSR framework (as shipped with
NAFNet). `setup.sh` fetches a NAFNet tree, installs the pinned dependencies, and copies the
paper-specific files into the right places so BasicSR auto-discovers them:

```bash
bash setup.sh                           # clones NAFNet into ./third_party/NAFNet
# or reuse an existing checkout:
NAFNET_ROOT=/path/to/NAFNet bash setup.sh
```

Then verify the install **without any data** (synthesizes a PSF to exercise the blur model, and
runs a forward pass through MSNet and NAFNet):

```bash
NAFNET_ROOT=./third_party/NAFNet python smoke_test.py
# expected: Part A PASS (sigma/theta/rho recovered), Part B PASS (16.84M / 17.11M), SMOKE TEST PASS
```

Dependencies are pinned in `requirements.txt`; **torch 2.1.0 + numpy<2** is required (torch 2.1 is
built against the numpy 1.x ABI). GPU inference/training needs CUDA 12.1 wheels (installed by
`setup.sh`).

## Config → paper mapping

| Config | Role in paper |
|---|---|
| `R2-NAFNet-aug-hardevalv3-200k` | NAFNet + severity aug. — **baseline** (Tables main / aug / ablation) |
| `R25-NAFNet-NOaug-hardevalv3-200k` | NAFNet **without** augmentation (aug ablation) |
| `R26-MSNet-FIXED-hardevalv3-200k` | **MSNet** (Table main; horizontal 1×65 in ablation) |
| `R27-MSNet-vertical-FIXED-200k` | vertical strip (ablation) |
| `R28-MSNet-iso8x8-FIXED-200k` | isotropic 8×8 (ablation) |
| `R29-MSNet-multiscale-FIXED-200k` | multi-scale strips (ablation) |
| `R2-NAFNet-aug-seed{1..5}` / `R26-MSNet-seed{1..5}` | 5-seed robustness study |
| `X6-Restormer-fair-maritime-200k` / `S2-MIMO-aug-hardevalv3-200k` | external baselines (Table main) |
| `X1..X5-*-gopro-200k` | GoPro cross-dataset no-regression study (NAFNet / MSNet / MIMO / Stripformer) |

> Numbering (`R#`, `S#`, `X#`) is internal experiment IDs; names are kept as-is for traceability.

## Reproducing the numbers

Configs and scripts use **relative paths**; point them at your data through environment variables
(defaults in parentheses) rather than editing the sources:

| Variable | Used by | Default |
|---|---|---|
| `NAFNET_ROOT` | eval scripts (the BasicSR tree to run inside) | current directory |
| `MAROB_ROOT` | eval scripts (your benchmark copy) | `datasets/MAROB2500` |
| `MAROB_KERNEL_ROOT` | blur model (`maritime_kernel.py`) | `data/vibration_kernel` |
| `YOLO_WEIGHTS` | detection eval | `yolov8n.pt` (auto-downloaded) |

Configs expect the benchmark at `datasets/MAROB2500/{train,valid,test}/{blur,sharp}` and
`datasets/MAROB2500/split_manifest.csv` (GoPro configs: `datasets/GoPro/...`); adjust the
`dataroot_*`/`manifest` lines or symlink your data to those paths.

- **Training:** BasicSR-style, `python basicsr/train.py -opt configs/<name>.yml`. 200k iters, 256×256 crops,
  PSNR loss, AdamW (lr 1e-3, cosine→1e-7). NAFNet/MSNet are parameter-matched (16.8 M vs 17.1 M).
- **Evaluation:** `scripts/eval_testsplit.py` — full-resolution 1920×1080 single-pass inference with
  **reflect pad-16** (÷16 for 4 downsamples), 3-region PSNR (overall / background / vessel-mask-dilated-15).
- **5-seed robustness:** `scripts/seed_eval_5seed.py`; results in `results/seed_eval_5seed_result.txt`.
- **Perceptual + detection:** `scripts/eval_lpips_detection.py` — LPIPS (AlexNet) and YOLOv8 vessel-detection
  rate; results in `results/eval_lpips_detection_result.txt`.

## Data availability

The maritime imagery and the measured kernel bank are **proprietary, defense-related data** and **cannot be
publicly released** owing to security and data-provider restrictions. The blur formation model and the
severity-augmentation model are **fully specified** in the manuscript and in `code/` — the kernel loader and
statistics in `code/data/maritime_kernel.py`, the masked bg/vessel compositing in `ship_mask_blur_pipeline.py`, and
the on-the-fly re-blurring in `code/data/maritime_dataset.py`. To run the pipeline on **your own** imagery,
place a bank of square single-channel PSF PNGs under a `21by21/` sub-directory and point the loader at it:

```bash
export MAROB_KERNEL_ROOT=/path/to/your/kernel_bank   # expects <root>/21by21/*.png
```

So the pipeline is reproducible on other maritime data (though not re-scored on our exact, non-released data).

### Building the benchmark on your own imagery

Blur is applied **per region**, so the first step is a **vessel mask** for every image — this is a
prerequisite, not an optional extra. `ship_mask_blur_pipeline.py` produces the masks and then the
masked background/vessel/mixed blur. Two ways to get the masks:

- **`--mask-mode annotation`** (default) — you already have COCO **segmentation** polygons; the mask
  is decoded directly. No extra dependency.
- **`--mask-mode sam2_bbox`** — you only have **ROI boxes**; **SAM2** turns each box into a mask (this
  is how our masks were produced). Needs SAM2 installed plus `--sam2-config`/`--sam2-checkpoint`
  (see `requirements.txt`; weights are downloaded from the SAM2 repo).

```bash
# ROI-box → SAM2 mask → masked measured-kernel blur
python code/ship_mask_blur_pipeline.py \
    --dataset-root   data/maritime_sharp \
    --annotations    data/maritime_sharp/annotations_coco.zip \
    --output-root    data/ship_blur_outputs \
    --vibration-root data/vibration_kernel \
    --mask-mode      sam2_bbox \
    --sam2-config    <sam2_cfg.yaml> --sam2-checkpoint <sam2_weights.pt>
# COCO-segmentation path: drop --mask-mode (defaults to `annotation`) and the --sam2-* args.
```

`--vibration-root` is the measured PSF bank (the non-released proprietary data); point it at your own
bank of PSF PNGs. The on-the-fly re-blurring used at **training** time (`maritime_dataset.py`) reads
the same bank via `MAROB_KERNEL_ROOT`.

## Acknowledgment / Funding

Supported by the Korea Research Institute for Defense Technology Planning and Advancement (KRIT) Grant
funded by the Korea Government (DAPA) in 2023 (KRIT-CT-23-045, Integrated Technology for Intelligent
Electro-Optical Payload Systems).

## Citation

```bibtex
@article{lee2026maritime,
  title   = {Maritime-Aware Image Deblurring: Benchmark, Information Limit,
             and a Lightweight Directional Context Module},
  author  = {Lee, Wonjong and Sin, Ilhan and Ha, Eunjae and Woo, Heejo and Kim, Sungho},
  journal = {IEEE Access},
  year    = {2026},
  note    = {to appear / under review}
}
```

## License

The code is released under the **MIT License** (see `LICENSE`). The dataset is **not** released
(see *Data availability*). The backbone builds on BasicSR / NAFNet — please also honor their
upstream licenses.
