#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the MSNet / MAROB-2500 authors. See the LICENSE file.
import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_sizes(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_pass_pairs(text: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(
                f"Invalid pass pair '{item}'. Use comma-separated values like '1x1,2x2,2x3,3x3'."
            )
        bg_text, ship_text = token.split("x", 1)
        try:
            bg_passes = int(bg_text)
            ship_passes = int(ship_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid pass pair '{item}'. Background/ship passes must be integers."
            ) from exc
        if bg_passes <= 0 or ship_passes <= 0:
            raise ValueError(f"Invalid pass pair '{item}'. Pass counts must be >= 1.")
        if ship_passes < bg_passes:
            raise ValueError(
                f"Invalid pass pair '{item}'. Ship passes must be >= background passes."
            )
        pairs.append((bg_passes, ship_passes))
    return pairs


def list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([path for path in folder.iterdir() if path.suffix.lower() in IMG_EXTS])


def shuffle_with_seed(items: Sequence[Path], base_seed: int, salt: str) -> List[Path]:
    rng = random.Random(base_seed ^ (hash(salt) & 0x7FFFFFFF))
    out = list(items)
    rng.shuffle(out)
    return out


def normalize_kernel(kernel: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if kernel.ndim == 3:
        if kernel.shape[2] == 4:
            kernel = cv2.cvtColor(kernel, cv2.COLOR_BGRA2BGR)
        kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
    kernel = kernel.astype(np.float32)
    if kernel.max() > 1.0:
        kernel = kernel / 255.0
    h, w = target_hw
    if (kernel.shape[0], kernel.shape[1]) != (h, w):
        kernel = cv2.resize(kernel, (w, h), interpolation=cv2.INTER_AREA)
    total = float(kernel.sum())
    if total <= 1e-12:
        out = np.zeros((h, w), dtype=np.float32)
        out[h // 2, w // 2] = 1.0
        return out
    return kernel / total


def load_kernel_from_image(path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    kernel = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if kernel is None:
        raise FileNotFoundError(f"Failed to read kernel image: {path}")
    return normalize_kernel(kernel, target_hw)


def convolve_image_with_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    channels = cv2.split(image)
    conv_channels = [
        cv2.filter2D(channel, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
        for channel in channels
    ]
    return cv2.merge(conv_channels)


def convolve_image_multi_pass(image: np.ndarray, kernel: np.ndarray, num_passes: int) -> np.ndarray:
    out = image
    for _ in range(max(1, num_passes)):
        out = convolve_image_with_kernel(out, kernel)
    return out


def choose_kernel(kernel_dir: Path, seed: int, salt: str) -> Path:
    kernels = list_images(kernel_dir)
    if not kernels:
        raise FileNotFoundError(f"No kernel images found in {kernel_dir}")
    return shuffle_with_seed(kernels, seed, salt=salt)[0]


def decode_uncompressed_rle(rle: Dict[str, object]) -> np.ndarray:
    counts = rle.get("counts")
    size = rle.get("size")
    if not isinstance(counts, list) or not isinstance(size, list) or len(size) != 2:
        raise ValueError("Unsupported RLE format. Expected uncompressed COCO RLE with list counts.")
    height, width = int(size[0]), int(size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    index = 0
    value = 0
    for run in counts:
        run = int(run)
        if run < 0:
            raise ValueError("RLE counts must be non-negative.")
        if value == 1 and run > 0:
            flat[index:index + run] = 1
        index += run
        value = 1 - value
    if index != flat.size:
        raise ValueError(f"Decoded RLE length mismatch: expected {flat.size}, got {index}")
    return flat.reshape((height, width), order="F")


def polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def decode_segmentation(segmentation: object, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        return decode_uncompressed_rle(segmentation)
    if isinstance(segmentation, list):
        return polygon_to_mask(segmentation, height, width)
    raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")


def feather_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = mask.astype(np.float32)
    if feather_px <= 0:
        return alpha
    ksize = max(3, feather_px * 2 + 1)
    alpha = cv2.GaussianBlur(alpha, (ksize, ksize), sigmaX=0, sigmaY=0)
    return np.clip(alpha, 0.0, 1.0)


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if dilate_px <= 0:
        return binary
    ksize = max(3, dilate_px * 2 + 1)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def blend_masked_region(source: np.ndarray, blurred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if source.ndim == 2:
        out = source.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(source.dtype)
    alpha3 = alpha[..., None]
    out = source.astype(np.float32) * (1.0 - alpha3) + blurred.astype(np.float32) * alpha3
    return np.clip(out, 0, 255).astype(source.dtype)


def make_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = image.copy()
    color_img = np.zeros_like(image)
    color_img[:, :] = color
    overlay_mask = (mask > 0).astype(np.float32)[..., None]
    out = overlay.astype(np.float32) * (1.0 - overlay_mask * alpha) + color_img.astype(np.float32) * (overlay_mask * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_sam2_config(model_cfg: str) -> str:
    cfg = model_cfg.strip()
    marker = "/sam2/configs/"
    if marker in cfg:
        return "configs/" + cfg.split(marker, 1)[1]
    if cfg.startswith("configs/"):
        return cfg
    if cfg.endswith(".yaml") and "sam2.1/" in cfg:
        return "configs/" + cfg.split("sam2.1/", 1)[1]
    return cfg


def build_sam2_predictor(model_cfg: str, checkpoint: str, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. Install it from https://github.com/facebookresearch/sam2 "
            "and provide --sam2-config and --sam2-checkpoint."
        ) from exc
    model_cfg = normalize_sam2_config(model_cfg)
    model = build_sam2(model_cfg, checkpoint, device=device)
    model = model.to(device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def predict_union_mask_with_sam2(
    predictor,
    image_bgr: np.ndarray,
    image_info: Dict[str, object],
    annotations: List[Dict[str, object]],
) -> np.ndarray:
    height = int(image_info["height"])
    width = int(image_info["width"])
    union = np.zeros((height, width), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if torch.cuda.is_available():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                if masks is None or len(masks) == 0:
                    continue
                union |= (masks[0] > 0).astype(np.uint8)
    else:
        with torch.inference_mode():
            predictor.set_image(image_rgb)
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                if masks is None or len(masks) == 0:
                    continue
                union |= (masks[0] > 0).astype(np.uint8)
    return union


def resolve_dataset_root(dataset_root: Path, image_rel_path: str) -> Path:
    image_path = dataset_root / image_rel_path
    if image_path.exists():
        return image_path
    alt = dataset_root / Path(image_rel_path).name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Image not found for relative path: {image_rel_path}")


def load_coco_json(annotation_path: Path) -> Dict[str, object]:
    if annotation_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(annotation_path, "r") as zf:
            json_names = [name for name in zf.namelist() if name.endswith(".json")]
            if not json_names:
                raise FileNotFoundError(f"No json annotation found inside zip: {annotation_path}")
            return json.loads(zf.read(json_names[0]))
    with open(annotation_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_coco_groups(annotation_path: Path) -> Tuple[List[Dict[str, object]], Dict[int, List[Dict[str, object]]]]:
    data = load_coco_json(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[int(ann["image_id"])].append(ann)
    return images, grouped


def select_kernel_specs(args) -> List[Tuple[str, str, Path]]:
    specs: List[Tuple[str, str, Path]] = []
    for size_label in parse_sizes(args.vibration_sizes):
        specs.append(("vibration", size_label, Path(args.vibration_root) / size_label))
    for size_label in parse_sizes(args.random_sizes):
        specs.append(("random", size_label, Path(args.random_root) / size_label))
    if not specs:
        raise ValueError("No blur kernel sizes were selected.")
    return specs


def size_label_to_int(size_label: str) -> int:
    return int(size_label.lower().split("by")[0])


def stable_seed_from_text(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_common_kernel_maps(kernel_root: Path, size_labels: List[str]) -> Tuple[Dict[str, Dict[str, Path]], List[str]]:
    kernel_maps: Dict[str, Dict[str, Path]] = {}
    common_names = None
    for size_label in size_labels:
        paths = list_images(kernel_root / size_label)
        if not paths:
            raise FileNotFoundError(f"No kernel images found in {kernel_root / size_label}")
        kernel_maps[size_label] = {path.name: path for path in paths}
        names = set(kernel_maps[size_label].keys())
        common_names = names if common_names is None else (common_names & names)
    if not common_names:
        raise ValueError(
            f"No shared kernel filenames exist across sizes: {', '.join(size_labels)}. "
            "Cannot guarantee direction-matched mixed blur."
        )
    return kernel_maps, sorted(common_names)


def build_paired_vibration_plan(args) -> Dict[str, object]:
    bg_sizes = parse_sizes(args.paired_bg_sizes)
    ship_sizes = parse_sizes(args.paired_ship_sizes)
    pass_pairs = parse_pass_pairs(args.paired_pass_pairs)
    if not bg_sizes or not ship_sizes:
        raise ValueError("Paired mixed mode requires non-empty --paired-bg-sizes and --paired-ship-sizes.")
    if not pass_pairs:
        raise ValueError("Paired mixed mode requires non-empty --paired-pass-pairs.")
    required_sizes = sorted(set(bg_sizes + ship_sizes), key=size_label_to_int)
    kernel_maps, common_names = build_common_kernel_maps(Path(args.vibration_root), required_sizes)

    valid_pairs: List[Tuple[str, str]] = []
    for bg_size in bg_sizes:
        for ship_size in ship_sizes:
            if size_label_to_int(ship_size) >= size_label_to_int(bg_size):
                valid_pairs.append((bg_size, ship_size))
    if not valid_pairs:
        raise ValueError("No valid paired mixed combinations found where ship kernel size >= background kernel size.")

    return {
        "kernel_maps": kernel_maps,
        "common_names": common_names,
        "valid_pairs": valid_pairs,
        "pass_pairs": pass_pairs,
    }


def choose_paired_vibration_kernels(
    plan: Dict[str, object],
    image_rel_path: str,
    seed: int,
) -> Tuple[str, str, int, int, str, Path, Path]:
    rng = random.Random(stable_seed_from_text(seed, image_rel_path))
    bg_size, ship_size = rng.choice(plan["valid_pairs"])
    bg_passes, ship_passes = rng.choice(plan["pass_pairs"])
    kernel_name = rng.choice(plan["common_names"])
    kernel_maps = plan["kernel_maps"]
    bg_kernel_path = kernel_maps[bg_size][kernel_name]
    ship_kernel_path = kernel_maps[ship_size][kernel_name]
    return bg_size, ship_size, bg_passes, ship_passes, kernel_name, bg_kernel_path, ship_kernel_path


def save_mask(mask: np.ndarray, path: Path) -> None:
    mask_img = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), mask_img):
        raise RuntimeError(f"Failed to save mask: {path}")


def save_image(image: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def invert_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - alpha, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ship masks from COCO annotations or SAM2 and save ship-blur, background-blur, mixed-blur, and final full-blur images."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/maritime_sharp",
        help="Root folder containing part1/sharp, part2/sharp, and annotations.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/maritime_sharp/annotations_coco.zip",
        help="COCO-style annotation json or zip exported from Label Studio.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/ship_blur_outputs",
        help="Directory where masks, overlays, and both blur variants will be written.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["annotation", "sam2_bbox"],
        default="annotation",
        help="annotation: use existing segmentation masks. sam2_bbox: use annotation bboxes as SAM2 prompts.",
    )
    parser.add_argument("--sam2-config", type=str, default="", help="SAM2 model config path.")
    parser.add_argument("--sam2-checkpoint", type=str, default="", help="SAM2 checkpoint path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for SAM2 inference, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--vibration-root",
        type=str,
        default="data/vibration_kernel",
        help="Root directory of vibration kernels (the proprietary measured PSF bank; not distributed).",
    )
    parser.add_argument(
        "--random-root",
        type=str,
        default="data/random_kernel",
        help="Root directory of random kernels.",
    )
    parser.add_argument(
        "--vibration-sizes",
        type=str,
        default="3by3,5by5,7by7,9by9,11by11,13by13",
        help="Comma-separated vibration kernel sizes.",
    )
    parser.add_argument(
        "--random-sizes",
        type=str,
        default="15by15,17by17,19by19,21by21",
        help="Comma-separated random kernel sizes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic kernel selection.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N annotated images. 0 means all.")
    parser.add_argument("--mask-dilate-px", type=int, default=3, help="Mask dilation radius in pixels before feathering.")
    parser.add_argument("--feather-px", type=int, default=5, help="Mask feather radius in pixels.")
    parser.add_argument("--overlay-alpha", type=float, default=0.35, help="Overlay visualization alpha.")
    parser.add_argument(
        "--overlay-color-bgr",
        type=str,
        default="0,255,255",
        help="Mask overlay color in B,G,R order.",
    )
    parser.add_argument(
        "--paired-mixed-vibration",
        action="store_true",
        help="Create one final mixed-blur image per input using direction-matched vibration kernels.",
    )
    parser.add_argument(
        "--paired-bg-sizes",
        type=str,
        default="5by5,7by7,9by9",
        help="Background kernel sizes for paired mixed mode.",
    )
    parser.add_argument(
        "--paired-ship-sizes",
        type=str,
        default="9by9,11by11,13by13",
        help="Ship kernel sizes for paired mixed mode.",
    )
    parser.add_argument(
        "--paired-pass-pairs",
        type=str,
        default="1x1",
        help=(
            "Comma-separated total pass pairs for paired mixed mode, e.g. "
            "'1x1,2x2,2x3,3x3'. Duplicates act as weights."
        ),
    )
    parser.add_argument(
        "--paired-output-dirname",
        type=str,
        default="paired_mixed_blur",
        help="Flat output folder name for one-image-per-input paired mixed results.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    annotation_path = Path(args.annotations)
    output_root = Path(args.output_root)

    ensure_dir(output_root)
    mask_dir = output_root / "masks"
    overlay_dir = output_root / "overlays"
    ship_blur_dir = output_root / "ship_blur"
    background_blur_dir = output_root / "background_blur"
    mixed_blur_dir = output_root / "mixed_blur"
    full_blur_dir = output_root / "full_blur"
    paired_mixed_dir = output_root / args.paired_output_dirname
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)
    ensure_dir(ship_blur_dir)
    ensure_dir(background_blur_dir)
    ensure_dir(mixed_blur_dir)
    ensure_dir(full_blur_dir)
    if args.paired_mixed_vibration:
        ensure_dir(paired_mixed_dir)

    try:
        color_parts = [int(v.strip()) for v in args.overlay_color_bgr.split(",")]
        if len(color_parts) != 3:
            raise ValueError
        overlay_color = tuple(color_parts)
    except ValueError as exc:
        raise ValueError("--overlay-color-bgr must be three comma-separated integers, e.g. 0,255,255") from exc

    images, grouped_annotations = load_coco_groups(annotation_path)
    annotated_image_count = sum(1 for img in images if int(img["id"]) in grouped_annotations)
    process_images = list(images)
    if args.limit > 0:
        process_images = process_images[:args.limit]

    predictor = None
    if args.mask_mode == "sam2_bbox":
        if not args.sam2_config or not args.sam2_checkpoint:
            raise ValueError("--sam2-config and --sam2-checkpoint are required when --mask-mode sam2_bbox")
        predictor = build_sam2_predictor(args.sam2_config, args.sam2_checkpoint, args.device)

    kernel_specs = select_kernel_specs(args)
    manifest_rows: List[List[object]] = []
    paired_manifest_rows: List[List[object]] = []
    paired_plan = build_paired_vibration_plan(args) if args.paired_mixed_vibration else None

    processed_count = 0

    for image_info in process_images:
        image_id = int(image_info["id"])
        image_rel_path = str(image_info["file_name"])
        image_path = resolve_dataset_root(dataset_root, image_rel_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        anns = grouped_annotations.get(image_id, [])
        if not anns:
            union_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        elif args.mask_mode == "annotation":
            union_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for ann in anns:
                union_mask |= decode_segmentation(ann["segmentation"], image.shape[0], image.shape[1]).astype(np.uint8)
        else:
            union_mask = predict_union_mask_with_sam2(predictor, image, image_info, anns)

        has_ship = bool(anns) and union_mask.max() > 0
        expanded_mask = dilate_mask(union_mask, args.mask_dilate_px) if has_ship else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        alpha_ship = feather_mask(expanded_mask, args.feather_px) if has_ship else np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        alpha_background = invert_alpha(alpha_ship) if has_ship else np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        mask_path = mask_dir / f"{Path(image_rel_path).stem}_mask.png"
        overlay_path = overlay_dir / f"{Path(image_rel_path).stem}_overlay.png"
        save_mask(union_mask, mask_path)
        save_image(make_overlay(image, union_mask, overlay_color, args.overlay_alpha), overlay_path)
        processed_count += 1

        if paired_plan is not None:
            bg_size, ship_size, bg_passes, ship_passes, kernel_name, bg_kernel_path, ship_kernel_path = choose_paired_vibration_kernels(
                paired_plan, image_rel_path, args.seed
            )
            bg_kernel = load_kernel_from_image(bg_kernel_path, (size_label_to_int(bg_size), size_label_to_int(bg_size)))
            ship_kernel = load_kernel_from_image(ship_kernel_path, (size_label_to_int(ship_size), size_label_to_int(ship_size)))
            background_blur = convolve_image_multi_pass(image, bg_kernel, bg_passes)
            if has_ship:
                ship_blur_stronger = convolve_image_multi_pass(image, ship_kernel, ship_passes)
                paired_mixed = blend_masked_region(background_blur, ship_blur_stronger, alpha_ship)
            else:
                paired_mixed = background_blur
            paired_output_path = paired_mixed_dir / Path(image_rel_path).name
            save_image(paired_mixed, paired_output_path)
            paired_manifest_rows.append([
                image_id,
                image_rel_path,
                str(image_path),
                args.mask_mode,
                len(anns),
                int(has_ship),
                int(union_mask.sum()),
                str(mask_path),
                str(overlay_path),
                bg_size,
                ship_size,
                bg_passes,
                ship_passes,
                kernel_name,
                str(bg_kernel_path),
                str(ship_kernel_path),
                str(paired_output_path),
            ])
            continue

        for family, size_label, kernel_dir in kernel_specs:
            kernel_path = choose_kernel(kernel_dir, args.seed, f"{image_rel_path}:{family}:{size_label}")
            kh, kw = [int(v) for v in size_label.lower().split("by")]
            kernel = load_kernel_from_image(kernel_path, (kh, kw))
            fully_blurred = convolve_image_with_kernel(image, kernel)
            background_only_blur = blend_masked_region(image, fully_blurred, alpha_background)
            stronger_ship_blur = convolve_image_with_kernel(fully_blurred, kernel)
            mixed_blur = blend_masked_region(fully_blurred, stronger_ship_blur, alpha_ship)
            ship_out_path = ""
            mixed_out_path = ""
            full_out_path = ""

            background_family_out_dir = background_blur_dir / family / size_label
            mixed_family_out_dir = mixed_blur_dir / family / size_label
            full_family_out_dir = full_blur_dir / family / size_label
            ensure_dir(background_family_out_dir)
            ensure_dir(mixed_family_out_dir)
            ensure_dir(full_family_out_dir)
            if has_ship:
                ship_only_blur = blend_masked_region(image, fully_blurred, alpha_ship)
                ship_family_out_dir = ship_blur_dir / family / size_label
                ensure_dir(ship_family_out_dir)
                if family == "random":
                    ship_out_name = f"{Path(image_rel_path).stem}_ship_Ran_{size_label}.png"
                else:
                    ship_out_name = f"{Path(image_rel_path).stem}_ship_{size_label}.png"
                ship_out_path = str(ship_family_out_dir / ship_out_name)
                save_image(ship_only_blur, Path(ship_out_path))

            if family == "random":
                background_out_name = f"{Path(image_rel_path).stem}_bg_Ran_{size_label}.png"
                mixed_out_name = f"{Path(image_rel_path).stem}_mix_Ran_{size_label}.png"
                full_out_name = f"{Path(image_rel_path).stem}_full_Ran_{size_label}.png"
            else:
                background_out_name = f"{Path(image_rel_path).stem}_bg_{size_label}.png"
                mixed_out_name = f"{Path(image_rel_path).stem}_mix_{size_label}.png"
                full_out_name = f"{Path(image_rel_path).stem}_full_{size_label}.png"
            background_out_path = background_family_out_dir / background_out_name
            mixed_out_path = str(mixed_family_out_dir / mixed_out_name)
            full_out_path = str(full_family_out_dir / full_out_name)
            save_image(background_only_blur, background_out_path)
            save_image(mixed_blur, Path(mixed_out_path))
            save_image(fully_blurred, Path(full_out_path))

            manifest_rows.append([
                image_id,
                image_rel_path,
                str(image_path),
                args.mask_mode,
                len(anns),
                int(union_mask.sum()),
                str(mask_path),
                str(overlay_path),
                family,
                size_label,
                str(kernel_path),
                ship_out_path,
                str(background_out_path),
                mixed_out_path,
                full_out_path,
            ])

    if paired_plan is not None:
        manifest_path = output_root / "paired_mixed_manifest.csv"
        manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
        with open(manifest_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not manifest_exists:
                writer.writerow([
                    "image_id",
                    "image_rel_path",
                    "image_path",
                    "mask_mode",
                    "annotation_count",
                    "has_ship",
                    "mask_area_px",
                    "mask_path",
                    "overlay_path",
                    "background_kernel_size",
                    "ship_kernel_size",
                    "background_passes",
                    "ship_passes",
                    "shared_kernel_name",
                    "background_kernel_path",
                    "ship_kernel_path",
                    "paired_mixed_output_path",
                ])
            writer.writerows(paired_manifest_rows)
        total_outputs = len(paired_manifest_rows)
    else:
        manifest_path = output_root / "ship_blur_manifest.csv"
        manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
        with open(manifest_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not manifest_exists:
                writer.writerow([
                    "image_id",
                    "image_rel_path",
                    "image_path",
                    "mask_mode",
                    "annotation_count",
                    "mask_area_px",
                    "mask_path",
                    "overlay_path",
                    "blur_family",
                    "kernel_size",
                    "kernel_path",
                    "ship_blur_output_path",
                    "background_blur_output_path",
                    "mixed_blur_output_path",
                    "full_blur_output_path",
                ])
            writer.writerows(manifest_rows)
        total_outputs = len(manifest_rows)

    print(f"Total images in annotation export: {len(images)}")
    print(f"Annotated images in json: {annotated_image_count}")
    print(f"Processed images: {processed_count}")
    print(f"Blur outputs: {total_outputs}")
    print(f"Manifest: {manifest_path}")
    if args.mask_mode == "annotation":
        print("Mask source: existing COCO segmentation in instances_Train.json")
    else:
        print("Mask source: SAM2 prediction from annotation bbox prompts")
    if paired_plan is not None:
        print(f"Paired mixed outputs: {paired_mixed_dir}")
        print("Images without ship annotation are saved as background-only blur in the paired output folder.")
    else:
        print("Images without ship annotation are saved with background blur only.")


if __name__ == "__main__":
    main()
