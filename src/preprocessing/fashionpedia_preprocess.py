from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import DEFAULT_CONFIG_DIR, ensure_parent, load_configs
from src.preprocessing.common import MultiGpuVaeEncoder, load_vae, random_mask
from src.preprocessing.progress import CountProgress


UPPER = {0, 1, 2, 3, 4, 5, 9, 10, 11, 12}
LOWER = {6, 7, 8, 20}


def read_yolo(path: str | Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                values = line.split()
                if len(values) >= 5:
                    boxes.append((int(values[0]), *map(float, values[1:5])))
    except Exception:
        pass
    return boxes


def load_image_and_boxes(img_path: str, label_path: str, height: int, width: int):
    """Apply the same center crop and resize to an image and its YOLO boxes."""
    image = Image.open(img_path).convert("RGB")
    original_w, original_h = image.size
    target_ratio = width / height
    if original_w / original_h > target_ratio:
        crop_w, crop_h = int(round(original_h * target_ratio)), original_h
        offset_x, offset_y = (original_w - crop_w) // 2, 0
    else:
        crop_w, crop_h = original_w, int(round(original_w / target_ratio))
        offset_x, offset_y = 0, (original_h - crop_h) // 2
    image = image.crop(
        (offset_x, offset_y, offset_x + crop_w, offset_y + crop_h)
    ).resize((width, height), Image.BICUBIC)
    image_array = np.asarray(image, np.float32) / 127.5 - 1.0
    resize_scale = width / crop_w

    boxes = []
    for class_id, center_x, center_y, box_w, box_h in read_yolo(label_path):
        x1 = ((center_x - box_w / 2) * original_w - offset_x) * resize_scale
        x2 = ((center_x + box_w / 2) * original_w - offset_x) * resize_scale
        y1 = ((center_y - box_h / 2) * original_h - offset_y) * resize_scale
        y2 = ((center_y + box_h / 2) * original_h - offset_y) * resize_scale
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 - x1 > 4 and y2 - y1 > 4:
            boxes.append((class_id, x1, y1, x2, y2))
    return image_array, boxes


def mask_from_boxes(boxes, height: int, width: int, rng: np.random.Generator):
    candidates = [box for box in boxes if box[0] in UPPER] or [
        box for box in boxes if box[0] in LOWER
    ]
    if not candidates:
        return random_mask(height, width, rng), False
    candidates.sort(key=lambda box: -(box[3] - box[1]) * (box[4] - box[2]))
    _, x1, y1, x2, y2 = candidates[0]
    box_w, box_h = x2 - x1, y2 - y1
    x1 += rng.uniform(-0.05, 0.05) * box_w
    x2 += rng.uniform(-0.05, 0.05) * box_w
    y1 += rng.uniform(-0.05, 0.05) * box_h
    y2 += rng.uniform(-0.05, 0.05) * box_h
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return random_mask(height, width, rng), False

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    radius_x, radius_y = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)
    mask = np.zeros((height, width), np.uint8)
    choice = rng.random()
    if choice < 0.45:
        mask[y1:y2, x1:x2] = 1
    elif choice < 0.75:
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1, -1)
    else:
        count = int(rng.integers(8, 14))
        theta = np.sort(rng.uniform(0, 2 * np.pi, count))
        radius = rng.uniform(0.75, 1.05, count)
        points = np.stack(
            [
                center_x + radius * radius_x * np.cos(theta),
                center_y + radius * radius_y * np.sin(theta),
            ],
            axis=1,
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    if rng.random() < 0.35:
        kernel_size = int(rng.integers(3, 11))
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )
    return mask.astype(np.float32), True


def list_pairs(root: str, split: str, min_area: float, limit: int = 0, workers: int = 16):
    image_dir = Path(root) / "images" / split
    label_dir = Path(root) / "labels" / split
    images = sorted(glob.glob(str(image_dir / "*.jpg")) + glob.glob(str(image_dir / "*.png")))
    print(f"{split}: {len(images)} images, scanning labels...")

    def valid_pair(path: str):
        label_path = label_dir / f"{Path(path).stem}.txt"
        if not label_path.exists():
            return None
        for class_id, _, _, box_w, box_h in read_yolo(label_path):
            if class_id in UPPER and box_w * box_h >= min_area:
                return path, str(label_path)
        return None

    with ThreadPoolExecutor(workers) as executor:
        pairs = [result for result in executor.map(valid_pair, images) if result]
    print(f"  upper-body bbox (area >= {min_area}): {len(pairs)}")
    if limit and len(pairs) > limit:
        indices = np.random.default_rng(0).choice(len(pairs), limit, replace=False)
        pairs = [pairs[index] for index in sorted(indices)]
        print(f"  sampled: {len(pairs)}")
    return pairs


class RawDS(Dataset):
    def __init__(self, pairs, height: int, width: int, masks: int, seed: int = 0):
        self.pairs = pairs
        self.height = height
        self.width = width
        self.masks = masks
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        cv2.setNumThreads(0)
        rng = np.random.default_rng(self.seed * 1000003 + index)
        image, boxes = load_image_and_boxes(
            *self.pairs[index], self.height, self.width
        )
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        masks, hits = [], 0
        for _ in range(2 * self.masks):
            mask, used_bbox = mask_from_boxes(boxes, self.height, self.width, rng)
            masks.append(mask)
            hits += int(used_bbox)
        return image_tensor, torch.from_numpy(np.stack(masks)), hits / (2 * self.masks)


def build_cache(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch: int = 24,
    chunk: int = 32,
    workers: int = 4,
    seed: int = 0,
    output: str | None = None,
    progress_every: int | None = None,
):
    configs = load_configs(config_dir)
    common = configs["dataset"]["common"]
    settings = configs["dataset"]["datasets"]["fashionpedia"]
    output = output or configs["path"]["paths"]["cache"]["fashionpedia"]
    output_path = ensure_parent(output)
    height, width = common["image_size"]
    masks = settings["masks_per_image"]
    scale = common["vae_scale"]
    if progress_every is None:
        progress_every = common["progress_every"]
    pairs = list_pairs(
        settings["root"],
        settings["split"],
        settings["min_area"],
        settings["image_limit"],
    )
    vae_settings = configs["model"]["vae"]
    vae = load_vae(
        vae_settings["pretrained_model_name_or_path"], vae_settings["dtype"]
    )
    encoder = MultiGpuVaeEncoder(vae, scale, chunk)
    print(
        f"{len(pairs)} imgs | {encoder.gpu_count} GPU | "
        f"~{len(pairs) * 2 * (1 + masks) * 4 * (height // 8) * (width // 8) * 2 / 1e6:.0f} MB"
    )
    loader = DataLoader(
        RawDS(pairs, height, width, masks, seed),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    full, agnostic, mask_tensors = [], [], []
    hit_sum = hit_count = 0
    start_time = time.time()
    processed = 0
    progress = CountProgress(len(pairs), progress_every)
    for images, pixel_masks, hits in loader:
        batch_size = images.shape[0]
        processed += batch_size
        hit_sum += float(hits.sum())
        hit_count += batch_size
        images = torch.cat([images, torch.flip(images, dims=[3])])
        pixel_masks = torch.cat(
            [pixel_masks[:, :masks], torch.flip(pixel_masks[:, masks:], dims=[3])]
        )
        combined = torch.cat(
            [images]
            + [images * (1 - pixel_masks[:, index : index + 1]) for index in range(masks)]
        )
        latent = encoder.encode(combined)
        doubled_batch = 2 * batch_size
        full.append(latent[:doubled_batch])
        agnostic.append(
            latent[doubled_batch:]
            .view(masks, doubled_batch, *latent.shape[1:])
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        mask_tensors.append(
            F.max_pool2d(
                pixel_masks.reshape(doubled_batch * masks, 1, height, width), 8
            )
            .view(doubled_batch, masks, 1, height // 8, width // 8)
            .to(torch.uint8)
        )
        progress.update(
            processed,
            suffix=f"bbox usage {hit_sum / hit_count:.2f}",
        )

    z_full = torch.cat(full)
    z_agnostic = torch.cat(agnostic)
    masks_out = torch.cat(mask_tensors)
    print("z1", tuple(z_full.shape), "| agn", tuple(z_agnostic.shape), "| mask", tuple(masks_out.shape))
    print("channel std:", [round(value, 3) for value in z_full.float().std(dim=(0, 2, 3)).tolist()])
    print("mask coverage:", round(float(masks_out.float().mean()), 3))
    print("bbox usage:", round(hit_sum / hit_count, 3))
    torch.save(
        {"z_full": z_full, "z_agn": z_agnostic, "masks": masks_out, "scale": scale},
        output_path,
    )
    print(f"saved {output_path.stat().st_size / 1e6:.0f} MB in {time.time() - start_time:.0f}s: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build the Fashionpedia VAE latent cache.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument(
        "--progress-every",
        type=int,
        help="Print completed/total progress at this image interval (default: dataset.yaml).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = load_configs(args.config_dir)
    output = Path(args.output or configs["path"]["paths"]["cache"]["fashionpedia"])
    if output.is_file() and not args.force:
        print(f"fashionpedia cache exists ({output.stat().st_size / 1e6:.0f} MB), skip: {output}")
        return
    build_cache(
        config_dir=args.config_dir,
        batch=args.batch,
        chunk=args.chunk,
        workers=args.workers,
        seed=args.seed,
        output=args.output,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
