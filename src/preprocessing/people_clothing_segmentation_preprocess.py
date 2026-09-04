from __future__ import annotations

import argparse
import csv
import glob
import os
import re
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
from src.preprocessing.common import MultiGpuVaeEncoder, load_vae
from src.preprocessing.progress import CountProgress


UPPER_NAMES = {
    "blazer", "blouse", "cardigan", "cape", "coat", "hoodie", "jacket",
    "jumper", "shirt", "sweater", "sweatshirt", "t-shirt", "top", "vest",
}
LOWER_NAMES = {"jeans", "leggings", "pants", "shorts", "skirt", "tights"}
FULL_NAMES = {"dress", "romper", "bodysuit", "suit", "swimwear"}
_SEG_CHECK = {"n": 0}


def find_labels_csv(root: str, configured_path: str | None = None) -> Path:
    candidates = [configured_path, str(Path(root) / "labels.csv")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("labels.csv was not found; set datasets.people_clothing_segmentation.labels_csv.")


def load_pcs_class_ids(root: str, labels_csv: str | None = None):
    path = find_labels_csv(root, labels_csv)
    name_to_id: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[0].strip().isdigit():
                name_to_id[row[1].strip().lower()] = int(row[0])

    def ids(names: set[str]) -> set[int]:
        output = {name_to_id[name] for name in names if name in name_to_id}
        missing = names - set(name_to_id)
        if missing:
            print(f"  [warn] classes absent from labels.csv and ignored: {sorted(missing)}")
        return output

    upper, lower, full = ids(UPPER_NAMES), ids(LOWER_NAMES), ids(FULL_NAMES)
    print(f"labels.csv: {path} ({len(name_to_id)} classes)")
    print(f"  UPPER {sorted(upper)}\n  LOWER {sorted(lower)}\n  FULL  {sorted(full)}")
    return upper, lower, full


def _num_key(path: str) -> int:
    matches = re.findall(r"\d+", os.path.basename(path))
    return int(matches[-1]) if matches else -1


def list_pcs_pairs(root: str):
    images = sorted(glob.glob(str(Path(root) / "png_images" / "IMAGES" / "*")), key=_num_key)
    masks = sorted(glob.glob(str(Path(root) / "png_masks" / "MASKS" / "*")), key=_num_key)
    if not images or not masks:
        raise FileNotFoundError(
            f"Expected {root}/png_images/IMAGES and {root}/png_masks/MASKS."
        )
    mask_map = {_num_key(path): path for path in masks}
    pairs = [(path, mask_map[_num_key(path)]) for path in images if _num_key(path) in mask_map]
    print(f"PCS: images {len(images)}, masks {len(masks)} -> matched {len(pairs)}")
    return pairs


def read_seg(path: str) -> np.ndarray:
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 3:
        array = array[..., 0]
    array = array.astype(np.int32)
    if _SEG_CHECK["n"] < 3:
        _SEG_CHECK["n"] += 1
        if array.max() > 58:
            print(
                f"  [warn] {os.path.basename(path)} max value {array.max()} > 58; "
                f"this may be a colorized rather than indexed mask. mode={image.mode}"
            )
    return array


def person_crop(img_path: str, seg: np.ndarray, height: int, width: int, margin: float):
    image = Image.open(img_path).convert("RGB")
    if image.size != (seg.shape[1], seg.shape[0]):
        image = image.resize((seg.shape[1], seg.shape[0]), Image.BICUBIC)
    original_h, original_w = seg.shape
    ratio = width / height
    ys, xs = np.nonzero(seg)
    if len(xs) == 0:
        center_x, center_y = original_w / 2, original_h / 2
        box_w, box_h = float(original_w), float(original_h)
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        box_w = (x1 - x0) * (1 + 2 * margin)
        box_h = (y1 - y0) * (1 + 2 * margin)
    if box_w / box_h > ratio:
        box_h = box_w / ratio
    else:
        box_w = box_h * ratio
    if box_w > original_w:
        box_w, box_h = float(original_w), original_w / ratio
    if box_h > original_h:
        box_h, box_w = float(original_h), original_h * ratio
    x = min(max(center_x - box_w / 2, 0.0), original_w - box_w)
    y = min(max(center_y - box_h / 2, 0.0), original_h - box_h)
    box = (
        int(round(x)), int(round(y)), int(round(x + box_w)), int(round(y + box_h))
    )
    image_array = (
        np.asarray(image.crop(box).resize((width, height), Image.BICUBIC), np.float32)
        / 127.5
        - 1.0
    )
    seg_image = Image.fromarray(seg.astype(np.uint8)).crop(box).resize(
        (width, height), Image.NEAREST
    )
    return image_array, np.array(seg_image, np.int32)


def _rect_mask(binary_mask: np.ndarray, height: int, width: int, shrink: float):
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    box_w, box_h = (x1 - x0) * (1 - shrink), (y1 - y0) * (1 - shrink)
    rx0, rx1 = int(round(center_x - box_w / 2)), int(round(center_x + box_w / 2))
    ry0, ry1 = int(round(center_y - box_h / 2)), int(round(center_y + box_h / 2))
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(width, rx1), min(height, ry1)
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None
    mask = np.zeros((height, width), np.uint8)
    mask[ry0:ry1, rx0:rx1] = 1
    return mask


def masks_from_seg(
    seg: np.ndarray,
    categories,
    height: int,
    width: int,
    rng: np.random.Generator,
    area_min: float,
    dilate_probability: float,
    rectangle_shrink: float,
):
    output = []
    for name, class_ids in categories:
        if not class_ids:
            continue
        binary_mask = np.isin(seg, list(class_ids))
        if binary_mask.mean() < area_min:
            continue
        segmentation_mask = binary_mask.astype(np.uint8)
        if rng.random() < dilate_probability:
            segmentation_mask = cv2.dilate(
                segmentation_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
        output.append((name + "_seg", segmentation_mask))
        rectangle_mask = _rect_mask(binary_mask, height, width, rectangle_shrink)
        if rectangle_mask is not None:
            output.append((name + "_rect", rectangle_mask))
    return output


def scan_pcs(
    pairs,
    categories,
    height: int,
    width: int,
    margin: float,
    area_min: float,
    workers: int = 16,
):
    def scan_one(indexed_pair):
        _, (image_path, mask_path) = indexed_pair
        try:
            seg = read_seg(mask_path)
            _, cropped_seg = person_crop(image_path, seg, height, width, margin)
        except Exception as error:
            return image_path, mask_path, [], f"err:{type(error).__name__}"
        present = []
        for name, class_ids in categories:
            if class_ids and np.isin(cropped_seg, list(class_ids)).mean() >= area_min:
                present.append(name)
        return image_path, mask_path, present, None

    start_time = time.time()
    with ThreadPoolExecutor(workers) as executor:
        results = list(executor.map(scan_one, enumerate(pairs)))
    kept, stats, errors = [], {"upper": 0, "lower": 0, "full": 0}, 0
    category_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for image_path, mask_path, present, error in results:
        if error:
            errors += 1
            continue
        category_counts[len(present)] += 1
        if not present:
            continue
        for category in present:
            stats[category] += 1
        kept.append((image_path, mask_path))
    print(f"scan {len(pairs)} images / {time.time() - start_time:.0f}s")
    print(f"  usable {len(kept)} | no clothing {category_counts[0]} | load errors {errors}")
    print(
        f"  category count: one {category_counts[1]}, two {category_counts[2]}, "
        f"three {category_counts[3]}"
    )
    print(f"  upper {stats['upper']} / lower {stats['lower']} / full {stats['full']}")
    estimated = stats["upper"] + stats["lower"] + stats["full"]
    print(
        f"  estimated valid masks {estimated * 2} "
        f"(average {estimated * 2 / max(len(kept), 1):.1f} per sample)"
    )
    return kept


class PCSRawDS(Dataset):
    def __init__(
        self,
        pairs,
        categories,
        height: int,
        width: int,
        masks: int,
        margin: float,
        area_min: float,
        dilate_probability: float,
        rectangle_shrink: float,
        seed: int = 0,
    ):
        self.pairs = pairs
        self.categories = categories
        self.height = height
        self.width = width
        self.masks = masks
        self.margin = margin
        self.area_min = area_min
        self.dilate_probability = dilate_probability
        self.rectangle_shrink = rectangle_shrink
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        cv2.setNumThreads(0)
        rng = np.random.default_rng(self.seed * 1000003 + index)
        image_path, mask_path = self.pairs[index]
        seg = read_seg(mask_path)
        image, cropped_seg = person_crop(
            image_path, seg, self.height, self.width, self.margin
        )
        found = masks_from_seg(
            cropped_seg,
            self.categories,
            self.height,
            self.width,
            rng,
            self.area_min,
            self.dilate_probability,
            self.rectangle_shrink,
        )
        if not found:
            found = [("fallback", np.ones((self.height, self.width), np.uint8))]
        valid_count = len(found)
        masks = [
            found[mask_index % valid_count][1].astype(np.float32)
            for mask_index in range(self.masks)
        ]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        return image_tensor, torch.from_numpy(np.stack(masks)), valid_count


@torch.no_grad()
def build_pcs_cache(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch: int = 24,
    chunk: int = 32,
    workers: int = 4,
    scan_workers: int = 16,
    seed: int = 0,
    output: str | None = None,
    progress_every: int | None = None,
    image_size: list[int] | None = None,
):
    configs = load_configs(config_dir)
    common = configs["dataset"]["common"]

    if image_size is None:
        height, width = common["image_size"]
    else:
        height, width = image_size

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("image size must be divisible by 8")

    settings = configs["dataset"]["datasets"]["people_clothing_segmentation"]
    output = output or configs["path"]["paths"]["cache"]["people_clothing_segmentation"]
    output_path = ensure_parent(output)
    masks = settings["masks_per_image"]
    scale = common["vae_scale"]
    if progress_every is None:
        progress_every = common["progress_every"]
    class_ids = load_pcs_class_ids(settings["root"], settings.get("labels_csv"))
    categories = list(zip(("upper", "lower", "full"), class_ids))
    pairs = list_pcs_pairs(settings["root"])
    pairs = scan_pcs(
        pairs,
        categories,
        height,
        width,
        settings["crop_margin"],
        settings["area_min"],
        scan_workers,
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
    dataset = PCSRawDS(
        pairs,
        categories,
        height,
        width,
        masks,
        settings["crop_margin"],
        settings["area_min"],
        settings["dilate_probability"],
        settings["rectangle_shrink"],
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    full, agnostic, mask_tensors, valid_counts = [], [], [], []
    start_time = time.time()
    processed = 0
    progress = CountProgress(len(pairs), progress_every)
    for images, pixel_masks, counts in loader:
        source_batch_size = images.shape[0]
        processed += source_batch_size
        images = torch.cat([images, torch.flip(images, dims=[3])])
        pixel_masks = torch.cat([pixel_masks, torch.flip(pixel_masks, dims=[3])])
        counts = torch.cat([counts, counts])
        batch_size = images.shape[0]
        valid_counts.append(counts.to(torch.uint8))
        combined = torch.cat(
            [images]
            + [images * (1 - pixel_masks[:, index : index + 1]) for index in range(masks)]
        )
        latent = encoder.encode(combined)
        full.append(latent[:batch_size])
        agnostic.append(
            latent[batch_size:]
            .view(masks, batch_size, *latent.shape[1:])
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        mask_tensors.append(
            F.max_pool2d(pixel_masks.reshape(batch_size * masks, 1, height, width), 8)
            .view(batch_size, masks, 1, height // 8, width // 8)
            .to(torch.uint8)
        )
        progress.update(processed)
    z_full = torch.cat(full)
    z_agnostic = torch.cat(agnostic)
    masks_out = torch.cat(mask_tensors)
    counts_out = torch.cat(valid_counts)
    print("z1", tuple(z_full.shape), "| agn", tuple(z_agnostic.shape), "| mask", tuple(masks_out.shape))
    print("channel std:", [round(value, 3) for value in z_full.float().std(dim=(0, 2, 3)).tolist()])
    print("mask coverage:", round(float(masks_out.float().mean()), 3))
    print("average valid masks:", round(float(counts_out.float().mean()), 2), f"/ {masks}")
    torch.save(
        {
            "z_full": z_full,
            "z_agn": z_agnostic,
            "masks": masks_out,
            "n_valid": counts_out,
            "scale": scale,
        },
        output_path,
    )
    print(f"saved {output_path.stat().st_size / 1e6:.0f} MB in {time.time() - start_time:.0f}s: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build the People Clothing Segmentation VAE cache.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
    )

    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scan-workers", type=int, default=16)
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
    output = Path(
        args.output
        or configs["path"]["paths"]["cache"]["people_clothing_segmentation"]
    )
    if output.is_file() and not args.force:
        print(f"PCS cache exists ({output.stat().st_size / 1e6:.0f} MB), skip: {output}")
        return
    build_pcs_cache(
        config_dir=args.config_dir,
        batch=args.batch,
        chunk=args.chunk,
        workers=args.workers,
        scan_workers=args.scan_workers,
        seed=args.seed,
        output=args.output,
        progress_every=args.progress_every,
        image_size=args.image_size,
    )


build_cache = build_pcs_cache


if __name__ == "__main__":
    main()
