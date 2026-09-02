from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

from src.config import DEFAULT_CONFIG_DIR, load_configs
from src.model import EMA
from src.model.factory import build_model
from src.torch_utils import resolve_dtype


def load_viton(path: str | Path, height: int, width: int):
    image = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    array = np.asarray(image, np.float32)
    tensor = torch.from_numpy(array / 127.5 - 1.0).permute(2, 0, 1)[None]
    return array, tensor


def gray_mask(rgb255: np.ndarray, tolerance: int = 14, min_area: int = 50):
    mask = (np.abs(rgb255 - 128).max(axis=2) < tolerance).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            output[labels == index] = 1
    return output.astype(np.float32)


@torch.no_grad()
def encode(vae, image: torch.Tensor, scale: float, device: str):
    vae_dtype = next(vae.parameters()).dtype
    return (
        vae.encode(image.to(device, vae_dtype)).latent_dist.mode() * scale
    ).float()


@torch.no_grad()
def decode(vae, latent: torch.Tensor, scale: float):
    vae_dtype = next(vae.parameters()).dtype
    image = vae.decode((latent / scale).to(vae_dtype)).sample.float().clamp(-1, 1)
    return ((image[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)


@torch.no_grad()
def run_inference(
    agnostic_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    configs: dict,
    cloth_path: str | Path | None = None,
    steps: int = 50,
    seed: int = 0,
    show: bool = False,
):
    common = configs["dataset"]["common"]
    model_config = configs["model"]["model"]
    vae_config = configs["model"]["vae"]
    train_config = configs["train"]["train"]
    height, width = common["image_size"]
    scale = common["vae_scale"]
    device = train_config["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA.")

    model = build_model(model_config, common["latent_size"]).to(device)
    ema = EMA(model, train_config["ema_decay"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ema.model.load_state_dict(checkpoint["ema"])
    print("weights loaded | keys:", list(checkpoint.keys()))
    vae = AutoencoderKL.from_pretrained(
        vae_config["pretrained_model_name_or_path"],
        torch_dtype=resolve_dtype(vae_config["dtype"]),
    ).to(device).eval()

    rgb, agnostic_image = load_viton(agnostic_path, height, width)
    pixel_mask = gray_mask(rgb)
    print(f"mask coverage: {pixel_mask.mean():.3f} (approximately 0.15-0.35 is expected)")
    agnostic_latent = encode(vae, agnostic_image, scale, device)
    mask = F.max_pool2d(torch.from_numpy(pixel_mask)[None, None], 8).to(device)
    garment = None
    if cloth_path is not None:
        _, cloth_image = load_viton(cloth_path, height, width)
        garment = encode(vae, cloth_image, scale, device)

    torch.manual_seed(seed)
    latent = torch.randn_like(agnostic_latent)
    net = ema.model.eval()
    for index in range(steps):
        t = torch.full((1,), index / steps, device=device)
        latent = latent + net(latent, t, agnostic_latent, mask, garment) / steps
    composite = agnostic_latent * (1 - mask) + latent * mask
    raw_output = decode(vae, latent, scale)
    final_output = decode(vae, composite, scale)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(14, 5))
    values = [rgb.astype(np.uint8), (pixel_mask * 255).astype(np.uint8), raw_output, final_output]
    titles = ["agnostic", "mask", "raw", "output"]
    for axis, value, title in zip(axes, values, titles):
        axis.imshow(value, cmap="gray" if value.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    print(f"saved inference grid: {output_path}")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Run TinyDiT inference on a VITON-HD agnostic image.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--agnostic")
    parser.add_argument("--cloth")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    configs = load_configs(args.config_dir)
    paths = configs["path"]["paths"]
    agnostic = args.agnostic or str(
        Path(paths["inference_dataset_root"]) / "agnostic-v3.2" / "00036_00.jpg"
    )
    checkpoint = args.checkpoint or paths["inference_checkpoint"]
    output = args.output or str(Path(paths["inference_output_dir"]) / "result.png")
    run_inference(
        agnostic,
        checkpoint,
        output,
        configs,
        args.cloth,
        args.steps,
        args.seed,
        args.show,
    )


if __name__ == "__main__":
    main()
