from __future__ import annotations

import copy

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL

from src.torch_utils import resolve_dtype


def random_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Notebook-compatible fallback mask generator."""
    mask = np.zeros((height, width), np.uint8)
    if rng.random() < 0.5:
        box_h = int(rng.integers(int(height * 0.35), int(height * 0.75)))
        box_w = int(rng.integers(int(width * 0.35), int(width * 0.85)))
        y = int(rng.integers(0, height - box_h + 1))
        x = int(rng.integers(0, width - box_w + 1))
        mask[y : y + box_h, x : x + box_w] = 1
    else:
        center_y = height // 2 + int(rng.integers(-height // 8, height // 8))
        center_x = width // 2 + int(rng.integers(-width // 8, width // 8))
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (int(rng.integers(width // 4, width // 2)), int(rng.integers(height // 4, height // 2))),
            int(rng.integers(0, 180)),
            0,
            360,
            1,
            -1,
        )
    return mask.astype(np.float32)


def load_vae(
    model_name: str, dtype: str = "float16", device: str = "cuda:0"
) -> AutoencoderKL:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the VAE latent cache.")
    return AutoencoderKL.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
    ).to(device).eval()


class MultiGpuVaeEncoder:
    """Replicate the notebook VAE over all visible GPUs and encode round-robin."""

    def __init__(self, vae: AutoencoderKL, scale: float, chunk: int):
        self.scale = scale
        self.chunk = chunk
        self.dtype = next(vae.parameters()).dtype
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count < 1:
            raise RuntimeError("At least one CUDA device is required.")
        self.vaes = [vae] + [
            copy.deepcopy(vae).to(f"cuda:{index}").eval()
            for index in range(1, self.gpu_count)
        ]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pending: list[torch.Tensor] = []
        output: list[torch.Tensor] = []
        for start in range(0, len(images), self.chunk):
            device_index = (start // self.chunk) % self.gpu_count
            batch = images[start : start + self.chunk].to(
                f"cuda:{device_index}", self.dtype, non_blocking=True
            )
            pending.append(
                self.vaes[device_index].encode(batch).latent_dist.mode() * self.scale
            )
            if len(pending) >= self.gpu_count * 4:
                output.extend(tensor.cpu().half() for tensor in pending)
                pending = []
        output.extend(tensor.cpu().half() for tensor in pending)
        return torch.cat(output)
