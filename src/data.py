from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset


class InpaintDS(Dataset):
    def __init__(self, path: str | Path, tag: str = ""):
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.z1 = data["z_full"]
        self.za = data["z_agn"]
        self.m = data["masks"]
        self.scale = data["scale"]
        self.K = self.za.shape[1]
        print(f"[{tag or path.name}] {len(self.z1)} latents (including flips), {self.K} masks")

    def __len__(self):
        return len(self.z1)

    def __getitem__(self, index: int):
        mask_index = int(torch.randint(self.K, (1,)))
        return (
            self.z1[index].float(),
            self.za[index, mask_index].float(),
            self.m[index, mask_index].float(),
        )


class InpaintConcatDataset(ConcatDataset):
    def __init__(self, datasets: list[InpaintDS]):
        super().__init__(datasets)
        self.scale = datasets[0].scale
        self.parts = datasets


def build_dataset(cache_paths: dict[str, str]):
    fashionpedia_path = cache_paths["fashionpedia"]
    if not os.path.isfile(fashionpedia_path):
        raise FileNotFoundError(
            f"Fashionpedia cache does not exist: {fashionpedia_path}. Run latent_save.sh first."
        )
    parts = [InpaintDS(fashionpedia_path, tag="fashionpedia")]
    pcs_path = cache_paths["people_clothing_segmentation"]
    if os.path.isfile(pcs_path):
        parts.append(InpaintDS(pcs_path, tag="people_clothing_segmentation"))
    else:
        print(f"[warn] PCS cache not found; training with Fashionpedia only: {pcs_path}")
    if len(parts) == 1:
        return parts[0]
    dataset = InpaintConcatDataset(parts)
    print(f"total samples: {len(dataset)}")
    return dataset
