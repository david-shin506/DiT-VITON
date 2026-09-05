from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from src.config import DEFAULT_CONFIG_DIR, load_configs
from src.data import build_dataset
from src.model import EMA, flow_matching_loss
from src.model.factory import build_model
from src.torch_utils import resolve_dtype


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_wandb(settings: dict, run_config: dict):
    if not settings.get("enabled", True):
        return None
    import wandb

    api_key_env = settings.get("api_key_env", "WANDB_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"W&B is enabled but ${api_key_env} is not set. Export it or set wandb.enabled=false."
        )
    wandb.login(key=api_key, relogin=settings.get("relogin", True))
    init_kwargs = {
        key: settings[key]
        for key in ("project", "name", "entity", "group", "job_type", "tags", "notes", "mode", "resume")
        if settings.get(key) is not None
    }
    run = wandb.init(config=run_config, **init_kwargs)
    print("wandb run id:", run.id)
    return wandb


@torch.no_grad()
def sample_grid(net, z1, agnostic, mask, scale: float, vae, device: str, steps: int = 32):
    net.eval()
    latent = torch.randn_like(z1)
    for index in range(steps):
        t = torch.full((latent.shape[0],), index / steps, device=device)
        latent = latent + net(latent, t, agnostic, mask) / steps
    composite = agnostic * (1 - mask) + latent * mask

    def decode(value):
        vae_dtype = next(vae.parameters()).dtype
        image = vae.decode((value / scale).to(vae_dtype)).sample.float().clamp(-1, 1)
        return ((image + 1) * 127.5).to(torch.uint8)

    rows = [torch.cat(list(decode(value)), dim=2) for value in (agnostic, latent, composite, z1)]
    return torch.cat(rows, dim=1).permute(1, 2, 0).cpu().numpy()


def fixed_sample_indices(dataset) -> list[int]:
    if hasattr(dataset, "parts") and len(dataset.parts) > 1:
        first_length = len(dataset.parts[0])
        return (
            torch.linspace(0, first_length - 1, 3).long().tolist()
            + [
                first_length + index
                for index in torch.linspace(0, len(dataset) - first_length - 1, 3).long().tolist()
            ]
        )
    return torch.linspace(0, len(dataset) - 1, 6).long().tolist()


def main():
    parser = argparse.ArgumentParser(description="Train TinyDiT for latent-space virtual try-on.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--resume", help="Override paths.resume_checkpoint.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override train.batch_size.",
    )

    parser.add_argument(
        "--dataset-name",
        choices=("all", "fashionpedia", "pcs"),
        default="all",
        help="Select the latent cache used for training.",
    )

    args = parser.parse_args()

    configs = load_configs(args.config_dir)
    dataset_config = configs["dataset"]
    paths = configs["path"]["paths"]
    model_config = configs["model"]["model"]
    vae_config = configs["model"]["vae"]
    train_config = configs["train"]["train"]

    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        train_config["batch_size"] = args.batch_size

    common = dataset_config["common"]
    height, width = args.image_size or common["image_size"]

    required_multiple = 8 * model_config["patch_size"]

    if (
        height % required_multiple != 0
        or width % required_multiple != 0
    ):
        raise ValueError(
            f"image size must be divisible by {required_multiple}"
        )

    common["image_size"] = [height, width]
    common["latent_size"] = [height // 8, width // 8]

    device = train_config["device"]
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("This training pipeline requires a CUDA device.")
    seed_everything(train_config["seed"])
    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        paths["cache"],
        dataset_name=args.dataset_name,
    )
    workers = train_config["dataloader_workers"]
    loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=train_config["pin_memory"],
        drop_last=train_config["drop_last"],
        persistent_workers=train_config["persistent_workers"] and workers > 0,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "The training DataLoader is empty. Lower train.batch_size or set train.drop_last=false."
        )
    model = build_model(model_config, dataset_config["common"]["latent_size"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=tuple(train_config["betas"]),
    )
    mixed_precision_dtype = resolve_dtype(train_config["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision_dtype == torch.float16)
    ema = EMA(model, train_config["ema_decay"])
    step = 0
    resume_path = args.resume if args.resume is not None else paths["resume_checkpoint"]
    if resume_path and Path(resume_path).is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.model.load_state_dict(checkpoint["ema"])
        print(f"weights loaded: {resume_path}")
    else:
        print("training from scratch")

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    run_config = {
        "dataset": dataset_config,
        "paths": paths,
        "model": configs["model"],
        "train": train_config,
        "params_M": params_m,
    }
    wandb = initialize_wandb(configs["wandb"]["wandb"], run_config)

    fixed_examples = [dataset[index] for index in fixed_sample_indices(dataset)]
    fixed_z1 = torch.stack([item[0] for item in fixed_examples]).to(device)
    fixed_agnostic = torch.stack([item[1] for item in fixed_examples]).to(device)
    fixed_mask = torch.stack([item[2] for item in fixed_examples]).to(device)
    vae = AutoencoderKL.from_pretrained(
        vae_config["pretrained_model_name_or_path"],
        torch_dtype=resolve_dtype(vae_config["dtype"]),
    ).to(device).eval()

    model.train()
    last_log_time = last_save_time = time.time()
    accumulated_loss = 0.0
    accumulated_batches = 0
    done = False
    while not done:
        for z1, agnostic, mask in loader:
            if step >= train_config["steps"]:
                done = True
                break
            z1 = z1.to(device, non_blocking=True)
            agnostic = agnostic.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            learning_rate = train_config["learning_rate"] * min(
                1.0, (step + 1) / train_config["warmup_steps"]
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            autocast_context = (
                torch.autocast("cuda", dtype=mixed_precision_dtype)
                if mixed_precision_dtype in (torch.float16, torch.bfloat16)
                else nullcontext()
            )
            with autocast_context:
                loss = flow_matching_loss(model, z1, agnostic, mask, garment=None)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["gradient_clip_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            accumulated_loss += loss.item()
            accumulated_batches += 1
            step += 1

            if step % train_config["log_every"] == 0:
                images_per_second = (
                    train_config["log_every"] * train_config["batch_size"]
                    / (time.time() - last_log_time)
                )
                metrics = {
                    "loss": accumulated_loss / accumulated_batches,
                    "lr": learning_rate,
                    "grad_norm": float(gradient_norm),
                    "img_per_s": images_per_second,
                    "gpu_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
                }
                if wandb:
                    wandb.log(metrics, step=step)
                print(
                    f"{step:6d}  loss {metrics['loss']:.4f}  lr {learning_rate:.2e}  "
                    f"|g| {float(gradient_norm):.2f}  {images_per_second:.0f} img/s"
                )
                accumulated_loss = 0.0
                accumulated_batches = 0
                last_log_time = time.time()

            if step % train_config["sample_every"] == 0:
                fixed_grid = sample_grid(
                    ema.model,
                    fixed_z1,
                    fixed_agnostic,
                    fixed_mask,
                    dataset.scale,
                    vae,
                    device,
                    train_config["sample_steps"],
                )
                random_indices = torch.randint(len(dataset), (4,)).tolist()
                random_examples = [dataset[index] for index in random_indices]
                random_z1 = torch.stack([item[0] for item in random_examples]).to(device)
                random_agnostic = torch.stack([item[1] for item in random_examples]).to(device)
                random_mask = torch.stack([item[2] for item in random_examples]).to(device)
                random_grid = sample_grid(
                    ema.model,
                    random_z1,
                    random_agnostic,
                    random_mask,
                    dataset.scale,
                    vae,
                    device,
                    train_config["sample_steps"],
                )
                if wandb:
                    caption = "rows: masked input / prediction (raw) / composite / target"
                    wandb.log(
                        {
                            "samples/fixed": wandb.Image(fixed_grid, caption=f"fixed - {caption}"),
                            "samples/random": wandb.Image(random_grid, caption=f"random - {caption}"),
                        },
                        step=step,
                    )
                model.train()
                last_log_time = time.time()

            if (
                time.time() - last_save_time > train_config["save_every_minutes"] * 60
                or step >= train_config["steps"]
            ):
                checkpoint_path = checkpoint_dir / "latest.ckpt"
                torch.save(
                    {"model": model.state_dict(), "ema": ema.model.state_dict()},
                    checkpoint_path,
                )
                print(f"  saved @ {step}: {checkpoint_path}")
                last_save_time = last_log_time = time.time()
    if wandb:
        wandb.finish()
    print("done @", step)


if __name__ == "__main__":
    main()
