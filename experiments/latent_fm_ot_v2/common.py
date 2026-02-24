#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml

from pyOMT.networks import autoencoder


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    if not rows:
        return
    ensure_dir(path.parent)
    keys = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(explicit: str | None = None) -> torch.device:
    if explicit:
        if explicit == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_transform(center_crop_size: int, image_size: int) -> transforms.Compose:
    # Use [-1, 1] normalization because the AE decoder ends with tanh.
    return transforms.Compose(
        [
            transforms.CenterCrop(center_crop_size),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def imagefolder_loader(
    root: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = datasets.ImageFolder(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_autoencoder_from_config(cfg: Dict[str, Any]) -> nn.Module:
    ae_cfg = cfg["ae"]
    return autoencoder(
        dim_z=int(ae_cfg["dim_z"]),
        dim_c=3,
        dim_f=int(ae_cfg["dim_f"]),
    )


def save_ae_checkpoint(model: nn.Module, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def load_ae_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    # Convert from [-1,1] to [0,1] for image saving/metrics.
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def latent_file_paths(seed_root: Path) -> Tuple[Path, Path]:
    latent_dir = seed_root / "latent_cache"
    return latent_dir / "train_latents.pt", latent_dir / "test_latents.pt"


def ae_dir(seed_root: Path) -> Path:
    return seed_root / "ae"


def fm_dir(seed_root: Path) -> Path:
    return seed_root / "fm"


def ot_dir(seed_root: Path) -> Path:
    return seed_root / "ot"


def decoded_dir(seed_root: Path) -> Path:
    return seed_root / "decoded"


def sanity_check_dataset_dirs(cfg: Dict[str, Any]) -> None:
    train_dir = Path(cfg["dataset"]["train_dir"]).expanduser().resolve()
    test_dir = Path(cfg["dataset"]["test_dir"]).expanduser().resolve()
    for path in [train_dir, test_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {path}")


def parse_optimizer_betas(values: List[float]) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"optimizer_betas must have length 2, got {values}")
    return float(values[0]), float(values[1])


def midpoint_step(model: nn.Module, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    t_batch = torch.full((x.shape[0], 1), fill_value=t, device=x.device, dtype=x.dtype)
    k1 = model(x, t_batch)
    mid_x = x + 0.5 * dt * k1
    mid_t = torch.full((x.shape[0], 1), fill_value=t + 0.5 * dt, device=x.device, dtype=x.dtype)
    k2 = model(mid_x, mid_t)
    return x + dt * k2


def euler_step(model: nn.Module, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    t_batch = torch.full((x.shape[0], 1), fill_value=t, device=x.device, dtype=x.dtype)
    return x + dt * model(x, t_batch)


def integrate_velocity(
    model: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    method: str,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    dt = 1.0 / float(n_steps)
    x = x0
    for i in range(n_steps):
        t = i * dt
        if method == "midpoint":
            x = midpoint_step(model, x, t=t, dt=dt)
        elif method == "euler":
            x = euler_step(model, x, t=t, dt=dt)
        else:
            raise ValueError(f"Unsupported ODE method: {method}")
    return x


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)
