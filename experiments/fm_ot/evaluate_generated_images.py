#!/usr/bin/env python3
"""Unified evaluator for generative image folders.

Computes FID/KID from local image directories and optionally precision/recall
using torch-fidelity when available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[Path]) -> None:
        self.image_paths = image_paths
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.to_tensor(img)


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    image_paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort()
    if not image_paths:
        raise ValueError(f"No images found under: {root}")
    return image_paths


def build_loader(image_paths: List[Path], batch_size: int, num_workers: int) -> DataLoader:
    dataset = ImagePathDataset(image_paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def sample_image_paths(
    image_paths: List[Path],
    max_images: int,
    seed: int,
) -> List[Path]:
    if max_images <= 0 or len(image_paths) <= max_images:
        return image_paths
    rng = random.Random(seed)
    sampled_idx = sorted(rng.sample(range(len(image_paths)), k=max_images))
    return [image_paths[i] for i in sampled_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated image folder.")
    parser.add_argument("--real_dir", required=True, type=str)
    parser.add_argument("--fake_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Set to 0 in Colab to avoid /dev/shm worker issues.",
    )
    parser.add_argument("--compute_kid", action="store_true")
    parser.add_argument("--compute_pr", action="store_true")
    parser.add_argument("--kid_subsets", default=50, type=int)
    parser.add_argument("--kid_subset_size", default=1000, type=int)
    parser.add_argument(
        "--max_images",
        default=0,
        type=int,
        help="If >0, evaluate using at most this many images from each set.",
    )
    parser.add_argument(
        "--sample_seed",
        default=0,
        type=int,
        help="Seed for deterministic image subset sampling when --max_images is used.",
    )
    parser.add_argument("--output_json", default="", type=str)
    return parser.parse_args()


def evaluate_fid(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    device: torch.device,
) -> float:
    metric = FrechetInceptionDistance(normalize=True).to(device)
    with torch.no_grad():
        for batch in real_loader:
            metric.update(batch.to(device, non_blocking=True), real=True)
        for batch in fake_loader:
            metric.update(batch.to(device, non_blocking=True), real=False)
    return float(metric.compute().detach().cpu())


def evaluate_kid(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    device: torch.device,
    subsets: int,
    subset_size: int,
) -> Dict[str, float]:
    metric = KernelInceptionDistance(
        feature=2048,
        subsets=subsets,
        subset_size=subset_size,
        normalize=True,
    ).to(device)
    with torch.no_grad():
        for batch in real_loader:
            metric.update(batch.to(device, non_blocking=True), real=True)
        for batch in fake_loader:
            metric.update(batch.to(device, non_blocking=True), real=False)
    mean, std = metric.compute()
    return {
        "kid_mean": float(mean.detach().cpu()),
        "kid_std": float(std.detach().cpu()),
    }


def evaluate_precision_recall(real_dir: Path, fake_dir: Path, device: torch.device) -> Dict[str, float]:
    try:
        from torch_fidelity import calculate_metrics
    except Exception as ex:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "torch-fidelity is required for precision/recall. "
            "Install via `pip install torch-fidelity`."
        ) from ex

    results = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=device.type == "cuda",
        isc=False,
        fid=False,
        kid=False,
        prc=True,
        verbose=False,
    )
    return {
        "precision": float(results["precision"]),
        "recall": float(results["recall"]),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    real_dir = Path(args.real_dir).resolve()
    fake_dir = Path(args.fake_dir).resolve()
    real_images = collect_images(real_dir)
    fake_images = collect_images(fake_dir)
    real_images = sample_image_paths(real_images, args.max_images, args.sample_seed)
    fake_images = sample_image_paths(fake_images, args.max_images, args.sample_seed)

    real_loader = build_loader(real_images, args.batch_size, args.num_workers)
    fake_loader = build_loader(fake_images, args.batch_size, args.num_workers)

    metrics: Dict[str, float] = {
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "num_real_images": len(real_images),
        "num_fake_images": len(fake_images),
        "fid": evaluate_fid(real_loader, fake_loader, device),
    }

    if args.compute_kid:
        metrics.update(
            evaluate_kid(
                real_loader=real_loader,
                fake_loader=fake_loader,
                device=device,
                subsets=args.kid_subsets,
                subset_size=args.kid_subset_size,
            )
        )

    if args.compute_pr:
        if args.max_images > 0:
            raise ValueError(
                "Precision/recall currently uses directory-level inputs via torch-fidelity. "
                "Use --max_images 0 when --compute_pr is enabled."
            )
        metrics.update(evaluate_precision_recall(real_dir, fake_dir, device))

    output = json.dumps(metrics, indent=2, sort_keys=True)
    print(output)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
