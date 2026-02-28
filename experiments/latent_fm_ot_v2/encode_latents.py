#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision import datasets

from common import (
    build_autoencoder_from_config,
    ensure_dir,
    image_transform,
    latent_file_paths,
    load_config,
    resolve_device,
    resolve_pretrained_ae_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode train/test sets with the shared AE encoder.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument(
        "--ae_checkpoint",
        default="",
        type=str,
        help="Optional explicit pretrained checkpoint path. Defaults to config paths.pretrained_ae_checkpoint.",
    )
    return parser.parse_args()


def _encode_dataset(
    model,
    dataset_root: str,
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    latents: List[torch.Tensor] = []
    image_paths: List[str] = [dataset.samples[i][0] for i in range(len(dataset.samples))]

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            z = model.encoder(images).view(images.shape[0], -1)
            latents.append(z.detach().cpu())

    if not latents:
        raise RuntimeError(f"No images encoded from dataset root: {dataset_root}")

    return torch.cat(latents, dim=0), image_paths


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    checkpoint_path = resolve_pretrained_ae_checkpoint(cfg=cfg, explicit=args.ae_checkpoint)

    model = build_autoencoder_from_config(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    transform = image_transform(
        center_crop_size=int(cfg["dataset"]["center_crop_size"]),
        image_size=int(cfg["dataset"]["image_size"]),
    )

    batch_size = int(cfg["ae"]["batch_size"])
    num_workers = int(cfg["dataset"]["num_workers"])

    train_latents, train_paths = _encode_dataset(
        model=model,
        dataset_root=cfg["dataset"]["train_dir"],
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    test_latents, test_paths = _encode_dataset(
        model=model,
        dataset_root=cfg["dataset"]["test_dir"],
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    train_latents_file, test_latents_file = latent_file_paths(seed_root)
    ensure_dir(train_latents_file.parent)

    torch.save(train_latents, train_latents_file)
    torch.save(test_latents, test_latents_file)

    save_json(
        {
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "train_latents_file": str(train_latents_file),
            "test_latents_file": str(test_latents_file),
            "train_count": int(train_latents.shape[0]),
            "test_count": int(test_latents.shape[0]),
            "latent_dim": int(train_latents.shape[1]),
            "train_paths_file": str(train_latents_file.with_suffix(".paths.txt")),
            "test_paths_file": str(test_latents_file.with_suffix(".paths.txt")),
        },
        train_latents_file.parent / "summary.json",
    )
    train_latents_file.with_suffix(".paths.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    test_latents_file.with_suffix(".paths.txt").write_text("\n".join(test_paths) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
