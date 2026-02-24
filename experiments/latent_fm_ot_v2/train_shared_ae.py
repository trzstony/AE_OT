#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from common import (
    ae_dir,
    build_autoencoder_from_config,
    ensure_dir,
    image_transform,
    imagefolder_loader,
    load_config,
    parse_optimizer_betas,
    psnr_from_mse,
    resolve_device,
    save_ae_checkpoint,
    save_json,
    save_jsonl,
    sanity_check_dataset_dirs,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared autoencoder for latent FM-vs-OT.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def evaluate_mse(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total = 0.0
    count = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            recon, _ = model(images)
            loss = criterion(recon, images)
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    sanity_check_dataset_dirs(cfg)

    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    ae_root = ensure_dir(ae_dir(seed_root))

    transform = image_transform(
        center_crop_size=int(cfg["dataset"]["center_crop_size"]),
        image_size=int(cfg["dataset"]["image_size"]),
    )
    train_loader = imagefolder_loader(
        root=cfg["dataset"]["train_dir"],
        transform=transform,
        batch_size=int(cfg["ae"]["batch_size"]),
        num_workers=int(cfg["dataset"]["num_workers"]),
        shuffle=True,
        drop_last=True,
    )
    test_loader = imagefolder_loader(
        root=cfg["dataset"]["test_dir"],
        transform=transform,
        batch_size=int(cfg["ae"]["batch_size"]),
        num_workers=int(cfg["dataset"]["num_workers"]),
        shuffle=False,
        drop_last=False,
    )

    model = build_autoencoder_from_config(cfg).to(device)

    final_ckpt = ae_root / "ae_final.pt"
    if args.resume and final_ckpt.exists():
        model.load_state_dict(torch.load(final_ckpt, map_location=device))

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["ae"]["learning_rate"]))

    num_epochs = int(cfg["ae"]["num_epochs"])
    l1_lambda = float(cfg["ae"]["l1_lambda"])
    save_every = int(cfg["ae"].get("save_every", 10))

    best_test_mse = float("inf")
    best_epoch = -1
    best_ckpt = ae_root / "ae_best.pt"

    rows = []
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_l1 = 0.0
        count = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            recon, z = model(images)
            rec_loss = criterion(recon, images)
            l1_loss = torch.norm(z, p=1) / float(z.numel())
            loss = rec_loss + l1_lambda * l1_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_rec += float(rec_loss.item())
            epoch_l1 += float(l1_loss.item())
            count += 1

        train_loss = epoch_loss / max(count, 1)
        train_rec = epoch_rec / max(count, 1)
        train_l1 = epoch_l1 / max(count, 1)
        test_mse = evaluate_mse(model, test_loader, device)
        test_psnr = psnr_from_mse(test_mse)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rec_mse": train_rec,
            "train_l1": train_l1,
            "test_mse": test_mse,
            "test_psnr": test_psnr,
        }
        rows.append(row)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_epoch = epoch
            save_ae_checkpoint(model, best_ckpt)

        if (epoch + 1) % save_every == 0:
            save_ae_checkpoint(model, ae_root / f"ae_epoch_{epoch+1}.pt")

    save_ae_checkpoint(model, final_ckpt)
    elapsed_sec = time.time() - start

    save_jsonl(rows, ae_root / "train_log.jsonl")
    save_json(
        {
            "seed": args.seed,
            "device": str(device),
            "num_epochs": num_epochs,
            "best_epoch": best_epoch,
            "best_test_mse": best_test_mse,
            "best_test_psnr": psnr_from_mse(best_test_mse),
            "best_checkpoint": str(best_ckpt),
            "final_checkpoint": str(final_ckpt),
            "elapsed_sec": elapsed_sec,
        },
        ae_root / "summary.json",
    )


if __name__ == "__main__":
    main()
