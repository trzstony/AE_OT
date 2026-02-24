#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from common import ae_dir, build_autoencoder_from_config, ensure_dir, load_config, resolve_device, save_json, set_seed, to_uint8_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode latent samples with shared AE decoder.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--latent_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--ae_checkpoint", default="", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    ae_root = ae_dir(seed_root)

    latent_file = Path(args.latent_file).resolve()
    if not latent_file.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_file}")

    if args.ae_checkpoint:
        ae_checkpoint = Path(args.ae_checkpoint).resolve()
    else:
        candidate_best = ae_root / "ae_best.pt"
        candidate_final = ae_root / "ae_final.pt"
        ae_checkpoint = candidate_best if candidate_best.exists() else candidate_final

    if not ae_checkpoint.exists():
        raise FileNotFoundError(f"Missing AE checkpoint: {ae_checkpoint}")

    model = build_autoencoder_from_config(cfg).to(device)
    model.load_state_dict(torch.load(ae_checkpoint, map_location=device))
    model.eval()

    latents = torch.load(latent_file, map_location="cpu").to(torch.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latent tensor shape [N, D], got {tuple(latents.shape)}")

    out_dir = ensure_dir(Path(args.output_dir).resolve())

    preview_images = []
    bs = int(args.batch_size)
    with torch.no_grad():
        for start in range(0, latents.shape[0], bs):
            end = min(start + bs, latents.shape[0])
            z = latents[start:end].to(device)
            z = z.view(z.shape[0], z.shape[1], 1, 1)
            decoded = model.decoder(z)
            decoded = to_uint8_image(decoded).cpu()

            for i in range(decoded.shape[0]):
                img_idx = start + i
                save_image(decoded[i], out_dir / f"gen_img_{img_idx:06d}.png")
                if len(preview_images) < 64:
                    preview_images.append(decoded[i])

    if preview_images:
        grid = torch.stack(preview_images, dim=0)
        save_image(grid, out_dir / "preview_grid.png", nrow=8)

    save_json(
        {
            "seed": args.seed,
            "latent_file": str(latent_file),
            "ae_checkpoint": str(ae_checkpoint),
            "output_dir": str(out_dir),
            "decoded_count": int(latents.shape[0]),
        },
        out_dir / "decode_summary.json",
    )


if __name__ == "__main__":
    main()
