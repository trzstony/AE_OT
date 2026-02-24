#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import ensure_dir, fm_dir, integrate_velocity, load_config, resolve_device, save_json, set_seed
from models import LatentVelocityMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample latent vectors from trained latent FM model.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fm_checkpoint", required=True, type=str)
    parser.add_argument("--output_latent_file", required=True, type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--num_samples", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    fm_root = ensure_dir(fm_dir(seed_root))

    checkpoint_path = Path(args.fm_checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing FM checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    latent_dim = int(ckpt["latent_dim"])
    model = LatentVelocityMLP(
        latent_dim=latent_dim,
        hidden_dim=int(ckpt["hidden_dim"]),
        depth=int(ckpt["depth"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    num_samples = int(args.num_samples) if args.num_samples > 0 else int(cfg["eval"]["generated_samples"])
    ode_steps = int(cfg["fm"].get("ode_steps", 100))
    ode_method = str(cfg["fm"].get("ode_method", "midpoint"))
    chunk_size = int(cfg["ot"].get("sample_chunk_size", 2048))

    all_latents = []
    remaining = num_samples

    with torch.no_grad():
        while remaining > 0:
            bs = min(remaining, chunk_size)
            x0 = torch.randn((bs, latent_dim), device=device)
            x1 = integrate_velocity(model=model, x0=x0, n_steps=ode_steps, method=ode_method)
            all_latents.append(x1.detach().cpu())
            remaining -= bs

    latents = torch.cat(all_latents, dim=0)[:num_samples]
    output_latent_file = Path(args.output_latent_file).resolve()
    ensure_dir(output_latent_file.parent)
    torch.save(latents, output_latent_file)

    save_json(
        {
            "seed": args.seed,
            "fm_checkpoint": str(checkpoint_path),
            "output_latent_file": str(output_latent_file),
            "num_samples": int(latents.shape[0]),
            "ode_steps": ode_steps,
            "ode_method": ode_method,
        },
        fm_root / f"sample_{output_latent_file.stem}.json",
    )


if __name__ == "__main__":
    main()
