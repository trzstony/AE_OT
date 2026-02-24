#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from common import ensure_dir, latent_file_paths, load_config, ot_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample latent vectors from trained OT dual potential.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--h_checkpoint", required=True, type=str)
    parser.add_argument("--output_latent_file", required=True, type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--num_samples", default=0, type=int)
    return parser.parse_args()


def _merge_topk(
    old_vals: torch.Tensor,
    old_idx: torch.Tensor,
    new_vals: torch.Tensor,
    new_idx: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_vals = torch.cat([old_vals, new_vals], dim=0)
    all_idx = torch.cat([old_idx, new_idx], dim=0)
    merged_vals, merged_pos = torch.topk(all_vals, k=topk, dim=0)
    merged_idx = torch.gather(all_idx, dim=0, index=merged_pos)
    return merged_vals, merged_idx


def topk_assignments(
    points: torch.Tensor,
    h: torch.Tensor,
    x_samples: torch.Tensor,
    point_batch_size: int,
    topk: int,
) -> torch.Tensor:
    num_points = points.shape[0]
    num_x = x_samples.shape[0]

    best_vals = torch.full((topk, num_x), fill_value=-1e30, device=x_samples.device)
    best_idx = torch.full((topk, num_x), fill_value=-1, device=x_samples.device, dtype=torch.long)

    x_t = x_samples.transpose(0, 1)

    for start in range(0, num_points, point_batch_size):
        end = min(start + point_batch_size, num_points)
        p_chunk = points[start:end]
        h_chunk = h[start:end]

        scores = p_chunk @ x_t
        scores = scores + h_chunk.unsqueeze(-1)

        k_chunk = min(topk, p_chunk.shape[0])
        vals, idx = torch.topk(scores, k=k_chunk, dim=0)
        idx = idx + start

        if k_chunk < topk:
            pad = topk - k_chunk
            vals = torch.cat(
                [vals, torch.full((pad, num_x), fill_value=-1e30, device=vals.device)],
                dim=0,
            )
            idx = torch.cat(
                [idx, torch.full((pad, num_x), fill_value=-1, device=idx.device, dtype=torch.long)],
                dim=0,
            )

        best_vals, best_idx = _merge_topk(best_vals, best_idx, vals, idx, topk=topk)

    return best_idx


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    ot_root = ensure_dir(ot_dir(seed_root))

    train_latents_file, _ = latent_file_paths(seed_root)
    if not train_latents_file.exists():
        raise FileNotFoundError(f"Missing train latents: {train_latents_file}")

    points = torch.load(train_latents_file, map_location="cpu").to(device=device, dtype=torch.float32)
    h = torch.load(Path(args.h_checkpoint).resolve(), map_location="cpu").to(device=device, dtype=torch.float32)

    if points.shape[0] != h.shape[0]:
        raise ValueError(
            f"OT checkpoint mismatch: points={points.shape[0]}, h={h.shape[0]}"
        )

    num_samples = int(args.num_samples) if args.num_samples > 0 else int(cfg["eval"]["generated_samples"])
    topk = int(cfg["ot"].get("topk", 5))
    point_batch_size = int(cfg["ot"].get("point_batch_size", 4096))
    proposal_batch = int(cfg["ot"].get("x_proposal_batch", 2048))
    angle_threshold = float(cfg["ot"].get("angle_threshold", 0.7))
    dissim = float(cfg["ot"].get("dissimilarity", 0.75))

    normals = torch.cat(
        [points, -torch.ones((points.shape[0], 1), device=device, dtype=points.dtype)],
        dim=1,
    )
    normals = normals / torch.norm(normals, dim=1, keepdim=True).clamp_min(1e-12)

    collected = []
    total = 0
    attempts = 0
    max_attempts = 2000

    while total < num_samples and attempts < max_attempts:
        attempts += 1
        x = torch.randn((proposal_batch, points.shape[1]), device=device)
        top_idx = topk_assignments(
            points=points,
            h=h,
            x_samples=x,
            point_batch_size=point_batch_size,
            topk=topk,
        )

        if top_idx.shape[0] < 2:
            continue

        i0 = top_idx[0].repeat(topk - 1)
        i1 = torch.cat([top_idx[k] for k in range(1, topk)], dim=0)

        valid = (i0 >= 0) & (i1 >= 0)
        if not torch.any(valid):
            continue
        i0 = i0[valid]
        i1 = i1[valid]

        cossim = torch.sum(normals[i0] * normals[i1], dim=1).clamp(-1.0, 1.0)
        theta = torch.acos(cossim)
        keep = theta <= angle_threshold
        if not torch.any(keep):
            continue

        i0 = i0[keep]
        i1 = i1[keep]

        perm = torch.randperm(i0.shape[0], device=device)
        i0 = i0[perm]
        i1 = i1[perm]

        z_interp = (1.0 - dissim) * points[i0] + dissim * points[i1]
        z_anchor = points[i0]
        candidates = torch.cat([z_interp, z_anchor], dim=0)

        need = num_samples - total
        take = min(need, candidates.shape[0])
        if take <= 0:
            break

        collected.append(candidates[:take].detach().cpu())
        total += take

    if total < num_samples:
        raise RuntimeError(
            f"Could not generate enough OT samples. requested={num_samples}, got={total}, attempts={attempts}"
        )

    output_latent_file = Path(args.output_latent_file).resolve()
    ensure_dir(output_latent_file.parent)
    latents = torch.cat(collected, dim=0)[:num_samples]
    torch.save(latents, output_latent_file)

    save_json(
        {
            "seed": args.seed,
            "h_checkpoint": str(Path(args.h_checkpoint).resolve()),
            "output_latent_file": str(output_latent_file),
            "num_samples": int(latents.shape[0]),
            "attempts": attempts,
            "topk": topk,
            "angle_threshold": angle_threshold,
            "dissimilarity": dissim,
        },
        ot_root / f"sample_{output_latent_file.stem}.json",
    )


if __name__ == "__main__":
    main()
