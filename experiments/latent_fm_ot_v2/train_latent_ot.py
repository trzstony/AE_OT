#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from common import ensure_dir, latent_file_paths, load_config, ot_dir, resolve_device, save_json, save_jsonl, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent semi-discrete OT with adaptive MC budget.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max_steps", default=0, type=int)
    return parser.parse_args()


def compute_ot_assignments(
    points: torch.Tensor,
    h: torch.Tensor,
    x_samples: torch.Tensor,
    point_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return argmax indices and max values for x_samples over i: <P_i, x> + h_i."""
    num_points = points.shape[0]
    num_x = x_samples.shape[0]

    best_vals = torch.full((num_x,), fill_value=-1e30, device=x_samples.device, dtype=x_samples.dtype)
    best_idx = torch.full((num_x,), fill_value=-1, device=x_samples.device, dtype=torch.long)

    x_t = x_samples.transpose(0, 1)  # [d, N]

    for start in range(0, num_points, point_batch_size):
        end = min(start + point_batch_size, num_points)
        p_chunk = points[start:end]  # [m, d]
        h_chunk = h[start:end]  # [m]

        scores = p_chunk @ x_t
        scores = scores + h_chunk.unsqueeze(-1)
        chunk_vals, chunk_idx = torch.max(scores, dim=0)

        better = chunk_vals > best_vals
        best_vals = torch.where(better, chunk_vals, best_vals)
        best_idx = torch.where(better, chunk_idx + start, best_idx)

    return best_idx, best_vals


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    ot_root = ensure_dir(ot_dir(seed_root))
    ckpt_root = ensure_dir(ot_root / "checkpoints")

    train_latents_file, _ = latent_file_paths(seed_root)
    if not train_latents_file.exists():
        raise FileNotFoundError(f"Missing train latents file: {train_latents_file}")

    points = torch.load(train_latents_file, map_location="cpu").to(device=device, dtype=torch.float32)
    num_points, dim = points.shape

    h = torch.zeros((num_points,), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([h], lr=float(cfg["ot"]["learning_rate"]))

    uniform_target = torch.full((num_points,), fill_value=1.0 / float(num_points), device=device)

    base_N = int(cfg["budget_schedule"]["base_N"])
    current_N = base_N
    max_doublings = int(cfg["budget_schedule"]["max_doublings"])
    patience = int(cfg["budget_schedule"]["ot_energy_patience"])
    min_delta = float(cfg["budget_schedule"]["ot_energy_min_delta"])

    max_steps = int(args.max_steps) if args.max_steps > 0 else int(cfg["ot"]["max_steps"])
    point_batch_size = int(cfg["ot"]["point_batch_size"])
    checkpoint_interval = int(cfg["budget_schedule"].get("checkpoint_interval_steps", 100))

    best_energy = float("inf")
    stale_steps = 0
    doublings = 0

    schedule_rows: List[Dict] = []
    checkpoint_rows: List[Dict] = []
    doubling_events: List[Dict] = []
    cumulative_budget = 0

    start_time = time.time()

    for step in range(max_steps):
        x = torch.randn((current_N, dim), device=device)
        idx, max_vals = compute_ot_assignments(
            points=points,
            h=h,
            x_samples=x,
            point_batch_size=point_batch_size,
        )

        g = torch.bincount(idx, minlength=num_points).float() / float(current_N)
        grad = g - uniform_target

        energy = float((max_vals.mean() - h.mean()).detach().cpu().item())

        optimizer.zero_grad(set_to_none=True)
        h.grad = grad
        optimizer.step()
        with torch.no_grad():
            h.sub_(h.mean())

        grad_norm = float(torch.norm(grad, p=2).detach().cpu().item())
        max_abs_err = float(torch.max(torch.abs(grad)).detach().cpu().item())

        cumulative_budget += current_N
        row = {
            "step": step,
            "N_k": current_N,
            "cumulative_budget": cumulative_budget,
            "energy": energy,
            "grad_l2": grad_norm,
            "grad_max_abs": max_abs_err,
            "doublings_so_far": doublings,
        }
        schedule_rows.append(row)

        improved = (best_energy - energy) > min_delta
        if improved:
            best_energy = energy
            stale_steps = 0
        else:
            stale_steps += 1

        doubled_this_step = False
        if stale_steps > patience and doublings < max_doublings:
            old_N = current_N
            current_N = current_N * 2
            doublings += 1
            stale_steps = 0
            doubled_this_step = True
            doubling_events.append(
                {
                    "step": step,
                    "cumulative_budget": cumulative_budget,
                    "old_N": old_N,
                    "new_N": current_N,
                    "energy": energy,
                }
            )

        should_checkpoint = (
            (step + 1) % checkpoint_interval == 0
            or step == max_steps - 1
            or doubled_this_step
        )
        if should_checkpoint:
            checkpoint_path = ckpt_root / f"h_step_{step:06d}.pt"
            torch.save(h.detach().cpu(), checkpoint_path)
            checkpoint_rows.append(
                {
                    "step": step,
                    "cumulative_budget": cumulative_budget,
                    "N_k": schedule_rows[-1]["N_k"],
                    "energy": energy,
                    "h_checkpoint": str(checkpoint_path),
                }
            )

    elapsed_sec = time.time() - start_time

    schedule_json = {
        "seed": args.seed,
        "latent_points": num_points,
        "latent_dim": dim,
        "base_N": base_N,
        "sync_fm_to_ot": bool(cfg["budget_schedule"].get("sync_fm_to_ot", True)),
        "max_steps": max_steps,
        "max_doublings": max_doublings,
        "patience": patience,
        "min_delta": min_delta,
        "doublings": doublings,
        "doubling_events": doubling_events,
        "schedule": schedule_rows,
        "checkpoints": checkpoint_rows,
        "elapsed_sec": elapsed_sec,
        "final_cumulative_budget": cumulative_budget,
    }

    save_json(schedule_json, ot_root / "ot_schedule.json")
    save_jsonl(schedule_rows, ot_root / "ot_schedule.jsonl")
    save_json(
        {
            "seed": args.seed,
            "elapsed_sec": elapsed_sec,
            "final_cumulative_budget": cumulative_budget,
            "doublings": doublings,
            "final_N": schedule_rows[-1]["N_k"] if schedule_rows else base_N,
            "num_checkpoints": len(checkpoint_rows),
            "schedule_json": str(ot_root / "ot_schedule.json"),
        },
        ot_root / "summary.json",
    )


if __name__ == "__main__":
    main()
