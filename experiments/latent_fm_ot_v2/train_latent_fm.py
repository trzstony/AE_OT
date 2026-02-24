#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

from common import (
    ensure_dir,
    fm_dir,
    latent_file_paths,
    load_config,
    parse_optimizer_betas,
    resolve_device,
    save_json,
    save_jsonl,
    set_seed,
)
from models import LatentVelocityMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent FM with mirrored OT budget schedule.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--ot_schedule_json", default="", type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max_steps", default=0, type=int)
    return parser.parse_args()


def _load_schedule(args: argparse.Namespace, cfg: Dict, seed_root: Path) -> List[Dict]:
    sync = bool(cfg["budget_schedule"].get("sync_fm_to_ot", True))
    if sync:
        schedule_path = Path(args.ot_schedule_json).resolve() if args.ot_schedule_json else (seed_root / "ot" / "ot_schedule.json")
        if not schedule_path.exists():
            raise FileNotFoundError(f"sync_fm_to_ot=true but OT schedule missing: {schedule_path}")
        data = __import__("json").loads(schedule_path.read_text(encoding="utf-8"))
        return list(data["schedule"])

    base_N = int(cfg["budget_schedule"]["base_N"])
    max_steps = int(args.max_steps) if args.max_steps > 0 else int(cfg["ot"]["max_steps"])
    rows = []
    cumulative = 0
    for step in range(max_steps):
        cumulative += base_N
        rows.append({"step": step, "N_k": base_N, "cumulative_budget": cumulative})
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    fm_root = ensure_dir(fm_dir(seed_root))
    ckpt_root = ensure_dir(fm_root / "checkpoints")

    train_latents_file, _ = latent_file_paths(seed_root)
    if not train_latents_file.exists():
        raise FileNotFoundError(f"Missing train latents file: {train_latents_file}")

    x1_all = torch.load(train_latents_file, map_location="cpu").to(device=device, dtype=torch.float32)
    num_points, latent_dim = x1_all.shape

    schedule_rows = _load_schedule(args=args, cfg=cfg, seed_root=seed_root)
    if args.max_steps > 0:
        schedule_rows = schedule_rows[: int(args.max_steps)]

    model = LatentVelocityMLP(
        latent_dim=latent_dim,
        hidden_dim=int(cfg["fm"]["hidden_dim"]),
        depth=int(cfg["fm"]["depth"]),
    ).to(device)

    betas = parse_optimizer_betas(list(cfg["fm"].get("optimizer_betas", [0.9, 0.95])))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["fm"]["learning_rate"]),
        betas=betas,
        weight_decay=float(cfg["fm"].get("weight_decay", 0.0)),
    )

    max_microbatch = int(cfg["fm"].get("max_microbatch", 1024))

    # Mirror OT checkpoints by step index when available.
    ot_checkpoint_steps = set()
    ot_schedule_path = seed_root / "ot" / "ot_schedule.json"
    if ot_schedule_path.exists():
        import json

        data = json.loads(ot_schedule_path.read_text(encoding="utf-8"))
        for ck in data.get("checkpoints", []):
            ot_checkpoint_steps.add(int(ck["step"]))

    train_logs: List[Dict] = []
    checkpoint_rows: List[Dict] = []
    start = time.time()

    for step_row in schedule_rows:
        step = int(step_row["step"])
        N_k = int(step_row["N_k"])
        cumulative_budget = int(step_row["cumulative_budget"])

        model.train()
        optimizer.zero_grad(set_to_none=True)

        remaining = N_k
        weighted_loss = 0.0

        while remaining > 0:
            bs = min(remaining, max_microbatch)
            remaining -= bs

            idx = torch.randint(0, num_points, size=(bs,), device=device)
            x_1 = x1_all[idx]
            x_0 = torch.randn((bs, latent_dim), device=device)
            t = torch.rand((bs, 1), device=device)

            x_t = t * x_1 + (1.0 - t) * x_0
            target_u = x_1 - x_0

            pred_u = model(x_t, t)
            chunk_loss = torch.mean((pred_u - target_u) ** 2)

            # Weight by sample fraction so each step reflects exactly N_k samples.
            (chunk_loss * (float(bs) / float(N_k))).backward()
            weighted_loss += float(chunk_loss.detach().cpu().item()) * float(bs) / float(N_k)

        optimizer.step()

        train_logs.append(
            {
                "step": step,
                "N_k": N_k,
                "cumulative_budget": cumulative_budget,
                "loss": weighted_loss,
            }
        )

        should_checkpoint = step in ot_checkpoint_steps
        if not ot_checkpoint_steps:
            interval = int(cfg["budget_schedule"].get("checkpoint_interval_steps", 100))
            should_checkpoint = ((step + 1) % interval == 0) or (step == len(schedule_rows) - 1)

        if should_checkpoint:
            ckpt_path = ckpt_root / f"model_step_{step:06d}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                    "hidden_dim": int(cfg["fm"]["hidden_dim"]),
                    "depth": int(cfg["fm"]["depth"]),
                    "step": step,
                    "cumulative_budget": cumulative_budget,
                },
                ckpt_path,
            )
            checkpoint_rows.append(
                {
                    "step": step,
                    "N_k": N_k,
                    "cumulative_budget": cumulative_budget,
                    "fm_checkpoint": str(ckpt_path),
                }
            )

    elapsed_sec = time.time() - start

    save_jsonl(train_logs, fm_root / "train_log.jsonl")
    save_json(
        {
            "seed": args.seed,
            "elapsed_sec": elapsed_sec,
            "sync_fm_to_ot": bool(cfg["budget_schedule"].get("sync_fm_to_ot", True)),
            "num_steps": len(schedule_rows),
            "final_cumulative_budget": int(train_logs[-1]["cumulative_budget"]) if train_logs else 0,
            "checkpoints": checkpoint_rows,
        },
        fm_root / "fm_schedule.json",
    )


if __name__ == "__main__":
    main()
