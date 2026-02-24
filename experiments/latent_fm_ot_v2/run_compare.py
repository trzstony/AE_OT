#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, List

from common import ensure_dir, load_config, save_jsonl, write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end shared-latent FM vs AE-OT comparison.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--seeds", default="", type=str, help="Optional comma-separated seed override.")
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_steps", default=0, type=int)
    return parser.parse_args()


def _run(cmd: List[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _flatten_method_row(
    metrics: Dict,
    method: str,
    seed: int,
    step: int,
    budget: int,
) -> Dict:
    m = metrics[method]
    return {
        "seed": seed,
        "step": step,
        "budget": budget,
        "method": method,
        "fid": m.get("fid"),
        "kid_mean": m.get("kid_mean"),
        "kid_std": m.get("kid_std"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "latent_mmd_rbf": m.get("latent_mmd_rbf"),
        "latent_swd": m.get("latent_swd"),
        "num_images_used": m.get("num_images_used"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())

    python_bin = str(cfg.get("paths", {}).get("python_bin", sys.executable))
    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    ensure_dir(output_root)

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = [int(s) for s in cfg["seeds"]]

    all_rows: List[Dict] = []
    paired_rows: List[Dict] = []

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        ensure_dir(seed_root)

        ae_summary = seed_root / "ae" / "summary.json"
        if (not args.reuse_existing) or (not ae_summary.exists()):
            cmd = [
                python_bin,
                "experiments/latent_fm_ot_v2/train_shared_ae.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if args.reuse_existing:
                cmd.append("--resume")
            _run(cmd, dry_run=args.dry_run)

        latent_summary = seed_root / "latent_cache" / "summary.json"
        if (not args.reuse_existing) or (not latent_summary.exists()):
            cmd = [
                python_bin,
                "experiments/latent_fm_ot_v2/encode_latents.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
            ]
            if args.device:
                cmd += ["--device", args.device]
            _run(cmd, dry_run=args.dry_run)

        ot_schedule_json = seed_root / "ot" / "ot_schedule.json"
        if (not args.reuse_existing) or (not ot_schedule_json.exists()):
            cmd = [
                python_bin,
                "experiments/latent_fm_ot_v2/train_latent_ot.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if args.max_steps > 0:
                cmd += ["--max_steps", str(args.max_steps)]
            _run(cmd, dry_run=args.dry_run)

        fm_schedule_json = seed_root / "fm" / "fm_schedule.json"
        if (not args.reuse_existing) or (not fm_schedule_json.exists()):
            cmd = [
                python_bin,
                "experiments/latent_fm_ot_v2/train_latent_fm.py",
                "--config",
                args.config,
                "--seed",
                str(seed),
                "--ot_schedule_json",
                str(ot_schedule_json),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if args.max_steps > 0:
                cmd += ["--max_steps", str(args.max_steps)]
            _run(cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        ot_schedule = json.loads(ot_schedule_json.read_text(encoding="utf-8"))
        fm_schedule = json.loads(fm_schedule_json.read_text(encoding="utf-8"))

        fm_ckpt_by_step = {int(c["step"]): c["fm_checkpoint"] for c in fm_schedule.get("checkpoints", [])}
        metrics_root = ensure_dir(seed_root / "metrics")
        fm_samples_root = ensure_dir(seed_root / "fm" / "samples")
        ot_samples_root = ensure_dir(seed_root / "ot" / "samples")
        decoded_fm_root = ensure_dir(seed_root / "decoded" / "fm")
        decoded_ot_root = ensure_dir(seed_root / "decoded" / "ot")

        real_latent = seed_root / "latent_cache" / "train_latents.pt"
        real_dir = Path(cfg["dataset"]["test_dir"]).expanduser().resolve()

        for ot_ck in ot_schedule.get("checkpoints", []):
            step = int(ot_ck["step"])
            budget = int(ot_ck["cumulative_budget"])
            h_ckpt = ot_ck["h_checkpoint"]

            if step not in fm_ckpt_by_step:
                raise RuntimeError(f"Missing FM checkpoint for OT checkpoint step={step}")

            fm_ckpt = fm_ckpt_by_step[step]
            fm_latent = fm_samples_root / f"fm_latents_step_{step:06d}.pt"
            ot_latent = ot_samples_root / f"ot_latents_step_{step:06d}.pt"

            fm_decoded = decoded_fm_root / f"step_{step:06d}_B_{budget}"
            ot_decoded = decoded_ot_root / f"step_{step:06d}_B_{budget}"

            metric_json = metrics_root / f"metrics_step_{step:06d}_B_{budget}.json"

            if (not args.reuse_existing) or (not fm_latent.exists()):
                cmd = [
                    python_bin,
                    "experiments/latent_fm_ot_v2/sample_fm.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--fm_checkpoint",
                    str(fm_ckpt),
                    "--output_latent_file",
                    str(fm_latent),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, dry_run=args.dry_run)

            if (not args.reuse_existing) or (not ot_latent.exists()):
                cmd = [
                    python_bin,
                    "experiments/latent_fm_ot_v2/sample_ot.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--h_checkpoint",
                    str(h_ckpt),
                    "--output_latent_file",
                    str(ot_latent),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, dry_run=args.dry_run)

            if (not args.reuse_existing) or (not (fm_decoded / "decode_summary.json").exists()):
                cmd = [
                    python_bin,
                    "experiments/latent_fm_ot_v2/decode_samples.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--latent_file",
                    str(fm_latent),
                    "--output_dir",
                    str(fm_decoded),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, dry_run=args.dry_run)

            if (not args.reuse_existing) or (not (ot_decoded / "decode_summary.json").exists()):
                cmd = [
                    python_bin,
                    "experiments/latent_fm_ot_v2/decode_samples.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--latent_file",
                    str(ot_latent),
                    "--output_dir",
                    str(ot_decoded),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, dry_run=args.dry_run)

            if (not args.reuse_existing) or (not metric_json.exists()):
                cmd = [
                    python_bin,
                    "experiments/latent_fm_ot_v2/evaluate.py",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--real_dir",
                    str(real_dir),
                    "--fm_dir",
                    str(fm_decoded),
                    "--ot_dir",
                    str(ot_decoded),
                    "--real_latent",
                    str(real_latent),
                    "--fm_latent",
                    str(fm_latent),
                    "--ot_latent",
                    str(ot_latent),
                    "--output_json",
                    str(metric_json),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, dry_run=args.dry_run)

            metrics = json.loads(metric_json.read_text(encoding="utf-8"))
            all_rows.append(_flatten_method_row(metrics=metrics, method="fm", seed=seed, step=step, budget=budget))
            all_rows.append(_flatten_method_row(metrics=metrics, method="ot", seed=seed, step=step, budget=budget))

            paired_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "budget": budget,
                    **metrics.get("paired_delta", {}),
                    "ae_test_mse": metrics.get("ae_reconstruction", {}).get("ae_test_mse"),
                    "ae_test_psnr": metrics.get("ae_reconstruction", {}).get("ae_test_psnr"),
                }
            )

    if not args.dry_run:
        save_jsonl(all_rows, output_root / "summary.jsonl")
        write_summary_csv(all_rows, output_root / "summary.csv")
        save_jsonl(paired_rows, output_root / "paired_summary.jsonl")
        write_summary_csv(paired_rows, output_root / "paired_summary.csv")


if __name__ == "__main__":
    main()
