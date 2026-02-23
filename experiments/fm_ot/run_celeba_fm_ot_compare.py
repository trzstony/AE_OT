#!/usr/bin/env python3
"""Run FM-vs-OT experiments on CelebA with matched per-step sample budgets.

Budget matching:
  OT per-step MC samples:  bat_size_n * num_bat
  FM per-step samples:     batch_size * accum_iter * world_size
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Dict, List


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FM-vs-OT CelebA runner")
    parser.add_argument("--config", required=True, type=str, help="Path to JSON config.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_fm", action="store_true")
    parser.add_argument("--skip_ot", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help=(
            "Reuse existing artifacts when present (OT generated images, "
            "FM checkpoint/fid_samples) instead of retraining/regenerating."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def largest_divisor_leq(n: int, limit: int) -> int:
    if n <= 0 or limit <= 0:
        raise ValueError(f"n and limit must be positive; got n={n}, limit={limit}")
    top = min(n, limit)
    for d in range(top, 0, -1):
        if n % d == 0:
            return d
    return 1


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"[RUN] {cmd_str}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_metrics(metrics_json: Path) -> Dict[str, float]:
    with metrics_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "summary.jsonl"
    csv_path = out_dir / "summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if rows:
        fieldnames = sorted(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def build_fm_budget(budget: int, world_size: int, max_batch_size: int) -> Dict[str, int]:
    if budget % world_size != 0:
        raise ValueError(
            f"Budget {budget} must be divisible by world_size {world_size} for exact match."
        )
    local_budget = budget // world_size
    batch_size = largest_divisor_leq(local_budget, max_batch_size)
    accum_iter = local_budget // batch_size
    effective = batch_size * accum_iter * world_size
    if effective != budget:
        raise RuntimeError(f"FM budget mismatch: expected {budget}, got {effective}")
    return {
        "batch_size": batch_size,
        "accum_iter": accum_iter,
        "effective_samples": effective,
    }


def build_ot_budget(budget: int, max_bat_size_n: int) -> Dict[str, int]:
    bat_size_n = largest_divisor_leq(budget, max_bat_size_n)
    num_bat = budget // bat_size_n
    effective = bat_size_n * num_bat
    if effective != budget:
        raise RuntimeError(f"OT budget mismatch: expected {budget}, got {effective}")
    return {
        "bat_size_n": bat_size_n,
        "num_bat": num_bat,
        "effective_samples": effective,
    }


def collect_image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def dir_has_images(root: Path) -> bool:
    return collect_image_count(root) > 0


def add_optional_kv_args(cmd: List[str], cfg: Dict, key_to_arg: Dict[str, str]) -> None:
    for key, arg_name in key_to_arg.items():
        if key in cfg and cfg[key] is not None:
            cmd.extend([arg_name, str(cfg[key])])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)

    python_bin = config.get("python_bin", sys.executable)
    project_root = Path(config["project_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    budgets = config["budgets"]
    seeds = config["seeds"]
    world_size = int(config.get("world_size", 1))
    if world_size != 1:
        raise ValueError(
            "This runner currently executes single-process jobs. Set world_size=1."
        )

    eval_cfg = config["evaluation"]
    real_eval_dir = Path(eval_cfg["real_eval_dir"]).resolve()
    generated_eval_samples = int(eval_cfg["generated_eval_samples"])
    enforce_equal_eval_samples = bool(eval_cfg.get("enforce_equal_eval_samples", True))

    fm_cfg = config["fm"]
    ot_cfg = config["ot"]

    evaluator_script = project_root / "experiments" / "fm_ot" / "evaluate_generated_images.py"
    fm_script = project_root / "flow_matching" / "examples" / "image" / "train.py"
    ot_script = project_root / "pyOMT" / "demo2.py"

    summary_rows: List[Dict] = []

    start = time.time()
    for budget in budgets:
        fm_budget = build_fm_budget(
            budget=budget,
            world_size=world_size,
            max_batch_size=int(fm_cfg["max_batch_size"]),
        )
        ot_budget = build_ot_budget(
            budget=budget,
            max_bat_size_n=int(ot_cfg["max_bat_size_n"]),
        )

        for seed in seeds:
            run_root = output_root / f"budget_{budget}" / f"seed_{seed}"
            fm_run_dir = run_root / "fm"
            ot_run_dir = run_root / "ot"
            metrics_dir = run_root / "metrics"
            fm_run_dir.mkdir(parents=True, exist_ok=True)
            ot_run_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)

            print(
                f"\n[INFO] budget={budget}, seed={seed}, "
                f"FM(batch={fm_budget['batch_size']}, accum={fm_budget['accum_iter']}), "
                f"OT(bat_size_n={ot_budget['bat_size_n']}, num_bat={ot_budget['num_bat']})"
            )

            if not args.skip_ot:
                ot_fake_dir = ot_run_dir / "gen_imgs"
                if args.reuse_existing and dir_has_images(ot_fake_dir):
                    print(f"[INFO] Reusing existing OT outputs at {ot_fake_dir}")
                else:
                    ot_actions = ot_cfg.get(
                        "actions",
                        ["extract_feature", "train_ot", "generate_feature", "decode_feature"],
                    )
                    ot_cmd = [python_bin, str(ot_script)]
                    for action in ot_actions:
                        ot_cmd.append(f"--{action}")
                    ot_cmd.extend(
                        [
                            "--data_root_train",
                            str(ot_cfg["data_root_train"]),
                            "--data_root_test",
                            str(ot_cfg["data_root_test"]),
                            "--result_root",
                            str(ot_run_dir),
                            "--seed",
                            str(seed),
                            "--ot_max_iter",
                            str(ot_cfg["max_iter"]),
                            "--ot_lr",
                            str(ot_cfg["lr"]),
                            "--ot_bat_size_n",
                            str(ot_budget["bat_size_n"]),
                            "--ot_num_bat",
                            str(ot_budget["num_bat"]),
                            "--ot_num_gen_x",
                            str(ot_cfg["num_gen_x"]),
                        ]
                    )
                    add_optional_kv_args(
                        ot_cmd,
                        ot_cfg,
                        {
                            "num_workers": "--num_workers",
                            "image_size": "--image_size",
                            "center_crop_size": "--center_crop_size",
                            "ae_num_epochs": "--ae_num_epochs",
                            "ae_batch_size": "--ae_batch_size",
                            "ae_learning_rate": "--ae_learning_rate",
                            "ae_dim_z": "--ae_dim_z",
                            "ae_dim_f": "--ae_dim_f",
                            "ae_l1_lambda": "--ae_l1_lambda",
                            "max_gen_samples": "--ot_max_gen_samples",
                            "angle_threshold": "--ot_angle_threshold",
                            "rec_gen_distance": "--ot_rec_gen_distance",
                            "device": "--device",
                        },
                    )
                    ot_decode_num_images = int(ot_cfg.get("decode_num_images", generated_eval_samples))
                    ot_cmd.extend(["--decode_num_images", str(ot_decode_num_images)])
                    run_command(ot_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

            if not args.skip_fm:
                fm_fake_dir = fm_run_dir / "fid_samples"
                checkpoint_epoch = int(fm_cfg["epochs"]) - 1
                checkpoint_path = fm_run_dir / f"checkpoint-{checkpoint_epoch}.pth"

                checkpoint_exists = checkpoint_path.exists()
                fake_count = collect_image_count(fm_fake_dir) if (args.reuse_existing and not args.dry_run) else 0

                if args.reuse_existing and checkpoint_exists:
                    print(f"[INFO] Reusing existing FM checkpoint at {checkpoint_path}")
                else:
                    fm_train_cmd = [
                        python_bin,
                        str(fm_script),
                        "--dataset",
                        str(fm_cfg["dataset"]),
                        "--data_path",
                        str(fm_cfg["data_path"]),
                        "--output_dir",
                        str(fm_run_dir),
                        "--seed",
                        str(seed),
                        "--batch_size",
                        str(fm_budget["batch_size"]),
                        "--accum_iter",
                        str(fm_budget["accum_iter"]),
                        "--epochs",
                        str(fm_cfg["epochs"]),
                        "--eval_frequency",
                        str(fm_cfg["eval_frequency"]),
                        "--lr",
                        str(fm_cfg["lr"]),
                        "--class_drop_prob",
                        str(fm_cfg["class_drop_prob"]),
                        "--cfg_scale",
                        str(fm_cfg["cfg_scale"]),
                        "--ode_method",
                        str(fm_cfg["ode_method"]),
                        "--num_workers",
                        str(fm_cfg["num_workers"]),
                        "--image_size",
                        str(fm_cfg.get("image_size", 64)),
                        "--center_crop_size",
                        str(fm_cfg.get("center_crop_size", 178)),
                    ]
                    ode_options = fm_cfg.get("ode_options")
                    if ode_options is None:
                        ode_options = {"step_size": fm_cfg["ode_step_size"]}
                    fm_train_cmd.extend(["--ode_options", json.dumps(ode_options)])

                    if fm_cfg.get("compute_fid_during_train", False):
                        fm_train_cmd.append("--compute_fid")
                    if fm_cfg.get("use_ema", False):
                        fm_train_cmd.append("--use_ema")
                    if fm_cfg.get("decay_lr", False):
                        fm_train_cmd.append("--decay_lr")
                    if fm_cfg.get("random_hflip", False):
                        fm_train_cmd.append("--random_hflip")
                    run_command(fm_train_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

                    checkpoint_exists = checkpoint_path.exists() if not args.dry_run else True

                if not args.dry_run and not checkpoint_exists:
                    raise FileNotFoundError(
                        f"Expected FM checkpoint not found: {checkpoint_path}. "
                        "Check FM training logs for early failures."
                    )

                need_fm_eval = True
                if args.reuse_existing and not args.dry_run and fake_count >= generated_eval_samples:
                    need_fm_eval = False
                    print(
                        f"[INFO] Reusing existing FM fid_samples at {fm_fake_dir} "
                        f"(found {fake_count} images)"
                    )

                if need_fm_eval:
                    fm_eval_cmd = [
                        python_bin,
                        str(fm_script),
                        "--dataset",
                        str(fm_cfg["dataset"]),
                        "--data_path",
                        str(fm_cfg["data_path"]),
                        "--output_dir",
                        str(fm_run_dir),
                        "--seed",
                        str(seed),
                        "--batch_size",
                        str(fm_budget["batch_size"]),
                        "--accum_iter",
                        str(fm_budget["accum_iter"]),
                        "--epochs",
                        str(fm_cfg["epochs"]),
                        "--eval_only",
                        "--resume",
                        str(checkpoint_path),
                        "--compute_fid",
                        "--save_fid_samples",
                        "--fid_samples",
                        str(generated_eval_samples),
                        "--class_drop_prob",
                        str(fm_cfg["class_drop_prob"]),
                        "--cfg_scale",
                        str(fm_cfg["cfg_scale"]),
                        "--ode_method",
                        str(fm_cfg["ode_method"]),
                        "--num_workers",
                        str(fm_cfg["num_workers"]),
                        "--image_size",
                        str(fm_cfg.get("image_size", 64)),
                        "--center_crop_size",
                        str(fm_cfg.get("center_crop_size", 178)),
                    ]
                    ode_options = fm_cfg.get("ode_options")
                    if ode_options is None:
                        ode_options = {"step_size": fm_cfg["ode_step_size"]}
                    fm_eval_cmd.extend(["--ode_options", json.dumps(ode_options)])
                    if fm_cfg.get("use_ema", False):
                        fm_eval_cmd.append("--use_ema")
                    if fm_cfg.get("random_hflip", False):
                        fm_eval_cmd.append("--random_hflip")
                    run_command(fm_eval_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

            if args.skip_metrics:
                continue

            ot_fake_dir = ot_run_dir / "gen_imgs"
            fm_fake_dir = fm_run_dir / "fid_samples"

            if args.dry_run:
                ot_available = generated_eval_samples
                fm_available = generated_eval_samples
            else:
                ot_available = 0 if args.skip_ot else collect_image_count(ot_fake_dir)
                fm_available = 0 if args.skip_fm else collect_image_count(fm_fake_dir)

            eval_samples = generated_eval_samples
            if enforce_equal_eval_samples:
                candidates = [generated_eval_samples]
                if not args.skip_ot:
                    candidates.append(ot_available)
                if not args.skip_fm:
                    candidates.append(fm_available)
                eval_samples = min(candidates)
                if eval_samples <= 0:
                    raise RuntimeError(
                        "No generated images found for fair evaluation. "
                        f"OT available={ot_available}, FM available={fm_available}."
                    )

            print(
                "[INFO] Evaluation sample budget: "
                f"target={generated_eval_samples}, OT_available={ot_available}, "
                f"FM_available={fm_available}, used={eval_samples}"
            )

            if not args.skip_ot:
                ot_metrics_json = metrics_dir / "ot_metrics.json"
                ot_metrics_cmd = [
                    python_bin,
                    str(evaluator_script),
                    "--real_dir",
                    str(real_eval_dir),
                    "--fake_dir",
                    str(ot_fake_dir),
                    "--device",
                    str(eval_cfg["device"]),
                    "--batch_size",
                    str(eval_cfg["batch_size"]),
                    "--num_workers",
                    str(eval_cfg["num_workers"]),
                    "--max_images",
                    str(eval_samples),
                    "--sample_seed",
                    str(seed),
                    "--output_json",
                    str(ot_metrics_json),
                ]
                if eval_cfg.get("compute_kid", False):
                    ot_metrics_cmd.append("--compute_kid")
                    ot_metrics_cmd.extend(
                        [
                            "--kid_subsets",
                            str(eval_cfg["kid_subsets"]),
                            "--kid_subset_size",
                            str(eval_cfg["kid_subset_size"]),
                        ]
                    )
                if eval_cfg.get("compute_pr", False):
                    ot_metrics_cmd.append("--compute_pr")

                run_command(ot_metrics_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

                if not args.dry_run:
                    ot_metrics = read_metrics(ot_metrics_json)
                    summary_rows.append(
                        {
                            "method": "ot",
                            "seed": seed,
                            "budget": budget,
                            "fm_batch_size": fm_budget["batch_size"],
                            "fm_accum_iter": fm_budget["accum_iter"],
                            "ot_bat_size_n": ot_budget["bat_size_n"],
                            "ot_num_bat": ot_budget["num_bat"],
                            "eval_samples_used": eval_samples,
                            "available_fake_images": ot_available,
                            **ot_metrics,
                        }
                    )

            if not args.skip_fm:
                fm_metrics_json = metrics_dir / "fm_metrics.json"
                fm_metrics_cmd = [
                    python_bin,
                    str(evaluator_script),
                    "--real_dir",
                    str(real_eval_dir),
                    "--fake_dir",
                    str(fm_fake_dir),
                    "--device",
                    str(eval_cfg["device"]),
                    "--batch_size",
                    str(eval_cfg["batch_size"]),
                    "--num_workers",
                    str(eval_cfg["num_workers"]),
                    "--max_images",
                    str(eval_samples),
                    "--sample_seed",
                    str(seed),
                    "--output_json",
                    str(fm_metrics_json),
                ]
                if eval_cfg.get("compute_kid", False):
                    fm_metrics_cmd.append("--compute_kid")
                    fm_metrics_cmd.extend(
                        [
                            "--kid_subsets",
                            str(eval_cfg["kid_subsets"]),
                            "--kid_subset_size",
                            str(eval_cfg["kid_subset_size"]),
                        ]
                    )
                if eval_cfg.get("compute_pr", False):
                    fm_metrics_cmd.append("--compute_pr")

                run_command(fm_metrics_cmd, cwd=project_root, env=env, dry_run=args.dry_run)

                if not args.dry_run:
                    fm_metrics = read_metrics(fm_metrics_json)
                    summary_rows.append(
                        {
                            "method": "fm",
                            "seed": seed,
                            "budget": budget,
                            "fm_batch_size": fm_budget["batch_size"],
                            "fm_accum_iter": fm_budget["accum_iter"],
                            "ot_bat_size_n": ot_budget["bat_size_n"],
                            "ot_num_bat": ot_budget["num_bat"],
                            "eval_samples_used": eval_samples,
                            "available_fake_images": fm_available,
                            **fm_metrics,
                        }
                    )

    if not args.dry_run and summary_rows:
        write_summary(summary_rows, output_root)

    elapsed = time.time() - start
    print(f"\n[INFO] Completed in {elapsed / 60:.2f} minutes.")
    if not args.dry_run:
        print(f"[INFO] Summary files: {output_root / 'summary.jsonl'} and {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
