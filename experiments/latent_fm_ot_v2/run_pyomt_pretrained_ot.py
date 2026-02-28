#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch

from common import ensure_dir, load_config, resolve_pretrained_ae_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyOMT/demo2.py using a pretrained AE checkpoint, "
            "then train OT + generate/decode samples."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--ae_checkpoint", default="", type=str)
    parser.add_argument("--feature_file", default="", type=str)
    parser.add_argument("--result_root", default="", type=str)
    parser.add_argument("--data_root_train", default="", type=str)
    parser.add_argument("--data_root_test", default="", type=str)
    parser.add_argument("--extract_feature_if_missing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _stage_pretrained_checkpoint(ckpt: Path, result_root: Path) -> str:
    # demo2.py searches for Epoch_*_sim_autoencoder*.pth in both models and saved_models.
    staged_name = "Epoch_0000_sim_autoencoder_pretrained.pth"
    model_dir = ensure_dir(result_root / "models")
    saved_dir = ensure_dir(result_root / "saved_models")

    for target_dir in [model_dir, saved_dir]:
        target = target_dir / staged_name
        if target.resolve() == ckpt.resolve():
            continue
        shutil.copy2(ckpt, target)
    return staged_name


def _build_demo2_common_args(
    cfg: Dict,
    seed: int,
    device: str,
    train_root: Path,
    test_root: Path,
    result_root: Path,
) -> List[str]:
    pyomt_cfg = cfg.get("pyomt", {})
    eval_cfg = cfg.get("eval", {})
    ot_cfg = cfg.get("ot", {})
    ae_cfg = cfg.get("ae", {})
    ds_cfg = cfg.get("dataset", {})

    return [
        "--seed",
        str(seed),
        "--data_root_train",
        str(train_root),
        "--data_root_test",
        str(test_root),
        "--result_root",
        str(result_root),
        "--device",
        str(device),
        "--num_workers",
        str(int(ds_cfg.get("num_workers", 4))),
        "--image_size",
        str(int(ds_cfg.get("image_size", 64))),
        "--center_crop_size",
        str(int(ds_cfg.get("center_crop_size", 178))),
        "--ae_batch_size",
        str(int(ae_cfg.get("batch_size", 256))),
        "--ae_dim_z",
        str(int(ae_cfg.get("dim_z", 100))),
        "--ae_dim_f",
        str(int(ae_cfg.get("dim_f", 80))),
        "--ot_max_iter",
        str(int(pyomt_cfg.get("ot_max_iter", 20000))),
        "--ot_lr",
        str(float(ot_cfg.get("learning_rate", 5e-2))),
        "--ot_bat_size_n",
        str(int(pyomt_cfg.get("ot_bat_size_n", 1000))),
        "--ot_num_bat",
        str(int(pyomt_cfg.get("ot_num_bat", 20))),
        "--ot_num_gen_x",
        str(int(pyomt_cfg.get("ot_num_gen_x", 20000))),
        "--ot_max_gen_samples",
        str(int(pyomt_cfg.get("ot_max_gen_samples", eval_cfg.get("generated_samples", 1000)))),
        "--ot_angle_threshold",
        str(float(ot_cfg.get("angle_threshold", 0.7))),
        "--ot_rec_gen_distance",
        str(float(ot_cfg.get("dissimilarity", 0.75))),
        "--decode_num_images",
        str(int(eval_cfg.get("generated_samples", 1000))),
    ]


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    python_bin = str(cfg.get("paths", {}).get("python_bin", sys.executable))

    project_root = Path(__file__).resolve().parents[2]
    pyomt_dir = project_root / "pyOMT"
    demo2_path = pyomt_dir / "demo2.py"
    if not demo2_path.exists():
        raise FileNotFoundError(f"Expected demo2.py at {demo2_path}")

    ckpt = resolve_pretrained_ae_checkpoint(cfg=cfg, explicit=args.ae_checkpoint)

    output_root = Path(cfg["paths"]["output_root"]).expanduser().resolve()
    seed_root = output_root / f"seed_{args.seed}"
    default_result_root = seed_root / "pyomt_demo2"
    result_root = Path(args.result_root).expanduser().resolve() if args.result_root else default_result_root
    ensure_dir(result_root)

    train_root = (
        Path(args.data_root_train).expanduser().resolve()
        if args.data_root_train
        else Path(cfg["dataset"]["train_dir"]).expanduser().resolve()
    )
    test_root = (
        Path(args.data_root_test).expanduser().resolve()
        if args.data_root_test
        else Path(cfg["dataset"]["test_dir"]).expanduser().resolve()
    )
    if not train_root.exists():
        raise FileNotFoundError(f"Training dataset directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {test_root}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    staged_name = _stage_pretrained_checkpoint(ckpt=ckpt, result_root=result_root)
    common = _build_demo2_common_args(
        cfg=cfg,
        seed=args.seed,
        device=device,
        train_root=train_root,
        test_root=test_root,
        result_root=result_root,
    )

    feature_file = result_root / "features.pt"
    feature_src = (
        Path(args.feature_file).expanduser().resolve()
        if args.feature_file
        else (
            Path(str(cfg.get("paths", {}).get("precomputed_feature_file", ""))).expanduser().resolve()
            if str(cfg.get("paths", {}).get("precomputed_feature_file", "")).strip()
            else None
        )
    )
    if feature_src is not None:
        if not feature_src.exists():
            raise FileNotFoundError(f"Configured feature file not found: {feature_src}")
        if feature_src.resolve() != feature_file.resolve():
            shutil.copy2(feature_src, feature_file)

    if args.extract_feature_if_missing and (not feature_file.exists()):
        _run(
            [python_bin, "demo2.py", "--extract_feature", *common],
            cwd=pyomt_dir,
            dry_run=args.dry_run,
        )

    _run(
        [python_bin, "demo2.py", "--train_ot", "--generate_feature", "--decode_feature", *common],
        cwd=pyomt_dir,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        save_json(
            {
                "seed": args.seed,
                "device": device,
                "pyomt_dir": str(pyomt_dir),
                "demo2": str(demo2_path),
                "python_bin": python_bin,
                "result_root": str(result_root),
                "pretrained_ae_checkpoint": str(ckpt),
                "staged_checkpoint_name": staged_name,
                "data_root_train": str(train_root),
                "data_root_test": str(test_root),
                "feature_file": str(feature_file),
                "feature_file_source": str(feature_src) if feature_src is not None else "",
                "ot_checkpoint_file": str(result_root / "h.pt"),
                "generated_feature_mat": str(result_root / "output_P_gen.mat"),
                "generated_image_dir": str(result_root / "gen_imgs"),
            },
            seed_root / "pyomt_demo2_summary.json",
        )


if __name__ == "__main__":
    main()
