#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from common import (
    build_autoencoder_from_config,
    image_transform,
    imagefolder_loader,
    load_config,
    psnr_from_mse,
    resolve_device,
    resolve_pretrained_ae_checkpoint,
    save_json,
    set_seed,
    to_uint8_image,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        self.to_tensor = ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            return self.to_tensor(img.convert("RGB"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FM vs OT decoded outputs and latent diagnostics.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--real_dir", required=True, type=str)
    parser.add_argument("--fm_dir", required=True, type=str)
    parser.add_argument("--ot_dir", required=True, type=str)
    parser.add_argument("--real_latent", required=True, type=str)
    parser.add_argument("--fm_latent", required=True, type=str)
    parser.add_argument("--ot_latent", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    parser.add_argument("--ae_checkpoint", default="", type=str)
    parser.add_argument("--device", default="", type=str)
    return parser.parse_args()


def _collect_images(root: Path) -> List[Path]:
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    paths.sort()
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


def _sample_paths(paths: List[Path], k: int, seed: int) -> List[Path]:
    if k <= 0 or len(paths) <= k:
        return paths
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(paths)), k=k))
    return [paths[i] for i in idx]


def _build_loader(paths: List[Path], batch_size: int) -> DataLoader:
    ds = ImagePathDataset(paths)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _compute_fid_kid(
    real_paths: List[Path],
    fake_paths: List[Path],
    device: torch.device,
    batch_size: int,
    compute_fid: bool,
    compute_kid: bool,
    kid_subsets: int,
    kid_subset_size: int,
) -> Dict[str, float | None | str]:
    out: Dict[str, float | None | str] = {
        "fid": None,
        "kid_mean": None,
        "kid_std": None,
        "status": "ok",
    }
    if not compute_fid and not compute_kid:
        out["status"] = "skipped"
        return out

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        out["status"] = "torchmetrics_unavailable"
        return out

    real_loader = _build_loader(real_paths, batch_size=batch_size)
    fake_loader = _build_loader(fake_paths, batch_size=batch_size)

    if compute_fid:
        try:
            fid = FrechetInceptionDistance(normalize=True).to(device)
            with torch.no_grad():
                for batch in real_loader:
                    fid.update(batch.to(device), real=True)
                for batch in fake_loader:
                    fid.update(batch.to(device), real=False)
            out["fid"] = float(fid.compute().detach().cpu().item())
        except Exception as ex:
            out["status"] = f"fid_error:{type(ex).__name__}"

    if compute_kid:
        try:
            kid = KernelInceptionDistance(
                feature=2048,
                subsets=kid_subsets,
                subset_size=kid_subset_size,
                normalize=True,
            ).to(device)
            with torch.no_grad():
                for batch in real_loader:
                    kid.update(batch.to(device), real=True)
                for batch in fake_loader:
                    kid.update(batch.to(device), real=False)
            mean, std = kid.compute()
            out["kid_mean"] = float(mean.detach().cpu().item())
            out["kid_std"] = float(std.detach().cpu().item())
        except Exception as ex:
            prev = str(out.get("status", "ok"))
            tag = f"kid_error:{type(ex).__name__}"
            out["status"] = tag if prev in {"ok", "skipped"} else f"{prev};{tag}"

    return out


def _compute_precision_recall(real_dir: Path, fake_dir: Path, device: torch.device) -> Dict[str, float | None | str]:
    out: Dict[str, float | None | str] = {"precision": None, "recall": None, "status": "ok"}
    try:
        from torch_fidelity import calculate_metrics
    except Exception:
        out["status"] = "torch_fidelity_unavailable"
        return out

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=device.type == "cuda",
        isc=False,
        fid=False,
        kid=False,
        prc=True,
        verbose=False,
    )
    out["precision"] = float(metrics["precision"])
    out["recall"] = float(metrics["recall"])
    return out


def _gaussian_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    # Median heuristic for RBF width.
    z = torch.cat([x, y], dim=0)
    with torch.no_grad():
        dists = torch.cdist(z, z, p=2)
        sigma = torch.median(dists[dists > 0]).item() if torch.any(dists > 0) else 1.0
        sigma = max(sigma, 1e-6)
        gamma = 1.0 / (2.0 * sigma * sigma)

        k_xx = torch.exp(-gamma * torch.cdist(x, x, p=2) ** 2).mean()
        k_yy = torch.exp(-gamma * torch.cdist(y, y, p=2) ** 2).mean()
        k_xy = torch.exp(-gamma * torch.cdist(x, y, p=2) ** 2).mean()
        mmd2 = k_xx + k_yy - 2.0 * k_xy
    return float(mmd2.detach().cpu().item())


def _sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    vals = []
    for _ in range(n_proj):
        v = rng.normal(size=(dim,))
        v /= np.linalg.norm(v) + 1e-12
        px = np.sort(x @ v)
        py = np.sort(y @ v)
        m = min(px.shape[0], py.shape[0])
        vals.append(np.mean(np.abs(px[:m] - py[:m])))
    return float(np.mean(vals))


def _compute_latent_metrics(real_latent: Path, fake_latent: Path, max_count: int, seed: int, n_proj: int) -> Dict[str, float]:
    real = torch.load(real_latent, map_location="cpu").to(torch.float32)
    fake = torch.load(fake_latent, map_location="cpu").to(torch.float32)

    n = min(real.shape[0], fake.shape[0])
    if max_count > 0:
        n = min(n, max_count)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    real_idx = torch.randperm(real.shape[0], generator=g)[:n]
    fake_idx = torch.randperm(fake.shape[0], generator=g)[:n]

    real_n = real[real_idx]
    fake_n = fake[fake_idx]

    mmd = _gaussian_mmd(real_n, fake_n)
    swd = _sliced_wasserstein(
        real_n.numpy(),
        fake_n.numpy(),
        n_proj=n_proj,
        seed=seed,
    )
    return {
        "latent_mmd_rbf": mmd,
        "latent_swd": swd,
        "latent_count": int(n),
    }


def _compute_ae_reconstruction_metrics(
    cfg: Dict,
    device: torch.device,
    ae_checkpoint_override: str = "",
) -> Dict[str, float | None | str]:
    try:
        ckpt = resolve_pretrained_ae_checkpoint(cfg=cfg, explicit=ae_checkpoint_override)
    except (ValueError, FileNotFoundError):
        return {
            "ae_test_mse": None,
            "ae_test_psnr": None,
            "ae_test_lpips": None,
            "ae_test_ssim": None,
            "status": "missing_pretrained_ae_checkpoint",
        }

    model = build_autoencoder_from_config(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    transform = image_transform(
        center_crop_size=int(cfg["dataset"]["center_crop_size"]),
        image_size=int(cfg["dataset"]["image_size"]),
    )

    loader = imagefolder_loader(
        root=cfg["dataset"]["test_dir"],
        transform=transform,
        batch_size=int(cfg["ae"]["batch_size"]),
        num_workers=int(cfg["dataset"]["num_workers"]),
        shuffle=False,
        drop_last=False,
    )

    mse_sum = 0.0
    count = 0
    lpips_val_sum = 0.0
    lpips_count = 0
    ssim_sum = 0.0
    ssim_count = 0

    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_status = "ok"
    except Exception:
        lpips_fn = None
        lpips_status = "lpips_unavailable"

    try:
        from skimage.metrics import structural_similarity as ssim_fn

        ssim_status = "ok"
    except Exception:
        ssim_fn = None
        ssim_status = "ssim_unavailable"

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon, _ = model(images)

            x = to_uint8_image(images)
            y = to_uint8_image(recon)
            mse_batch = torch.mean((x - y) ** 2).item()
            mse_sum += mse_batch
            count += 1

            if lpips_fn is not None:
                lpips_val = lpips_fn(images, recon).mean().item()
                lpips_val_sum += lpips_val
                lpips_count += 1

            if ssim_fn is not None:
                x_np = x.cpu().numpy().transpose(0, 2, 3, 1)
                y_np = y.cpu().numpy().transpose(0, 2, 3, 1)
                for i in range(x_np.shape[0]):
                    ssim_sum += float(ssim_fn(x_np[i], y_np[i], channel_axis=2, data_range=1.0))
                    ssim_count += 1

    mse = mse_sum / max(count, 1)
    return {
        "ae_test_mse": mse,
        "ae_test_psnr": psnr_from_mse(mse),
        "ae_test_lpips": (lpips_val_sum / lpips_count) if lpips_count > 0 else None,
        "ae_test_ssim": (ssim_sum / ssim_count) if ssim_count > 0 else None,
        "lpips_status": lpips_status,
        "ssim_status": ssim_status,
        "status": "ok",
    }


def _evaluate_method(
    name: str,
    real_dir: Path,
    fake_dir: Path,
    real_paths: List[Path],
    max_images: int,
    seed: int,
    cfg: Dict,
    device: torch.device,
) -> Dict[str, object]:
    fake_paths_all = _collect_images(fake_dir)
    k = min(len(real_paths), len(fake_paths_all))
    if bool(cfg["eval"].get("equal_sample_count", True)):
        if max_images > 0:
            k = min(k, max_images)
    elif max_images > 0:
        k = min(max_images, len(fake_paths_all))

    fake_paths = _sample_paths(fake_paths_all, k=k, seed=seed)
    real_subset = _sample_paths(real_paths, k=k, seed=seed)

    img_metrics = _compute_fid_kid(
        real_paths=real_subset,
        fake_paths=fake_paths,
        device=device,
        batch_size=64,
        compute_fid=bool(cfg["eval"].get("compute_fid", False)),
        compute_kid=bool(cfg["eval"].get("compute_kid", False)),
        kid_subsets=int(cfg["eval"].get("kid_subsets", 10)),
        kid_subset_size=int(cfg["eval"].get("kid_subset_size", 100)),
    )

    pr_metrics = (
        _compute_precision_recall(real_dir=real_dir, fake_dir=fake_dir, device=device)
        if bool(cfg["eval"].get("compute_pr", False))
        else {"precision": None, "recall": None, "status": "disabled"}
    )

    return {
        "method": name,
        "fake_dir": str(fake_dir),
        "num_fake_images_total": len(fake_paths_all),
        "num_images_used": len(fake_paths),
        **img_metrics,
        **pr_metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device or None)

    real_dir = Path(args.real_dir).resolve()
    fm_dir = Path(args.fm_dir).resolve()
    ot_dir = Path(args.ot_dir).resolve()

    real_paths_all = _collect_images(real_dir)
    max_eval = int(cfg["eval"].get("max_eval_images", 0))

    fm_metrics = _evaluate_method(
        name="fm",
        real_dir=real_dir,
        fake_dir=fm_dir,
        real_paths=real_paths_all,
        max_images=max_eval,
        seed=args.seed,
        cfg=cfg,
        device=device,
    )
    ot_metrics = _evaluate_method(
        name="ot",
        real_dir=real_dir,
        fake_dir=ot_dir,
        real_paths=real_paths_all,
        max_images=max_eval,
        seed=args.seed,
        cfg=cfg,
        device=device,
    )

    latent_real = Path(args.real_latent).resolve()
    latent_fm = Path(args.fm_latent).resolve()
    latent_ot = Path(args.ot_latent).resolve()

    latent_eval_count = int(cfg["eval"].get("max_eval_images", 0))
    n_proj = int(cfg["eval"].get("latent_swd_projections", 128))

    fm_latent_metrics = _compute_latent_metrics(
        real_latent=latent_real,
        fake_latent=latent_fm,
        max_count=latent_eval_count,
        seed=args.seed,
        n_proj=n_proj,
    )
    ot_latent_metrics = _compute_latent_metrics(
        real_latent=latent_real,
        fake_latent=latent_ot,
        max_count=latent_eval_count,
        seed=args.seed,
        n_proj=n_proj,
    )

    ae_rec = _compute_ae_reconstruction_metrics(
        cfg=cfg,
        device=device,
        ae_checkpoint_override=args.ae_checkpoint,
    )

    result = {
        "seed": args.seed,
        "real_dir": str(real_dir),
        "device": str(device),
        "ae_reconstruction": ae_rec,
        "fm": {**fm_metrics, **fm_latent_metrics},
        "ot": {**ot_metrics, **ot_latent_metrics},
        "paired_delta": {
            "fid_fm_minus_ot": (
                (fm_metrics.get("fid") - ot_metrics.get("fid"))
                if isinstance(fm_metrics.get("fid"), float) and isinstance(ot_metrics.get("fid"), float)
                else None
            ),
            "kid_mean_fm_minus_ot": (
                (fm_metrics.get("kid_mean") - ot_metrics.get("kid_mean"))
                if isinstance(fm_metrics.get("kid_mean"), float) and isinstance(ot_metrics.get("kid_mean"), float)
                else None
            ),
            "latent_mmd_fm_minus_ot": fm_latent_metrics["latent_mmd_rbf"] - ot_latent_metrics["latent_mmd_rbf"],
            "latent_swd_fm_minus_ot": fm_latent_metrics["latent_swd"] - ot_latent_metrics["latent_swd"],
        },
    }

    out_path = Path(args.output_json).resolve()
    save_json(result, out_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
