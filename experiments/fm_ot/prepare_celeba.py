#!/usr/bin/env python3
"""Download CelebA and build split folders usable by both FM and OT pipelines.

Output layout:
  <output_root>/training/face/*.jpg
  <output_root>/testing/face/*.jpg
  <output_root>/validation/face/*.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import urllib.request
import zipfile
from typing import Dict, Iterable, List, Optional


UDACITY_CELEBA_ZIP_URL = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
HF_PARTITION_URL = "https://huggingface.co/datasets/noname110/celeba/resolve/main/list_eval_partition.txt"

# Official CelebA split counts for 202,599 images.
OFFICIAL_SPLIT_COUNTS = {
    "train": 162770,
    "valid": 19867,
    "test": 19962,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebA split folders for FM-vs-OT.")
    parser.add_argument(
        "--download_root",
        type=str,
        default="/content/data/torchvision",
        help="Root directory used for source downloads/cache.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/content/data/celeba",
        help="Output root for split folders (training/testing/validation).",
    )
    parser.add_argument(
        "--link_mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "symlink", "copy"],
        help="How to populate split folders from downloaded images.",
    )
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=0,
        help="If >0, keep only this many train images (deterministically sampled).",
    )
    parser.add_argument(
        "--max_test_images",
        type=int,
        default=0,
        help="If >0, keep only this many test images (deterministically sampled).",
    )
    parser.add_argument(
        "--max_valid_images",
        type=int,
        default=0,
        help="If >0, keep only this many validation images (deterministically sampled).",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Seed used when subsampling splits.",
    )
    parser.add_argument(
        "--source_backend",
        type=str,
        default="auto",
        choices=["auto", "torchvision", "udacity_zip"],
        help=(
            "Image source backend. 'auto' tries torchvision first, then falls back to udacity_zip "
            "if Google Drive quota fails."
        ),
    )
    parser.add_argument(
        "--celeba_zip_url",
        type=str,
        default=UDACITY_CELEBA_ZIP_URL,
        help="URL of fallback celeba.zip archive.",
    )
    parser.add_argument(
        "--partition_file",
        type=str,
        default="",
        help="Optional local path to list_eval_partition.txt.",
    )
    parser.add_argument(
        "--partition_url",
        type=str,
        default=HF_PARTITION_URL,
        help="Fallback URL to fetch list_eval_partition.txt if unavailable locally.",
    )
    parser.add_argument(
        "--allow_partition_download",
        action="store_true",
        help="Allow downloading list_eval_partition.txt from --partition_url.",
    )
    return parser.parse_args()


def select_filenames(
    filenames: Iterable[str],
    max_images: int,
    seed: int,
) -> List[str]:
    selected = list(filenames)
    if max_images > 0 and len(selected) > max_images:
        rng = random.Random(seed)
        selected = sorted(rng.sample(selected, k=max_images))
    return selected


def materialize_image(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return

    if link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            # Fallback for filesystems where hard links are unsupported.
            shutil.copy2(src, dst)
            return

    if link_mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Reusing existing file: {dst}")
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, str(dst))


def find_image_root_in_extracted(extracted_root: Path) -> Path:
    candidates = [
        extracted_root / "img_align_celeba",
        extracted_root / "img_align_celeba" / "img_align_celeba",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    nested = list(extracted_root.rglob("img_align_celeba"))
    for c in nested:
        if c.is_dir():
            return c

    raise FileNotFoundError(
        f"Could not locate img_align_celeba folder under extracted archive root: {extracted_root}"
    )


def ensure_images_from_udacity_zip(download_root: Path, zip_url: str) -> Path:
    cache_root = download_root / "celeba_fallback"
    zip_path = cache_root / "celeba.zip"
    extracted_root = cache_root / "extracted"

    if not extracted_root.exists():
        extracted_root.mkdir(parents=True, exist_ok=True)

    download_file(zip_url, zip_path)

    marker = extracted_root / ".extracted_ok"
    if not marker.exists():
        print(f"Extracting archive to: {extracted_root}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_root)
        marker.write_text("ok\n", encoding="utf-8")

    image_root = find_image_root_in_extracted(extracted_root)
    num_images = len(list(image_root.glob("*.jpg")))
    print(f"Fallback image root ready: {image_root} (jpg_count={num_images})")
    return image_root


def try_torchvision_image_root(download_root: Path) -> Path:
    from torchvision.datasets import CelebA

    dataset = CelebA(
        root=str(download_root),
        split="train",
        target_type="attr",
        download=True,
    )
    image_root = Path(dataset.root) / dataset.base_folder / "img_align_celeba"
    if not image_root.exists():
        raise FileNotFoundError(f"torchvision download completed but image root missing: {image_root}")
    return image_root


def load_partition_from_text(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            split_id = int(parts[1])
            mapping[name] = split_id
    if not mapping:
        raise RuntimeError(f"No partition entries found in: {path}")
    return mapping


def deterministic_partition_from_counts(filenames: List[str]) -> Dict[str, int]:
    sorted_names = sorted(filenames)
    n = len(sorted_names)
    train_n = min(OFFICIAL_SPLIT_COUNTS["train"], n)
    valid_n = min(OFFICIAL_SPLIT_COUNTS["valid"], max(0, n - train_n))
    test_n = max(0, n - train_n - valid_n)

    mapping: Dict[str, int] = {}
    i = 0
    for name in sorted_names[i : i + train_n]:
        mapping[name] = 0
    i += train_n
    for name in sorted_names[i : i + valid_n]:
        mapping[name] = 1
    i += valid_n
    for name in sorted_names[i : i + test_n]:
        mapping[name] = 2

    return mapping


def maybe_download_partition_file(args: argparse.Namespace, cache_root: Path) -> Optional[Path]:
    if args.partition_file:
        path = Path(args.partition_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"partition_file not found: {path}")
        return path

    if not args.allow_partition_download:
        return None

    part_path = cache_root / "list_eval_partition.txt"
    try:
        download_file(args.partition_url, part_path)
        return part_path
    except Exception as ex:
        print(f"[WARN] Failed to download partition file from {args.partition_url}: {ex}")
        return None


def build_split_mapping(args: argparse.Namespace, image_root: Path) -> Dict[str, int]:
    # Try official partition via local/downloaded text first.
    cache_root = Path(args.download_root).resolve() / "celeba_fallback"
    part_path = maybe_download_partition_file(args, cache_root)
    if part_path is not None:
        try:
            mapping = load_partition_from_text(part_path)
            print(f"Using partition file: {part_path}")
            return mapping
        except Exception as ex:
            print(f"[WARN] Could not parse partition file {part_path}: {ex}")

    # Fallback: deterministic partition by sorted filename.
    filenames = [p.name for p in image_root.glob("*.jpg")]
    if not filenames:
        raise RuntimeError(f"No JPG images found in image root: {image_root}")
    print("Using deterministic split by filename order (fallback mode).")
    return deterministic_partition_from_counts(filenames)


def build_split_lists(split_map: Dict[str, int]) -> Dict[str, List[str]]:
    out = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for name, split_id in split_map.items():
        if split_id == 0:
            out["train"].append(name)
        elif split_id == 1:
            out["valid"].append(name)
        elif split_id == 2:
            out["test"].append(name)
    for key in out:
        out[key].sort()
    return out


def materialize_split(
    image_root: Path,
    output_root: Path,
    split_names: List[str],
    split_dir_name: str,
    link_mode: str,
    max_images: int,
    seed: int,
) -> int:
    selected = select_filenames(split_names, max_images=max_images, seed=seed)
    out_dir = output_root / split_dir_name / "face"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(selected, start=1):
        src = image_root / filename
        if not src.exists():
            continue
        dst = out_dir / filename
        materialize_image(src, dst, link_mode)
        if idx % 10000 == 0:
            print(f"[{split_dir_name}] prepared {idx}/{len(selected)} images")

    print(f"[{split_dir_name}] done: {len(selected)} images -> {out_dir} (mode={link_mode})")
    return len(selected)


def resolve_image_root(args: argparse.Namespace, download_root: Path) -> Path:
    if args.source_backend == "torchvision":
        return try_torchvision_image_root(download_root)

    if args.source_backend == "udacity_zip":
        return ensure_images_from_udacity_zip(download_root=download_root, zip_url=args.celeba_zip_url)

    # auto
    try:
        print("Trying torchvision CelebA download...")
        return try_torchvision_image_root(download_root)
    except Exception as ex:
        print(f"[WARN] torchvision CelebA download failed: {ex}")
        print("Falling back to celeba.zip mirror.")
        return ensure_images_from_udacity_zip(download_root=download_root, zip_url=args.celeba_zip_url)


def main() -> None:
    args = parse_args()

    download_root = Path(args.download_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_root = resolve_image_root(args, download_root)
    split_map = build_split_mapping(args, image_root=image_root)
    split_lists = build_split_lists(split_map)

    counts: Dict[str, int] = {}
    counts["train"] = materialize_split(
        image_root=image_root,
        output_root=output_root,
        split_names=split_lists["train"],
        split_dir_name="training",
        link_mode=args.link_mode,
        max_images=args.max_train_images,
        seed=args.sample_seed,
    )
    counts["valid"] = materialize_split(
        image_root=image_root,
        output_root=output_root,
        split_names=split_lists["valid"],
        split_dir_name="validation",
        link_mode=args.link_mode,
        max_images=args.max_valid_images,
        seed=args.sample_seed,
    )
    counts["test"] = materialize_split(
        image_root=image_root,
        output_root=output_root,
        split_names=split_lists["test"],
        split_dir_name="testing",
        link_mode=args.link_mode,
        max_images=args.max_test_images,
        seed=args.sample_seed,
    )

    print("Preparation complete.")
    print(
        "Summary: "
        f"train={counts['train']}, valid={counts['valid']}, test={counts['test']}, "
        f"output_root={output_root}, image_root={image_root}"
    )


if __name__ == "__main__":
    main()
