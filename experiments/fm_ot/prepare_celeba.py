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
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebA split folders for FM-vs-OT.")
    parser.add_argument(
        "--download_root",
        type=str,
        default="/content/data/torchvision",
        help="Root directory used by torchvision for CelebA download/cache.",
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


def prepare_split(
    download_root: Path,
    output_root: Path,
    split: str,
    split_dir_name: str,
    link_mode: str,
    max_images: int,
    seed: int,
) -> int:
    from torchvision.datasets import CelebA

    dataset = CelebA(
        root=str(download_root),
        split=split,
        target_type="attr",
        download=True,
    )

    image_root = Path(dataset.root) / dataset.base_folder / "img_align_celeba"
    if not image_root.exists():
        raise FileNotFoundError(
            f"Could not find downloaded CelebA images under {image_root}."
        )

    output_face_dir = output_root / split_dir_name / "face"
    output_face_dir.mkdir(parents=True, exist_ok=True)

    split_filenames = select_filenames(
        filenames=dataset.filename,
        max_images=max_images,
        seed=seed,
    )

    for idx, filename in enumerate(split_filenames, start=1):
        src = image_root / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        dst = output_face_dir / filename
        materialize_image(src, dst, link_mode)
        if idx % 10000 == 0:
            print(f"[{split}] prepared {idx}/{len(split_filenames)} images")

    print(
        f"[{split}] done: {len(split_filenames)} images -> {output_face_dir} "
        f"(mode={link_mode})"
    )
    return len(split_filenames)


def main() -> None:
    args = parse_args()

    download_root = Path(args.download_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {}
    counts["train"] = prepare_split(
        download_root=download_root,
        output_root=output_root,
        split="train",
        split_dir_name="training",
        link_mode=args.link_mode,
        max_images=args.max_train_images,
        seed=args.sample_seed,
    )
    counts["valid"] = prepare_split(
        download_root=download_root,
        output_root=output_root,
        split="valid",
        split_dir_name="validation",
        link_mode=args.link_mode,
        max_images=args.max_valid_images,
        seed=args.sample_seed,
    )
    counts["test"] = prepare_split(
        download_root=download_root,
        output_root=output_root,
        split="test",
        split_dir_name="testing",
        link_mode=args.link_mode,
        max_images=args.max_test_images,
        seed=args.sample_seed,
    )

    print("Preparation complete.")
    print(
        "Summary: "
        f"train={counts['train']}, valid={counts['valid']}, test={counts['test']}, "
        f"output_root={output_root}"
    )


if __name__ == "__main__":
    main()
