#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${CONFIG_PATH:=experiments/fm_ot/config.colab.json}"
: "${DOWNLOAD_ROOT:=/content/data/torchvision}"
: "${CELEBA_OUTPUT_ROOT:=/content/data/celeba}"
: "${MAX_TRAIN_IMAGES:=50000}"
: "${MAX_TEST_IMAGES:=10000}"
: "${MAX_VALID_IMAGES:=0}"
: "${SAMPLE_SEED:=0}"
: "${LINK_MODE:=hardlink}"
: "${DRY_RUN:=0}"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Installing dependencies"
pip install -q -e ./flow_matching
pip install -q torchmetrics torch-fidelity scipy pillow gdown

echo "[INFO] Preparing CelebA dataset"
python experiments/fm_ot/prepare_celeba.py \
  --download_root "$DOWNLOAD_ROOT" \
  --output_root "$CELEBA_OUTPUT_ROOT" \
  --link_mode "$LINK_MODE" \
  --max_train_images "$MAX_TRAIN_IMAGES" \
  --max_test_images "$MAX_TEST_IMAGES" \
  --max_valid_images "$MAX_VALID_IMAGES" \
  --sample_seed "$SAMPLE_SEED"

echo "[INFO] Running FM-vs-OT comparison"
if [[ "$DRY_RUN" == "1" ]]; then
  python experiments/fm_ot/run_celeba_fm_ot_compare.py --config "$CONFIG_PATH" --dry_run
else
  python experiments/fm_ot/run_celeba_fm_ot_compare.py --config "$CONFIG_PATH"
fi

echo "[INFO] Done. Summary under /content/AE_OT/experiments/fm_ot/runs/celeba_fm_vs_ot"
