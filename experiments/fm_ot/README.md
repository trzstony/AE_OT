# CelebA FM-vs-OT Comparison

This folder provides an end-to-end, Colab-ready experiment to compare:

- `FM`: flow matching (`flow_matching/examples/image/train.py`)
- `OT`: adaptive Monte Carlo OT (`pyOMT/demo2.py`)

## Fairness Protocol

The pipeline enforces fairness in three ways:

1. Same per-step sample budget (`N`):
   - OT: `ot_bat_size_n * ot_num_bat = N`
   - FM: `batch_size * accum_iter * world_size = N`
2. Same data preprocessing for both methods:
   - center crop to `center_crop_size` (default `178`)
   - resize to `image_size` (default `64x64`)
3. Same evaluation sample count per run:
   - metrics use `min(generated_eval_samples, OT_available, FM_available)` when `enforce_equal_eval_samples=true`
   - deterministic subset sampling controlled by `seed`

## Colab: One Command

From a Colab runtime where this repo is already at `/content/AE_OT`:

```bash
%cd /content/AE_OT
!bash experiments/fm_ot/run_colab_end_to_end.sh
```

That command does all of the following:

- installs dependencies
- downloads CelebA via `torchvision`
- builds split folders used by both FM and OT
- runs FM-vs-OT train/generate/evaluate
- writes summary CSV/JSONL

### Optional Runtime Knobs (Colab env vars)

```bash
%env MAX_TRAIN_IMAGES=50000
%env MAX_TEST_IMAGES=10000
%env SOURCE_BACKEND=auto
%env DRY_RUN=0
!bash experiments/fm_ot/run_colab_end_to_end.sh
```

Available env vars in `run_colab_end_to_end.sh`:

- `CONFIG_PATH` (default `experiments/fm_ot/config.colab.json`)
- `DOWNLOAD_ROOT` (default `/content/data/torchvision`)
- `CELEBA_OUTPUT_ROOT` (default `/content/data/celeba`)
- `MAX_TRAIN_IMAGES` (default `50000`)
- `MAX_TEST_IMAGES` (default `10000`)
- `MAX_VALID_IMAGES` (default `0`)
- `SAMPLE_SEED` (default `0`)
- `LINK_MODE` (`hardlink|symlink|copy`, default `hardlink`)
- `SOURCE_BACKEND` (`auto|torchvision|udacity_zip`, default `auto`)
- `ALLOW_PARTITION_DOWNLOAD` (`1|0`, default `1`)
- `PARTITION_URL` (default HF mirror of `list_eval_partition.txt`)
- `DRY_RUN` (`1` for command preview)

## Local/Custom Run

1. Prepare CelebA split folders:

```bash
python experiments/fm_ot/prepare_celeba.py \
  --download_root /path/to/torchvision \
  --output_root /path/to/celeba
```

2. Copy and edit config:

```bash
cp experiments/fm_ot/config.example.json experiments/fm_ot/config.local.json
```

3. Run:

```bash
python experiments/fm_ot/run_celeba_fm_ot_compare.py \
  --config experiments/fm_ot/config.local.json
```

## Main Files

- `experiments/fm_ot/run_colab_end_to_end.sh`: one-command Colab launcher
- `experiments/fm_ot/prepare_celeba.py`: automatic CelebA download + split preparation
- `experiments/fm_ot/run_celeba_fm_ot_compare.py`: budget-matched FM/OT runner
- `experiments/fm_ot/evaluate_generated_images.py`: FID/KID/PR evaluator with deterministic subset support
- `experiments/fm_ot/config.colab.json`: ready-to-run Colab config
- `experiments/fm_ot/config.example.json`: local template

## Outputs

Per seed/budget run:

- OT outputs: `.../budget_<N>/seed_<S>/ot/`
- FM outputs: `.../budget_<N>/seed_<S>/fm/`
- Metrics JSON: `.../budget_<N>/seed_<S>/metrics/`

Overall summary:

- `summary.jsonl`
- `summary.csv`

under `output_root`.
