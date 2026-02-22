# CelebA FM-vs-OT Comparison Runner

This folder contains a runnable experiment pipeline to compare Flow Matching (FM) and Adaptive Monte Carlo OT on CelebA with exactly matched per-step sample budgets.

Budget match is enforced as:

- OT per optimization step: `ot_bat_size_n * ot_num_bat = N`
- FM per optimization step: `batch_size * accum_iter * world_size = N`

## Files

- `run_celeba_fm_ot_compare.py`: end-to-end launcher (train, generate, evaluate, summarize).
- `evaluate_generated_images.py`: unified evaluator for generated image folders (FID, optional KID, optional precision/recall).
- `config.example.json`: template config.

## Dataset Layout

### OT (pyOMT)

`pyOMT/demo2.py` expects directory trees for:

- `ot.data_root_train`
- `ot.data_root_test`

with image files in subfolders.

### FM (flow_matching)

`flow_matching/examples/image/train.py` uses `torchvision.datasets.ImageFolder`, so `fm.data_path` must follow ImageFolder format, for example:

```text
/path/to/celeba_imagefolder/train/
  face/
    000001.jpg
    000002.jpg
    ...
```

Using one class folder (`face/`) is acceptable for unconditional generation (`class_drop_prob=1.0`, `cfg_scale=0.0`).

## Before Running

1. Copy and edit config:

```bash
cp experiments/fm_ot/config.example.json experiments/fm_ot/config.local.json
```

2. Set all paths in `config.local.json`:

- `evaluation.real_eval_dir`
- `fm.data_path`
- `ot.data_root_train`
- `ot.data_root_test`

3. Confirm OT AE checkpoint availability.

If `ot.actions` includes `extract_feature` / `decode_feature`, `pyOMT/demo2.py` should be able to load a trained AE checkpoint from its expected model folders.

## Colab Quickstart

Run these cells after `git clone`:

```bash
%cd /content
!git clone https://github.com/<your-org-or-user>/AE_OT.git
%cd /content/AE_OT
!pip install -q -e ./flow_matching
!pip install -q torchmetrics torch-fidelity scipy pillow
```

Create a Colab config:

```bash
!cp experiments/fm_ot/config.colab.example.json experiments/fm_ot/config.colab.json
```

Edit `experiments/fm_ot/config.colab.json` paths to your dataset location:

- `evaluation.real_eval_dir`
- `fm.data_path`
- `ot.data_root_train`
- `ot.data_root_test`

If you do not already have AE checkpoints for `pyOMT/demo2.py`, add `train_ae` and `refine_ae` into `ot.actions` before `extract_feature`.

If your FM data is not in ImageFolder layout, create a single-class folder:

```bash
!mkdir -p /content/data/celeba_imagefolder/train/face
!find /content/data/celeba/training -type f \( -iname "*.jpg" -o -iname "*.png" \) -exec cp -n {} /content/data/celeba_imagefolder/train/face/ \;
```

Dry run:

```bash
!python experiments/fm_ot/run_celeba_fm_ot_compare.py \
  --config experiments/fm_ot/config.colab.json \
  --dry_run
```

Full run:

```bash
!python experiments/fm_ot/run_celeba_fm_ot_compare.py \
  --config experiments/fm_ot/config.colab.json
```

## Run

Dry run (print commands only):

```bash
python experiments/fm_ot/run_celeba_fm_ot_compare.py \
  --config experiments/fm_ot/config.local.json \
  --dry_run
```

Full run:

```bash
python experiments/fm_ot/run_celeba_fm_ot_compare.py \
  --config experiments/fm_ot/config.local.json
```

Useful switches:

- `--skip_ot`
- `--skip_fm`
- `--skip_metrics`

## Outputs

Per run:

- OT outputs under `.../budget_<N>/seed_<S>/ot/`
- FM outputs under `.../budget_<N>/seed_<S>/fm/`
- Metrics JSON files under `.../budget_<N>/seed_<S>/metrics/`

Global summary:

- `summary.jsonl`
- `summary.csv`

inside `output_root`.
