# latent_fm_ot_v2 (Pretrained-AE Only)

This folder no longer trains an autoencoder.  
All AE usage is **frozen pre-trained checkpoint only**.

## Colab Setup

1. Mount Drive and clone/open this repo in Colab.
2. Put your pre-trained AE checkpoint in Drive, for example:
   `/content/drive/MyDrive/AE_OT/pretrained/Epoch_0000_sim_autoencoder_pretrained.pth`
3. Set `paths.pretrained_ae_checkpoint` in:
   `experiments/latent_fm_ot_v2/configs/celeba64_fair.yaml`
4. Set dataset paths (`dataset.train_dir`, `dataset.test_dir`) to your Colab-visible paths.
5. Optional: set `paths.precomputed_feature_file` if you already have `features.pt`.

## Run pyOMT `demo2.py` with Pretrained AE + OT Training

This wrapper implements the README flow:
`--train_ot --generate_feature --decode_feature`.

```bash
python experiments/latent_fm_ot_v2/run_pyomt_pretrained_ot.py \
  --config experiments/latent_fm_ot_v2/configs/celeba64_fair.yaml \
  --seed 0 \
  --ae_checkpoint /content/drive/MyDrive/AE_OT/pretrained/Epoch_0000_sim_autoencoder_pretrained.pth \
  --extract_feature_if_missing
```

Notes:
- `--extract_feature_if_missing` is recommended when `features.pt` is not already present in `result_root`.
- If you already have `features.pt`, set `paths.precomputed_feature_file` or pass `--feature_file`.
- No model download is performed by the wrapper. It uses the checkpoint path you provide.

## Run FM vs OT Comparison Pipeline (No AE Training Stage)

```bash
python experiments/latent_fm_ot_v2/run_compare.py \
  --config experiments/latent_fm_ot_v2/configs/celeba64_fair.yaml \
  --ae_checkpoint /content/drive/MyDrive/AE_OT/pretrained/Epoch_0000_sim_autoencoder_pretrained.pth
```
