# Shared-Latent FM vs AE-OT Fair Comparison (CelebA-64)

## Goal
Build a clean experiment track where Flow Matching (FM) and AE-OT use one shared autoencoder `(E, D)` and one shared latent space `Z`, so the comparison is method-only and not representation-dependent.

## Mathematical Formulation

### Shared latent empirical target
Given training images `x_i`, train one AE and freeze it:
- `z_i = E(x_i)`
- `nu = (1/n) * sum_i delta(z_i)`

Use a common latent source `mu = N(0, I)`.

### FM in latent space
Use CondOT affine path:

```math
X_t = alpha_t X_1 + sigma_t X_0, \quad alpha_t = t, \ sigma_t = 1 - t
```

with coupling `(X_0, X_1) ~ pi_{0,1}`, default independent `pi_{0,1} = mu \otimes nu`.

Train `u_t^theta` with conditional Flow Matching objective:

```math
L_{CFM}(\theta) = E_{t,(X_0,X_1)\sim\pi_{0,1}} ||u_t^\theta(X_t) - \dot{X}_t||^2
```

For CondOT:

```math
\dot{X}_t = X_1 - X_0
```

Sample by solving ODE `dX_t/dt = u_t^theta(X_t)` from `X_0 ~ mu` to `X_1`, then decode `x_hat = D(X_1)`.

### OT in latent space
Use same latent targets `P_i = z_i` and optimize semi-discrete OT dual variable `h`.

Dual score / cell assignment:

```math
i^*(x) = argmax_i { <P_i, x> + h_i }, \quad x \sim mu
```

Dual energy (uniform target weights):

```math
E(h) = E_{x\sim\mu}[max_i(<P_i, x> + h_i)] - (1/n) * sum_i h_i
```

Gradient estimate via Monte Carlo cell masses `g_i`:

```math
\nabla_{h_i} E(h) = g_i - nu_i, \quad nu_i = 1/n
```

Sample latent points via the learned assignment/interpolation rule, then decode with same decoder `D`.

## Fairness Constraints
- Same preprocessing and dataset splits.
- Same AE checkpoint and latent cache for both methods.
- Same latent dimension, same source `mu`, same decoder `D`.
- Same generated sample count per evaluation point.
- Same seeds.

## Training Pipeline
1. Train shared AE once per seed (`train_shared_ae.py`).
2. Freeze AE and encode train/test latents (`encode_latents.py`).
3. Train latent OT with adaptive MC budget and save schedule/checkpoints (`train_latent_ot.py`).
4. Train latent FM with mirrored `N_k` schedule from OT (`train_latent_fm.py`).
5. Sample latent generations for FM/OT checkpoints (`sample_fm.py`, `sample_ot.py`).
6. Decode latent generations (`decode_samples.py`).
7. Evaluate image + latent metrics and aggregate (`evaluate.py`, `run_compare.py`).

## Budget Synchronization Rule
Primary control is per-step Monte Carlo budget `N_k`.

Initialize both branches with `N_0 = base_N`.

OT escalation rule:
- If `E(h)` does not decrease by at least `ot_energy_min_delta` for `ot_energy_patience` steps,
- and remaining doubling budget exists,
- set `N_{k+1} = 2 * N_k`.

FM synchronization rule:

```math
N_k^{FM} = N_k^{OT} \quad \forall k
```

including OT-triggered doublings.

Compare on matched cumulative budgets:

```math
B_K = sum_{k=1..K} N_k
```

## Evaluation Protocol
- Image-space: FID, KID, Precision/Recall (optional, dependency-gated).
- AE reconstruction sanity: MSE, PSNR, LPIPS (optional), SSIM (optional fallback).
- Latent diagnostics: RBF-MMD, sliced Wasserstein.
- Compute diagnostics: wall-clock time, total budget, doubling events.
- Statistical report: mean/std across seeds and paired FM-OT deltas at each matched `B_K`.

## Ablation/Failure Analysis
- `sync_fm_to_ot=false` ablation (expected fairness violation).
- No-doubling OT ablation (`max_doublings=0`).
- Different source `mu` ablation (Gaussian vs uniform cube if desired).
- Sensitivity to OT interpolation controls (`angle_threshold`, `dissimilarity`).
- Failure checks: latent collapse, empty OT filter set, AE decode saturation.

## Acceptance Criteria
- Single shared AE checkpoint is used by both FM and OT in each seed run.
- Both FM and OT train only in latent space.
- OT budget log proves adaptive `N_k` schedule.
- FM log proves exact mirrored `N_k` and matched cumulative `B_K`.
- Summary outputs include checkpoint-wise quality curves and paired FM-OT deltas.

## Run Commands
Single command orchestration:

```bash
python experiments/latent_fm_ot_v2/run_compare.py \
  --config experiments/latent_fm_ot_v2/configs/celeba64_fair.yaml
```

Run per stage (manual):

```bash
python experiments/latent_fm_ot_v2/train_shared_ae.py --config ... --seed 0
python experiments/latent_fm_ot_v2/encode_latents.py --config ... --seed 0
python experiments/latent_fm_ot_v2/train_latent_ot.py --config ... --seed 0
python experiments/latent_fm_ot_v2/train_latent_fm.py --config ... --seed 0 --ot_schedule_json <.../ot_schedule.json>
python experiments/latent_fm_ot_v2/sample_ot.py --config ... --seed 0 --h_checkpoint <.../h_step_*.pt>
python experiments/latent_fm_ot_v2/sample_fm.py --config ... --seed 0 --fm_checkpoint <.../model_step_*.pt>
python experiments/latent_fm_ot_v2/decode_samples.py --config ... --seed 0 --latent_file <...>.pt --output_dir <...>
python experiments/latent_fm_ot_v2/evaluate.py --config ... --seed 0 --real_dir <...> --fm_dir <...> --ot_dir <...>
```
