# Shared-Latent FM vs AE-OT Fair Comparison (CelebA-64, Fresh v2)

## Summary
Build a new experiment track under `experiments/latent_fm_ot_v2` where both methods use the **same pretrained autoencoder** $E,D$, and all generative modeling is performed in the same latent space $ \mathcal{Z} $.  
The comparison will use FM notation from the Flow Matching guide ($p_t$, $p_{t|1}$, $u_t$, $L_{\text{CFM}}$, coupling $\pi_{0,1}$) and AE-OT notation from the semi-discrete OT setup ($\mu,\nu,\delta(z_i),h$, convex dual energy, Monte Carlo gradient updates).

This implementation uses a pre-trained AE checkpoint (no AE retraining in this folder), keeping latent representation fixed for fairness.

## 1. Mathematical Protocol (Paper-Aligned)

### 1.1 Shared latent dataset
Load one pre-trained autoencoder $E:\mathcal{X}\to\mathcal{Z}$, $D:\mathcal{Z}\to\mathcal{X}$.  
Keep both $E,D$ frozen.  
Encode train images $x_i$ once to latent codes $z_i = E(x_i)$, forming empirical latent target measure:
$$
\nu = \frac{1}{n}\sum_{i=1}^n \delta(z_i).
$$

Define a common source latent distribution $\mu$, default $ \mathcal{N}(0,I) $.

### 1.2 FM branch in latent space
Use conditional OT affine path (Flow Matching Sec. 4.7/4.8):
$$
X_t = \alpha_t X_1 + \sigma_t X_0,\quad \alpha_t=t,\ \sigma_t=1-t,
$$
with $(X_0,X_1)\sim \pi_{0,1}$, default independent coupling $ \pi_{0,1}(x_0,x_1)=\mu(x_0)\nu(x_1)$.

Train latent velocity $u_t^\theta$ using conditional FM objective:
$$
L_{\text{CFM}}(\theta)=\mathbb{E}_{t,(X_0,X_1)\sim\pi_{0,1}}
\|u_t^\theta(X_t)-\dot{X}_t\|^2,
$$
and for CondOT path $\dot{X}_t=X_1-X_0$, matching the $L_{\text{CFM}}^{\text{OT,Gauss}}$ form used in the guide.

Sampling: draw $X_0\sim\mu$, solve ODE $\frac{dX_t}{dt}=u_t^\theta(X_t)$, get $\hat z=X_1$, decode $\hat x=D(\hat z)$.

### 1.3 OT branch in latent space
Use the same latent target points $P_i=z_i$, source $\mu$, and semi-discrete OT dual variable $h$.  
Optimize the convex dual energy (AE-OT style) with Monte Carlo estimates, equivalent to power-cell / argmax form used by `pyOMT_raw.py`:
$$
\max_i \{\langle P_i, x\rangle + h_i\},\quad x\sim\mu.
$$
Gradient is cell-mass mismatch $g_i-\nu_i$ (here $\nu_i=\frac{1}{n}$).

Sampling: map source latent samples through learned OT assignment / interpolation rule to produce $\hat z$, then decode with the same $D$: $\hat x=D(\hat z)$.

## 2. Fairness Design (Your MC constraint included)

### 2.1 Shared invariants
Use identical image preprocessing and split files for AE/FM/OT.
Use one fixed latent cache file for both methods.
Use same source distribution $\mu$, same latent dimension, same decoder $D$, same number of generated samples per seed.
Use same seed set for dataloading and sampling.

### 2.2 Budget variable and adaptive schedule
Primary fairness variable is per-step Monte Carlo sample budget $N_k$.

Initialize $N_k=N_0$ for both FM and OT.  
OT rule: when OT energy does not decrease sufficiently for `patience` iterations, escalate $N_k \leftarrow 2N_k$.  
FM rule: **mirror the same $N_k$ schedule at the same global training steps** so sample budget remains aligned step-by-step.

This yields:
$$
N_k^{\text{FM}} = N_k^{\text{OT}} \ \forall k,
$$
including escalation epochs.

### 2.3 Comparison units
Report quality at equal cumulative sample budgets:
$$
B_K = \sum_{k=1}^{K} N_k.
$$
Evaluate both methods at matched $B_K$ checkpoints (not only at final epoch).

## 3. New v2 Module Layout and Interfaces

Create a clean track in `experiments/latent_fm_ot_v2` and keep old `experiments/fm_ot` untouched for traceability.

### 3.1 Planned files
`experiments/latent_fm_ot_v2/EXPERIMENT_PLAN.md`  
`experiments/latent_fm_ot_v2/configs/celeba64_fair.yaml`  
`experiments/latent_fm_ot_v2/encode_latents.py`  
`experiments/latent_fm_ot_v2/train_latent_fm.py`  
`experiments/latent_fm_ot_v2/train_latent_ot.py`  
`experiments/latent_fm_ot_v2/sample_fm.py`  
`experiments/latent_fm_ot_v2/sample_ot.py`  
`experiments/latent_fm_ot_v2/decode_samples.py`  
`experiments/latent_fm_ot_v2/evaluate.py`  
`experiments/latent_fm_ot_v2/run_compare.py`  
`experiments/latent_fm_ot_v2/run_pyomt_pretrained_ot.py`

### 3.2 Public config interface additions
Define one shared YAML schema with these top-level keys:
`dataset`, `ae`, `latent_cache`, `fm`, `ot`, `budget_schedule`, `eval`, `pyomt`, `seeds`, `paths`.

Important typed fields:
`budget_schedule.base_N: int`  
`budget_schedule.ot_energy_patience: int`  
`budget_schedule.ot_energy_min_delta: float`  
`budget_schedule.max_doublings: int`  
`budget_schedule.sync_fm_to_ot: bool=true`  
`paths.pretrained_ae_checkpoint: str`  
`eval.generated_samples: int`  
`eval.equal_sample_count: bool=true`

### 3.3 Output contract
Each run writes:
`.../seed_{s}/latent_cache/`  
`.../seed_{s}/fm/`  
`.../seed_{s}/ot/`  
`.../seed_{s}/decoded/{fm,ot}/`  
`.../seed_{s}/metrics/`  
Optional `pyOMT/demo2` artifacts under `.../seed_{s}/pyomt_demo2/`.

Global summary:
`.../summary.csv` and `.../summary.jsonl`.

## 4. Experiment Matrix and Metrics

### 4.1 Core matrix
Dataset: CelebA-64.  
Seeds: $\{0,1,2\}$.  
Budgets: matched $B_K$ checkpoints using adaptive $N_k$ schedule.

### 4.2 Metrics
Image space: FID, KID, Precision/Recall (if enabled).  
Reconstruction sanity: AE test MSE, PSNR, LPIPS (or SSIM if LPIPS unavailable).  
Latent-space diagnostics: MMD or sliced Wasserstein between generated latents and encoded real latents.  
Compute diagnostics: wall-clock time, total $B_K$, number/timing of OT doublings.

### 4.3 Statistical reporting
Report mean ± std across seeds for each matched $B_K$.  
Use paired deltas $(\text{FM}-\text{OT})$ per seed at each $B_K$.

## 5. Validation and Test Scenarios

### 5.1 Unit and consistency tests
Shape consistency for $E,D$, latent tensors, FM $X_t,\dot X_t$, OT feature tensors.
Determinism test with fixed seed for latent cache generation.
Budget parity test asserting $N_k^{FM}=N_k^{OT}$ under escalation.
Frozen-AE guard test that no AE parameters update during FM/OT training.

### 5.2 Integration tests
Smoke run with very small train subset and $N_0$ to validate full pipeline.
Mid-scale run to verify OT doubling triggers and FM mirrors schedule.
Decode/evaluate run to ensure both branches generate equal image counts and evaluator uses identical subset size.

### 5.3 Acceptance criteria
Single shared AE checkpoint is used by both branches.
Both methods train exclusively in latent space.
Budget logs prove matched $N_k$ schedule and matched cumulative $B_K$.
Final report includes matched-budget quality curves and per-seed paired deltas.

## 6. Cleanup/Restart Policy (“delete content that doesn’t make sense”)

Because you selected a fresh v2 folder, cleanup is logical rather than destructive:
Mark `experiments/fm_ot` as legacy in docs.
Do not delete legacy code now; avoid losing reproducibility history.
All new work and docs live under `experiments/latent_fm_ot_v2`.

## 7. Markdown Deliverable Spec

Generate `experiments/latent_fm_ot_v2/EXPERIMENT_PLAN.md` with these sections:
`Goal`, `Mathematical Formulation`, `Fairness Constraints`, `Training Pipeline`, `Budget Synchronization Rule`, `Evaluation Protocol`, `Ablation/Failure Analysis`, `Acceptance Criteria`, `Run Commands`.

The markdown should include the exact equations above for:
$\nu$, $X_t$, $L_{\text{CFM}}$, and OT dual/argmax interpretation.

## Assumptions and Defaults Chosen
Dataset is CelebA-64.
AE is pre-trained externally and frozen for both FM and OT.
Coupling default is independent $\pi_{0,1}=\mu\otimes\nu$.
FM path is CondOT ($\alpha_t=t,\sigma_t=1-t$).
Fairness is stepwise matched $N_k$, with OT-triggered doubling mirrored to FM.
Fresh implementation path is `experiments/latent_fm_ot_v2` (legacy retained).
