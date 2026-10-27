# Adversarial Propensity Learning (APL)

Code for the paper **"Adversarial Propensity Learning for Robust CVR Modeling under Propensity Misspecification"**.

APL wraps inverse-propensity-weighted CVR debiasing objectives (IPS / DR / ESCM²-IPS / ESCM²-DR) with a min–max objective over a propensity uncertainty set around the CTR estimate, and trains against the worst-case propensity in the set.

## Variants

| Variant | Uncertainty set | Adversary | Key hyperparameters |
|---|---|---|---|
| APL-Box-H | Per-sample interval from CTR mean/std $(\mu, \sigma)$ | Lower-endpoint clipping | $\alpha$ |
| APL-Box-MC | Per-sample interval from MC-Dropout $(\mu, \sigma)$ | Lower-endpoint clipping | $\alpha$, $T$ |
| APL-KL | KL ball over IPS-induced weights | Exponential tilt | $\lambda$ |
| APL-Hybrid-H / -MC | Box interval, then KL tilt | Two-step chain | $\alpha$, $\lambda$ ($T$) |

Implementation notes (consistent with the paper): adversarial weights are treated as constants w.r.t. model parameters (no gradient flows through the adversary); Box-variant inverse weights are capped at 100; the KL adversary normalizes weights within the clicked subset of each mini-batch; all propensities are floored at $10^{-6}$.

## Requirements

- Python 3.10+
- PyTorch (developed with 2.x)
- pandas, numpy, scikit-learn

```bash
pip install torch pandas numpy scikit-learn
```

## Data

Both datasets are public:

- **Ali-CCP** — download from the [Alibaba challenge page](https://tianchi.aliyun.com/dataset/646) (search "Ali-CCP"). Place `ali_ccp_train.csv`, `ali_ccp_val.csv`, `ali_ccp_test.csv` under `data/ali-ccp/`.
- **AliExpress** — regional subsets (ES / FR / NL / US). Place `<region>_train.csv` etc. under `data/ali-express/<region>/`.

The loaders in `models/utils.py` cache preprocessed arrays on first run.

## Reproduce

### 1. Main results (Table: 6 variants × 4 base estimators × 5 datasets)

Single run (Ali-CCP, IPS base, APL-KL, 5 seeds):

```bash
python3 models/mmoe_aliccp.py --loss_type ips --apl_method kl --apl_lambda 1.0 --seeds 0 1 2 3 4
```

Full grid: `bash models/run_exp.sh` (edit the dataset/estimator/variant lists at the top).

Variant switches: `--apl_method {heuristic, mc_dropout, kl, heuristic_kl, mc_dropout_kl}`, `--apl_alpha`, `--apl_lambda`, `--mcdp_n` (MC passes $T$). AliExpress: `models/mmoe_aliexpress.py --region ES|FR|NL|US ...`.

### 2. Bias injection (RQ3: scaling / shift / noise)

```bash
python3 models/run_bias_experiments.py            # or: bash models/run_bias_exp_parallel.sh
```

Perturbs the estimated propensity $\hat p$ via $\hat p^\gamma$, $\hat p(1+\delta)$, $\hat p(1+\epsilon)$ with $\gamma\in\{0.8,...,1.2\}$, $\delta\in\{-0.2,...,0.2\}$, $\sigma_\epsilon\in\{0,...,0.2\}$.

### 3. Sensitivity (RQ4: $\alpha$ / $\lambda$ sweeps)

```bash
for a in 0.0 0.5 1.0 1.5 2.0; do
  python3 models/mmoe_aliccp.py --loss_type ips --apl_method heuristic --apl_alpha $a --seeds 0 1 2 3 4
done
# likewise --apl_method kl --apl_lambda in {0.1, 0.5, 1.0, 5.0, 10.0}
```

### 4. Diagnostics (bin homogeneity / dispersion / tail)

```bash
bash models/run_propensity_proxy_profile.sh       # trains a vanilla model, dumps the CTR-proxy profile
python3 models/analyze_propensity_proxy_bias.py   # per-bin stats: chi-square, residuals, weight tails
```

## Evaluation protocol

CVR score vs. the `purchase` label on the **full test set** (unclicked impressions have label 0); checkpoint selection per run by validation CVR AUC; all numbers are mean ± std over 5 seeds. Each training script prints per-seed results and the final aggregate.

## Repository structure

```
models/                  training & analysis entry points
  mmoe_aliccp.py         Ali-CCP runner (all APL variants)
  mmoe_aliexpress.py     AliExpress runner
  run_exp.sh             main-result grid
  run_bias_experiments.py / run_bias_exp_parallel.sh     RQ3 bias injection
  run_propensity_proxy_profile.sh / analyze_propensity_proxy_bias.py   diagnostics
  utils.py               dataset configs, loaders, metrics
torch_rechub_main/       customized torch-rechub fork
  torch_rechub/trainers/loss.py       APL weighting (Box / KL / Hybrid) over IPS/DR/ESCM² losses
  torch_rechub/trainers/mtl_trainer.py  multi-task trainer (MMoE backbone, early stopping)
  torch_rechub/models/multi_task/       MMOE / ESMM / AITM / PLE towers
```

## License

MIT (the `torch_rechub_main` subtree retains the upstream [torch-rechub](https://github.com/datawhalechina/torch-rechub) MIT license; APL modifications under the same terms).
