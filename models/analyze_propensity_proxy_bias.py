import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../torch_rechub_main")))

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import AITM, ESMM, MMOE
from torch_rechub.trainers import MTLTrainer

from utils import (
    ALI_CCP_CONFIG,
    ALI_EXPRESS_CONFIG,
    calculate_metrics,
    get_ali_ccp_data_loader,
    get_aliexpress_data_loader,
    seed_everything,
)


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def safe_logloss(y_true, y_pred):
    eps = 1e-7
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def build_model(model_name, features, task_types, config, mcdp_n=None):
    tower_params = [{"dims": config.TOWER_DIMS, "dropout": config.DROPOUT}] * len(task_types)
    if model_name == "MMOE":
        return MMOE(
            features,
            task_types,
            n_expert=config.N_EXPERTS,
            expert_params={"dims": config.EXPERT_DIMS, "dropout": config.DROPOUT},
            tower_params_list=tower_params,
            mcdp_n=mcdp_n,
        )
    if model_name == "ESMM":
        return ESMM(user_features=features, item_features=[], cvr_params=tower_params[0], ctr_params=tower_params[1])
    if model_name == "AITM":
        return AITM(
            features,
            n_task=len(task_types),
            bottom_params={"dims": config.EXPERT_DIMS, "dropout": config.DROPOUT},
            tower_params_list=tower_params,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_data(args, config):
    # The preprocessing cache key hashes the absolute CSV *path* alongside mtime and size
    # (utils._build_cache_key), so running the same data from a different checkout misses a
    # perfectly good cache and re-encodes ~86M rows. --data_root lets a run point at the path
    # the cache was originally built under.
    root = Path(args.data_root).resolve() if getattr(args, "data_root", "") else Path(__file__).resolve().parents[1] / "data"
    if args.dataset == "aliccp":
        data_path = root / "ali-ccp"
        outputs = get_ali_ccp_data_loader(
            data_path=str(data_path),
            batch_size=args.batch_size or config.BATCH_SIZE,
            embed_dim=config.EMBED_DIM,
            loss_type=args.loss_type,
            cab_enable=False,
        )
    else:
        data_path = root / "ali-express"
        outputs = get_aliexpress_data_loader(
            data_path=str(data_path),
            batch_size=args.batch_size or config.BATCH_SIZE,
            embed_dim=config.EMBED_DIM,
            region=args.region,
            loss_type=args.loss_type,
            cab_enable=False,
        )
    train_dl, val_dl, test_dl, features, task_types, y_test = outputs
    profile_dl = val_dl if args.split == "val" else test_dl
    return train_dl, val_dl, test_dl, profile_dl, features, task_types, y_test


def select_features(args, features):
    dense = [f.name for f in features if isinstance(f, DenseFeature)]
    sparse = [f.name for f in features if isinstance(f, SparseFeature)]
    if args.feature_names:
        requested = parse_csv_strings(args.feature_names)
        available = {f.name for f in features}
        return [name for name in requested if name in available]
    selected_dense = dense[: args.max_dense_features] if args.max_dense_features > 0 else dense
    selected_sparse = sparse[: args.max_sparse_features] if args.max_sparse_features > 0 else sparse
    return selected_dense + selected_sparse


def train_profile_model(args, train_dl, val_dl, features, task_types, config, device):
    seed_everything(args.seed)
    model = build_model(args.model_name, features, task_types, config, mcdp_n=args.mcdp_n).to(device)
    epochs = args.epochs if args.epochs is not None else config.EPOCHS
    patience = args.earlystop_patience if args.earlystop_patience is not None else config.EARLY_STOP_PATIENCE
    trainer = MTLTrainer(
        model,
        task_types=task_types,
        optimizer_params={"lr": config.LR, "weight_decay": config.WEIGHT_DECAY},
        n_epoch=epochs,
        earlystop_patience=patience,
        earlystop_taskid=0,
        loss_type=args.loss_type,
        device=device,
        model_path=str(args.output_dir / "checkpoint"),
    )
    trainer.fit(train_dl, val_dl, mode="profile", seed=args.seed, save_model=args.save_model)
    return model


def collect_predictions_and_sample(args, model, data_loader, selected_features, device):
    total = len(data_loader.dataset)
    sample_target = min(args.feature_sample_rows, total) if args.feature_sample_rows > 0 else total
    sample_rate = 1.0 if sample_target >= total else float(sample_target) / float(total)
    rng = np.random.default_rng(args.seed + 20260413)

    p_hat_parts = []
    click_parts = []
    purchase_parts = []
    sample_parts = {"p_hat": [], "click": [], "purchase": []}
    for name in selected_features:
        sample_parts[name] = []

    model.eval()
    seen = 0
    with torch.no_grad():
        for x_dict, y in data_loader:
            batch_size = len(y)
            x_device = {k: v.to(device) for k, v in x_dict.items()}
            y_pred, _, _ = model(x_device)
            pred_np = y_pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
            p_hat = pred_np[:, 1].astype(np.float32)
            purchase = y_np[:, 0].astype(np.int8)
            click = y_np[:, 1].astype(np.int8)

            p_hat_parts.append(p_hat)
            click_parts.append(click)
            purchase_parts.append(purchase)

            if sample_rate >= 1.0:
                mask = np.ones(batch_size, dtype=bool)
            else:
                mask = rng.random(batch_size) < sample_rate
            if mask.any():
                sample_parts["p_hat"].append(p_hat[mask])
                sample_parts["click"].append(click[mask])
                sample_parts["purchase"].append(purchase[mask])
                for name in selected_features:
                    values = x_dict[name]
                    values_np = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
                    sample_parts[name].append(values_np[mask])

            seen += batch_size
            if args.max_profile_rows > 0 and seen >= args.max_profile_rows:
                break

    p_hat_all = np.concatenate(p_hat_parts)
    click_all = np.concatenate(click_parts)
    purchase_all = np.concatenate(purchase_parts)
    sample = {}
    for name, parts in sample_parts.items():
        sample[name] = np.concatenate(parts) if parts else np.array([])
    return p_hat_all, click_all, purchase_all, sample


def dump_per_sample(args, split_name, p_hat, click, purchase, sample, selected_features):
    """Persist per-sample propensity records so every downstream diagnostic (ECE/Brier,
    inverse-weight tail, Kish ESS, weight CV, weighted SMD) and every post-hoc baseline
    (SNIPS/clipping/histogram binning/Platt/isotonic/Beta/temperature/CAPC) can be computed
    without retraining. `p_hat/click/purchase` cover all profiled rows; the `sample_*` and
    `cov__*` arrays are the row-aligned covariate subsample used for covariate balance."""
    payload = {
        "p_hat": p_hat.astype(np.float32),
        "click": click.astype(np.int8),
        "purchase": purchase.astype(np.int8),
    }
    if len(sample.get("p_hat", [])):
        payload["sample_p_hat"] = sample["p_hat"].astype(np.float32)
        payload["sample_click"] = sample["click"].astype(np.int8)
        payload["sample_purchase"] = sample["purchase"].astype(np.int8)
        if args.dump_covariates:
            for name in selected_features:
                values = np.asarray(sample.get(name, []))
                if values.size == 0:
                    continue
                payload[f"cov__{name}"] = np.reshape(values, (values.shape[0], -1))[:, 0]

    out = args.output_dir / f"per_sample_{split_name}.npz"
    np.savez_compressed(out, **payload)
    print(f"Wrote per-sample dump {out} rows={len(p_hat)} cov_rows={len(payload.get('sample_p_hat', []))}", flush=True)
    return {
        "path": str(out),
        "rows": int(len(p_hat)),
        "cov_rows": int(len(payload.get("sample_p_hat", []))),
        "covariates": selected_features if args.dump_covariates else [],
    }


def build_bin_ids(values, bin_count):
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(values, quantiles)
    inner = edges[1:-1]
    if len(inner):
        inner = np.unique(inner)
    bin_ids = np.searchsorted(inner, values, side="right")
    return bin_ids.astype(np.int32), edges


def aggregate_by_group(group_ids, p_hat, click, prefix="bin"):
    rows = []
    n_total = len(p_hat)
    inv_w = 1.0 / np.clip(p_hat, 1e-6, 1.0)
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        n = int(mask.sum())
        if n == 0:
            continue
        p = p_hat[mask]
        y = click[mask].astype(np.float32)
        residual = float(y.mean() - p.mean())
        inv = inv_w[mask]
        rows.append(
            {
                prefix: int(group_id) if isinstance(group_id, (np.integer, int)) else str(group_id),
                "n": n,
                "share": n / n_total,
                "click_count": int(y.sum()),
                "mean_p_hat": float(p.mean()),
                "empirical_ctr": float(y.mean()),
                "signed_residual": residual,
                "abs_residual": abs(residual),
                "ece_contribution": n / n_total * abs(residual),
                "brier": float(np.mean((p - y) ** 2)),
                "logloss": safe_logloss(y, p),
                "inv_weight_mean": float(inv.mean()),
                "inv_weight_p95": float(np.quantile(inv, 0.95)),
                "inv_weight_p99": float(np.quantile(inv, 0.99)),
                "inv_weight_max": float(inv.max()),
            }
        )
    return pd.DataFrame(rows)


def write_score_bins(args, p_hat, click):
    score_summaries = {}
    for bin_count in args.profile_bins:
        bin_ids, edges = build_bin_ids(p_hat, bin_count)
        df = aggregate_by_group(bin_ids, p_hat, click, prefix="score_bin")
        edge_left = []
        edge_right = []
        for score_bin in df["score_bin"].tolist():
            left = edges[score_bin] if score_bin < len(edges) else np.nan
            right = edges[score_bin + 1] if score_bin + 1 < len(edges) else np.nan
            edge_left.append(left)
            edge_right.append(right)
        df.insert(1, "edge_left", edge_left)
        df.insert(2, "edge_right", edge_right)
        out = args.output_dir / f"score_bins_B{bin_count}.csv"
        df.to_csv(out, index=False)
        score_summaries[str(bin_count)] = {
            "path": str(out),
            "ece": float(df["ece_contribution"].sum()),
            "max_abs_residual": float(df["abs_residual"].max()),
            "mean_inv_weight_mean": float(df["inv_weight_mean"].mean()),
            "max_inv_weight_mean": float(df["inv_weight_mean"].max()),
            "max_inv_weight_p99": float(df["inv_weight_p99"].max()),
        }
    return score_summaries


def dense_group_ids(values, dense_bins):
    values = values.astype(np.float64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, dense_bins + 1))
    inner = np.unique(edges[1:-1])
    ids = np.searchsorted(inner, values, side="right")
    return np.array([f"q{idx}" for idx in ids], dtype=object)


def sparse_group_ids(values, topk):
    counts = Counter(values.tolist())
    top_values = {value for value, _ in counts.most_common(topk)}
    return np.array([str(v) if v in top_values else "__tail__" for v in values], dtype=object)


def write_feature_cohorts(args, sample, selected_features, features):
    if len(sample.get("p_hat", [])) == 0:
        return {"path": None, "rows": 0}
    feature_types = {f.name: ("dense" if isinstance(f, DenseFeature) else "sparse") for f in features}
    rows = []
    p_hat = sample["p_hat"].astype(np.float32)
    click = sample["click"].astype(np.int8)
    n_total = len(p_hat)
    inv_w = 1.0 / np.clip(p_hat, 1e-6, 1.0)

    for name in selected_features:
        values = sample.get(name)
        if values is None or len(values) == 0:
            continue
        if feature_types.get(name) == "dense":
            group_ids = dense_group_ids(values, args.dense_bins)
        else:
            group_ids = sparse_group_ids(values, args.topk_values)

        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            n = int(mask.sum())
            if n < args.min_cohort_n:
                continue
            p = p_hat[mask]
            y = click[mask].astype(np.float32)
            inv = inv_w[mask]
            residual = float(y.mean() - p.mean())
            ece = n / n_total * abs(residual)
            rows.append(
                {
                    "feature": name,
                    "feature_type": feature_types.get(name, "unknown"),
                    "cohort": str(group_id),
                    "n": n,
                    "sample_share": n / n_total,
                    "mean_p_hat": float(p.mean()),
                    "empirical_ctr": float(y.mean()),
                    "signed_residual": residual,
                    "abs_residual": abs(residual),
                    "ece_contribution": ece,
                    "inv_weight_mean": float(inv.mean()),
                    "inv_weight_p95": float(np.quantile(inv, 0.95)),
                    "inv_weight_p99": float(np.quantile(inv, 0.99)),
                    # Coverage-weighted calibration error: sample share times absolute residual.
                    "risk_score": float(ece),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["risk_score", "abs_residual", "n"], ascending=[False, False, False])
    out = args.output_dir / "feature_cohorts.csv"
    df.to_csv(out, index=False)
    return {"path": str(out), "rows": int(len(df))}


def write_heatmap_table(args, sample, selected_features, features):
    if not selected_features or len(sample.get("p_hat", [])) == 0:
        return {"path": None, "rows": 0}
    top_feature = selected_features[0]
    p_hat = sample["p_hat"].astype(np.float32)
    click = sample["click"].astype(np.int8)
    values = sample[top_feature]
    feature_type = "dense"
    for f in features:
        if f.name == top_feature:
            feature_type = "dense" if isinstance(f, DenseFeature) else "sparse"
            break
    score_ids, _ = build_bin_ids(p_hat, min(args.heatmap_score_bins, max(2, len(p_hat))))
    cohort_ids = dense_group_ids(values, args.dense_bins) if feature_type == "dense" else sparse_group_ids(values, args.topk_values)
    rows = []
    inv_w = 1.0 / np.clip(p_hat, 1e-6, 1.0)
    for score_bin in np.unique(score_ids):
        for cohort in np.unique(cohort_ids):
            mask = (score_ids == score_bin) & (cohort_ids == cohort)
            n = int(mask.sum())
            if n < args.min_heatmap_n:
                continue
            p = p_hat[mask]
            y = click[mask].astype(np.float32)
            inv = inv_w[mask]
            residual = float(y.mean() - p.mean())
            rows.append(
                {
                    "feature": top_feature,
                    "score_bin": int(score_bin),
                    "cohort": str(cohort),
                    "n": n,
                    "mean_p_hat": float(p.mean()),
                    "empirical_ctr": float(y.mean()),
                    "signed_residual": residual,
                    "abs_residual": abs(residual),
                    "inv_weight_p99": float(np.quantile(inv, 0.99)),
                }
            )
    df = pd.DataFrame(rows)
    out = args.output_dir / "score_x_feature_heatmap.csv"
    df.to_csv(out, index=False)
    return {"path": str(out), "rows": int(len(df)), "feature": top_feature}


def maybe_write_plots(args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}", flush=True)
        return

    for bin_count in args.profile_bins:
        path = args.output_dir / f"score_bins_B{bin_count}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["mean_p_hat"], df["empirical_ctr"], marker="o", label="empirical CTR")
        lo = min(df["mean_p_hat"].min(), df["empirical_ctr"].min())
        hi = max(df["mean_p_hat"].max(), df["empirical_ctr"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="perfect calibration")
        ax.set_xlabel("Mean predicted propensity")
        ax.set_ylabel("Empirical click rate")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"score_bin_calibration_B{bin_count}.png", dpi=200)
        plt.close(fig)

    cohort_path = args.output_dir / "feature_cohorts.csv"
    if cohort_path.exists():
        df = pd.read_csv(cohort_path)
        if not df.empty:
            top = df.head(20).iloc[::-1]
            labels = top["feature"].astype(str) + "=" + top["cohort"].astype(str)
            fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(top))))
            ax.barh(labels, top["risk_score"])
            ax.set_xlabel("Risk score (= share x |residual|)")
            fig.tight_layout()
            fig.savefig(args.output_dir / "feature_cohort_topk.png", dpi=200)
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a seed-0 model and profile propensity proxy calibration error.")
    parser.add_argument("--dataset", choices=["aliccp", "aliexpress"], required=True)
    parser.add_argument("--region", default="ES", choices=["ES", "FR", "NL", "US"])
    parser.add_argument("--model_name", default="MMOE", choices=["MMOE", "ESMM", "AITM"])
    parser.add_argument("--loss_type", default="ips", choices=["ips", "doubly_robust", "escm2_ips", "escm2_dr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--earlystop_patience", type=int, default=None)
    parser.add_argument("--mcdp_n", type=int, default=None)
    parser.add_argument("--output_root", default="analysis/propensity_proxy_bias")
    parser.add_argument("--profile_bins", default="20,50")
    parser.add_argument("--feature_sample_rows", type=int, default=1000000)
    parser.add_argument("--max_profile_rows", type=int, default=0)
    parser.add_argument("--max_dense_features", type=int, default=12)
    parser.add_argument("--max_sparse_features", type=int, default=20)
    parser.add_argument("--feature_names", default="")
    parser.add_argument("--dense_bins", type=int, default=5)
    parser.add_argument("--topk_values", type=int, default=20)
    parser.add_argument("--min_cohort_n", type=int, default=1000)
    parser.add_argument("--heatmap_score_bins", type=int, default=20)
    parser.add_argument("--min_heatmap_n", type=int, default=500)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_predictions_npz", action="store_true")
    parser.add_argument("--dump_splits", default="", help="Comma list of splits (train/val/test) to dump per-sample propensity records for; the model is trained once and reused across them")
    parser.add_argument("--dump_covariates", action="store_true", help="Include the selected-feature subsample in the per-sample dump (needed for weighted SMD covariate balance)")
    parser.add_argument("--data_root", default="", help="Override the data directory (defaults to <repo>/data). Point this at the path an existing .apl_cache was built under to reuse it -- the cache key hashes absolute paths.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.profile_bins = parse_csv_ints(args.profile_bins)
    config = ALI_CCP_CONFIG if args.dataset == "aliccp" else ALI_EXPRESS_CONFIG
    tag = args.dataset if args.dataset == "aliccp" else f"aliexpress_{args.region}"
    args.output_dir = Path(args.output_root) / tag / f"{args.loss_type}_seed{args.seed}_{args.split}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Dataset: {args.dataset} region={args.region} split={args.split} seed={args.seed}", flush=True)

    train_dl, val_dl, test_dl, profile_dl, features, task_types, _ = load_data(args, config)
    print(
        f"Batches: train={len(train_dl)} val={len(val_dl)} test={len(test_dl)} profile={len(profile_dl)}",
        flush=True,
    )
    print(f"Feature count: {len(features)}", flush=True)
    selected_features = select_features(args, features)
    print(f"Selected feature count for cohort profile: {len(selected_features)}", flush=True)
    print(f"Selected features: {selected_features}", flush=True)

    model = train_profile_model(args, train_dl, val_dl, features, task_types, config, device)
    p_hat, click, purchase, sample = collect_predictions_and_sample(args, model, profile_dl, selected_features, device)
    ctr_auc, ctr_logloss, ctr_ks = calculate_metrics(click, p_hat)
    print(f"CTR propensity proxy profile metrics - AUC: {ctr_auc:.6f}, LogLoss: {ctr_logloss:.6f}, KS: {ctr_ks:.6f}", flush=True)

    score_summary = write_score_bins(args, p_hat, click)
    cohort_summary = write_feature_cohorts(args, sample, selected_features, features)
    heatmap_summary = write_heatmap_table(args, sample, selected_features, features)

    if args.save_predictions_npz:
        np.savez_compressed(args.output_dir / "predictions_minimal.npz", p_hat=p_hat, click=click, purchase=purchase)

    dump_summaries = {}
    if args.dump_splits:
        loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
        for split_name in parse_csv_strings(args.dump_splits):
            if split_name not in loaders:
                raise ValueError(f"--dump_splits got unknown split {split_name!r}; expected train/val/test")
            if split_name == args.split:
                dump_summaries[split_name] = dump_per_sample(args, split_name, p_hat, click, purchase, sample, selected_features)
            else:
                split_p_hat, split_click, split_purchase, split_sample = collect_predictions_and_sample(args, model, loaders[split_name], selected_features, device)
                dump_summaries[split_name] = dump_per_sample(args, split_name, split_p_hat, split_click, split_purchase, split_sample, selected_features)

    maybe_write_plots(args)

    summary = {
        "dataset": args.dataset,
        "region": args.region if args.dataset == "aliexpress" else None,
        "split": args.split,
        "seed": args.seed,
        "loss_type": args.loss_type,
        "profile_rows": int(len(p_hat)),
        "feature_sample_rows": int(len(sample.get("p_hat", []))),
        "selected_features": selected_features,
        "ctr_auc": float(ctr_auc),
        "ctr_logloss": float(ctr_logloss),
        "ctr_ks": float(ctr_ks),
        "score_bins": score_summary,
        "feature_cohorts": cohort_summary,
        "heatmap": heatmap_summary,
        "per_sample_dumps": dump_summaries,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
