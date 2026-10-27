import numpy as np
import torch


CAB_SAMPLE_ID_KEY = "__sample_id__"


def split_batch_inputs(x_dict):
    sample_ids = x_dict.get(CAB_SAMPLE_ID_KEY)
    model_inputs = {k: v for k, v in x_dict.items() if k != CAB_SAMPLE_ID_KEY}
    return model_inputs, sample_ids


def build_calibration_anchor(propensity, residual, eps=1e-6):
    return torch.clamp(propensity + residual, min=eps, max=1.0 - eps)


def interpolate_propensity(propensity, anchor, beta):
    return (1.0 - beta) * propensity + beta * anchor


# ----------------------------------------------------------------------------------
# Calibration / variance-control baselines (issue 22).
#
# Every baseline is a transform of the CTR-derived propensity p_hat, injected through
# the same ``propensity_override`` hook CAPC uses, so the loss structure is untouched.
# Definitions mirror ``models/diagnose_propensity_baselines.py`` (the mechanism-domain
# diagnostic of issue 01) so mechanism and downstream numbers share one caliber.
# ----------------------------------------------------------------------------------

BASELINE_CHOICES = ("snips", "clip", "hb", "platt", "isotonic")
# Baselines that need a pass over the calibration split before they can be applied.
BASELINES_NEEDING_CALIBRATION = ("clip", "hb", "platt", "isotonic")
# Baselines that consume click labels during that pass (``clip`` only reads propensities).
BASELINES_USING_LABELS = ("hb", "platt", "isotonic")


def _subsample(scores, clicks, max_rows, seed):
    if max_rows is None or max_rows <= 0 or scores.shape[0] <= max_rows:
        return scores, clicks
    rng = np.random.default_rng(seed)
    idx = rng.choice(scores.shape[0], size=int(max_rows), replace=False)
    return scores[idx], clicks[idx]


def fit_platt_scaling(scores, clicks, max_rows=5000000, seed=0, eps=1e-6):
    """p = sigmoid(a * logit(p_hat) + c), fit by logistic regression on the click label."""
    from sklearn.linear_model import LogisticRegression

    s, y = _subsample(np.asarray(scores, dtype=np.float64), np.asarray(clicks, dtype=np.float64), max_rows, seed)
    clipped = np.clip(s, eps, 1.0 - eps)
    x = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def fit_isotonic_thresholds(scores, clicks, max_rows=5000000, seed=0):
    """Isotonic regression of click on p_hat, returned as interpolation knots."""
    from sklearn.isotonic import IsotonicRegression

    s, y = _subsample(np.asarray(scores, dtype=np.float64), np.asarray(clicks, dtype=np.float64), max_rows, seed)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(s, y)
    return (np.asarray(iso.X_thresholds_, dtype=np.float32), np.asarray(iso.y_thresholds_, dtype=np.float32))


def weight_clip_threshold(scores, quantile, eps=1e-6):
    """w_max as a quantile of the inverse-propensity distribution (textbook weight clipping).

    Reads propensities only -- no click labels -- so it is strictly less information
    than the CAPC anchor uses.
    """
    s = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    return float(np.quantile(1.0 / s, float(quantile)))


def apply_snips(propensity, click_mask, eps=1e-6):
    """Self-normalised IPS.

    The caller's loss computes ``sum_clicked(w_i * l_i) / N``.  Dividing every weight by
    the batch mean of the *masked* weights ``S/N`` (``S = sum_clicked w``) turns that into
    ``sum_clicked(w_i l_i) / S`` -- the SNIPS estimator exactly.

    NOTE: dividing by the mean over *clicked* samples instead would leave an extra factor
    ``n_clicked / N`` (~the click rate), i.e. SNIPS *plus* a large CVR:CTR loss-weight
    change -- a confounded arm, not SNIPS.  Do not "simplify" this.
    """
    p = torch.clamp(propensity, min=eps, max=1.0 - eps)
    weights = (1.0 / p) * click_mask
    normalizer = weights.mean()
    if not torch.isfinite(normalizer) or normalizer <= 0:
        return p
    return torch.clamp(p * normalizer, min=eps, max=1.0 - eps)


def apply_weight_clip(propensity, w_max, eps=1e-6):
    """w <- min(1/p, w_max)  <=>  p <- max(p, 1/w_max)."""
    p = torch.clamp(propensity, min=eps, max=1.0 - eps)
    return torch.clamp(p, min=min(max(1.0 / float(w_max), eps), 1.0 - eps))


def apply_platt(propensity, a, c, eps=1e-6):
    p = torch.clamp(propensity, min=eps, max=1.0 - eps)
    logit = torch.log(p / (1.0 - p))
    return torch.clamp(torch.sigmoid(a * logit + c), min=eps, max=1.0 - eps)


def apply_isotonic(propensity, x_thresholds, y_thresholds, eps=1e-6):
    """Piecewise-linear interpolation over the isotonic knots, clipped at the ends.

    Matches ``sklearn.isotonic.IsotonicRegression.predict`` with ``out_of_bounds='clip'``.
    """
    p = torch.clamp(propensity, min=eps, max=1.0 - eps)
    if x_thresholds.numel() == 1:
        return torch.clamp(y_thresholds.expand_as(p).clone(), min=eps, max=1.0 - eps)
    q = torch.clamp(p, min=float(x_thresholds[0].item()), max=float(x_thresholds[-1].item()))
    hi = torch.clamp(torch.searchsorted(x_thresholds, q.contiguous()), min=1, max=x_thresholds.numel() - 1)
    lo = hi - 1
    x_lo, x_hi = x_thresholds[lo], x_thresholds[hi]
    y_lo, y_hi = y_thresholds[lo], y_thresholds[hi]
    span = x_hi - x_lo
    frac = torch.where(span > 0, (q - x_lo) / torch.clamp(span, min=1e-12), torch.zeros_like(q))
    return torch.clamp(y_lo + frac * (y_hi - y_lo), min=eps, max=1.0 - eps)


def compute_binned_calibration(scores, clicks, num_bins):
    scores = np.asarray(scores, dtype=np.float32)
    clicks = np.asarray(clicks, dtype=np.float32)

    if scores.ndim != 1 or clicks.ndim != 1:
        raise ValueError("scores and clicks must be 1D arrays")
    if scores.shape[0] != clicks.shape[0]:
        raise ValueError("scores and clicks must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("scores and clicks must be non-empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    bin_edges = np.quantile(scores, quantiles).astype(np.float32)
    bin_ids = np.searchsorted(bin_edges[1:-1], scores, side="right").astype(np.int64)

    clipped_scores = np.clip(scores, 1e-6, 1.0 - 1e-6).astype(np.float32)
    logits = np.log(clipped_scores / (1.0 - clipped_scores)).astype(np.float32)

    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float32)
    score_sums = np.bincount(bin_ids, weights=scores, minlength=num_bins).astype(np.float32)
    click_sums = np.bincount(bin_ids, weights=clicks, minlength=num_bins).astype(np.float32)
    logit_sums = np.bincount(bin_ids, weights=logits, minlength=num_bins).astype(np.float32)
    logit_sq_sums = np.bincount(bin_ids, weights=np.square(logits), minlength=num_bins).astype(np.float32)

    denom = np.maximum(counts, 1.0)
    mean_scores = score_sums / denom
    mean_clicks = click_sums / denom
    mean_logits = logit_sums / denom
    logit_var = np.maximum(logit_sq_sums / denom - np.square(mean_logits), 0.0)
    std_logits = np.sqrt(logit_var).astype(np.float32)
    residuals = mean_clicks - mean_scores
    anchors = np.clip(mean_clicks, 0.0, 1.0).astype(np.float32)
    return (
        bin_ids,
        residuals.astype(np.float32),
        bin_edges,
        counts,
        mean_scores.astype(np.float32),
        mean_clicks.astype(np.float32),
        anchors,
        mean_logits.astype(np.float32),
        std_logits,
    )
