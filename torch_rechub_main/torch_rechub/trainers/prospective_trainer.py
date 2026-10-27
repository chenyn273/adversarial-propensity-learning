"""Prospective single-pass trainer for the CAPC long-track protocol.

Spec: `.scratch/capc-recovery/specs/18-prospective-tracking.md` (D1-D7 settled 2026-07-28).

This module is a NEW layer on top of `MTLTrainer`; the old training path is left untouched so
the AAAI/ticket-01/ticket-09 numbers stay reproducible.  What lands here now is what Step 0
needs -- stage A (base warm-up) and the ordered scoring pass.  Stage B (the per-slice
prospective loop over the five arms) lands after the Step 0 gate decides the W grid.

Protocol, in short (spec sections 1.2-1.3):

    stage A   slices 0-25, E epochs, plain IPS, no correction, NO early-stop selection.
              One checkpoint per seed, shared bitwise by every arm.
    stage B   slices 26-83, strict single pass. For each slice t:
                1. score slice t with theta_{t-1}   (one forward pass, scores stored)
                2. prospective eval from those scores  (free -- no extra forward)
                3. push (score, click) into the ring buffer, rebuild the anchor
                4. train on slice t, IPS denominator built from the STORED scores

Two invariants this module enforces rather than documents:

* `score_ordered` scatters by sample id and asserts full coverage + parameter invariance.
  Ticket 01's dump appended in shuffled DataLoader order instead, which silently destroyed the
  time axis (see `analysis/prospective_tracking/anchor_window_probe.py` for the forensics).
* the IPS denominator is a stored score, so `1/p` is constant w.r.t. the current parameters by
  construction -- all arms are gradient-comparable without relying on `loss.py`'s
  `propensity_override is None` detach branch (spec section 0.2).
"""
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score

from .cab_utils import CAB_SAMPLE_ID_KEY, compute_binned_calibration, split_batch_inputs
from .loss import ips_loss
from .mtl_trainer import MTLTrainer

PCORR_KEY = "__p_corr__"          # per-row corrected propensity, carried through the batch dict
CAB_BETA_FEATURE_CLIP = 2.0       # mirrors MTLTrainer.cab_beta_feature_clip


def _logit(x, eps=1e-6):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def build_p_corr(scores, edges, anchors, std_logit, beta0, dev_scale, eps=1e-6):
    """Per-row corrected propensity, numpy mirror of MTLTrainer's own construction.

    Reproduces `_compute_cab_beta_feature` + `_current_cab_sample_beta` + `interpolate_propensity`
    (`mtl_trainer.py` 760-791) with ONE deliberate change from D2: every p_hat here is the LOGGED
    score from theta_{t-1}, not a live forward pass. `tests/test_prospective_tracking.py::T6` pins
    the equivalence.
    """
    b = np.searchsorted(edges[1:-1], scores, side="right")
    a = np.clip(np.asarray(anchors)[b], eps, 1.0 - eps)
    s = np.clip(scores, eps, 1.0 - eps)
    z = (_logit(s, eps) - _logit(a, eps)) / np.maximum(np.asarray(std_logit)[b], 1e-3)
    z = np.clip(z, -CAB_BETA_FEATURE_CLIP, CAB_BETA_FEATURE_CLIP)
    beta_raw = _logit(np.full_like(s, float(beta0)), 1e-4)
    beta = 1.0 / (1.0 + np.exp(-(beta_raw + float(dev_scale) * np.abs(z))))
    return np.clip((1.0 - beta) * s + beta * a, eps, 1.0 - eps).astype(np.float32), b


class AnchorRule:
    """Which slices feed an arm's anchor at slice t. One instance per arm.

    kind:
      none     -- no correction; p_corr = the logged score (arm 1). Routed through the same
                  injection point as every other arm, which `T1` proves is bitwise identical to
                  `propensity_override=None` (spec 0.2).
      frozen   -- a fixed slice range, built once and never refreshed (arm 2a).
      all      -- everything seen since `origin` (arm 3, W = infinity).
      rolling  -- the last `window` slices (arm 4). Clipped to >= origin during warm-up, so
                  W=26 only reaches its full width at slice origin+26.
      oracle   -- slice t itself (arm 5): uses the labels of the slice being trained on, so it is
                  an upper bound, not a method.
      platt    -- frozen 1-D logistic recalibration fit on `freeze` (arm 2b).
    """

    def __init__(self, name, kind, window=None, freeze=None, origin=26):
        self.name, self.kind, self.window, self.freeze, self.origin = name, kind, window, freeze, origin

    def slices_for(self, t):
        if self.kind == "none":
            return []
        if self.kind in ("frozen", "platt"):
            return list(range(*self.freeze))
        if self.kind == "all":
            return list(range(self.origin, t))
        if self.kind == "rolling":
            return list(range(max(self.origin, t - self.window), t))
        if self.kind == "oracle":
            return [t]
        raise ValueError(f"unknown anchor kind {self.kind}")

    def is_static(self):
        return self.kind in ("frozen", "platt")


def safe_auc(y_true, y_score):
    """AUC that returns nan instead of raising on a degenerate (single-class) input."""
    y_true = np.asarray(y_true)
    if y_true.size == 0 or y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def cvr_metrics(cvr_pred, cvr_label, click_label):
    """Both CVR AUC conventions (spec D6: the paper reports both).

    post-click  -- the standard convention: only clicked impressions enter the AUC.
    exposure    -- the convention the earlier submissions actually used (no click mask);
                   kept as a secondary column so the two are reconcilable.
    """
    cvr_pred = np.asarray(cvr_pred, dtype=np.float64)
    cvr_label = np.asarray(cvr_label)
    click_label = np.asarray(click_label)
    m = click_label > 0
    return {
        "auc_postclick": safe_auc(cvr_label[m], cvr_pred[m]),
        "auc_exposure": safe_auc(cvr_label, cvr_pred),
        "n_rows": int(cvr_label.size),
        "n_clicked": int(m.sum()),
        "n_purchase_clicked": int(cvr_label[m].sum()),
    }


class ProspectiveTracker(MTLTrainer):
    """Stage A + ordered scoring. Instantiate exactly like `MTLTrainer`."""

    # ---------------------------------------------------------------- helpers

    def _param_checksum(self):
        with torch.no_grad():
            return sum(float(p.detach().double().sum().item()) for p in self.model.parameters())

    # ---------------------------------------------------------------- stage A

    def pretrain_base(self, base_loader, n_epochs, eval_loader=None):
        """Stage A: fixed `n_epochs` over the base period. No early stop, no best-weight restore.

        D1 pinned E as a hyperparameter precisely so the contemporaneous holdout (slices 26-31)
        stays a clean control instead of becoming a selection set: we *measure* convergence on it
        and never select on it.  `MTLTrainer.fit` is deliberately not used -- it early-stops and
        reloads best weights.
        """
        records = []
        for epoch in range(int(n_epochs)):
            losses = self.train_one_epoch(base_loader)
            rec = {"epoch": epoch}
            rec.update({f"train_task{i}_loss": float(v) for i, v in enumerate(np.atleast_1d(losses))})
            if eval_loader is not None:
                rec.update({f"holdout_{k}": v for k, v in self.evaluate_prospective(eval_loader).items()})
            records.append(rec)
            print(f"[stage A] epoch {epoch}: " + ", ".join(f"{k}={v}" for k, v in rec.items() if k != "epoch"),
                  flush=True)
        return records

    def save_checkpoint(self, path, extra=None):
        payload = {"model_state_dict": self.model.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[stage A] checkpoint -> {path}", flush=True)

    def load_checkpoint(self, path, map_location=None):
        payload = torch.load(path, map_location=map_location or self.device)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        self.model.load_state_dict(state)
        self.model.to(self.device)
        return payload

    # ------------------------------------------------------- ordered scoring

    @torch.no_grad()
    def score_ordered(self, loader, n_rows):
        """Score `n_rows` rows and return arrays indexed by ORIGINAL ROW NUMBER.

        Requires batches to carry `CAB_SAMPLE_ID_KEY`; results are scattered by that id, so the
        output is row-ordered even if the loader ever reorders.  Raises if any row is unscored or
        if the pass moved a single parameter.
        """
        checksum_before = self._param_checksum()
        was_training = self.model.training
        self.model.eval()

        p_hat = np.zeros(n_rows, dtype=np.float32)
        cvr_pred = np.zeros(n_rows, dtype=np.float32)
        click = np.zeros(n_rows, dtype=np.int8)
        purchase = np.zeros(n_rows, dtype=np.int8)
        seen = np.zeros(n_rows, dtype=bool)

        for raw_x_dict, ys in tqdm.tqdm(loader, desc="score", smoothing=0, mininterval=1.0):
            model_x, sample_ids = split_batch_inputs(raw_x_dict)
            if sample_ids is None:
                raise ValueError(
                    f"score_ordered requires batches carrying '{CAB_SAMPLE_ID_KEY}'. Without ids the "
                    "dump order depends on the loader and the time axis is silently lost.")
            model_x = {k: v.to(self.device) for k, v in model_x.items()}
            y_preds, _, _ = self.model(model_x)
            idx = sample_ids.detach().cpu().numpy().astype(np.int64)
            p_hat[idx] = y_preds[:, 1].detach().float().cpu().numpy()
            cvr_pred[idx] = y_preds[:, 0].detach().float().cpu().numpy()
            click[idx] = ys[:, 1].detach().cpu().numpy().astype(np.int8)
            purchase[idx] = ys[:, 0].detach().cpu().numpy().astype(np.int8)
            seen[idx] = True

        if was_training:
            self.model.train()
        n_missing = int((~seen).sum())
        if n_missing:
            raise RuntimeError(f"score_ordered left {n_missing:,} of {n_rows:,} rows unscored")
        checksum_after = self._param_checksum()
        if checksum_before != checksum_after:
            raise RuntimeError("scoring pass mutated model parameters (it must be a pure forward pass)")
        return {"p_hat": p_hat, "cvr_pred": cvr_pred, "click": click, "purchase": purchase}

    # --------------------------------------------------------------- stage B

    def train_slice(self, loader):
        """One pass over one slice with the IPS loss and a PRECOMPUTED per-row denominator.

        Deliberately not `train_one_epoch`: that path rebuilds the denominator from the live
        forward pass (`mtl_trainer.py:352`), which D2 replaced with the logged score. Everything
        this protocol does not use -- APL, meta-learning, adaptive weighting, tilt -- is absent
        rather than disabled, so there is no branch left that could silently re-enable it.
        """
        self.model.train()
        total, n_batch = 0.0, 0
        for raw_x_dict, ys in tqdm.tqdm(loader, desc="train", smoothing=0, mininterval=1.0):
            p_corr = raw_x_dict.pop(PCORR_KEY, None)
            if p_corr is None:
                raise ValueError(f"train_slice requires '{PCORR_KEY}' in every batch")
            model_x, _ = split_batch_inputs(raw_x_dict)
            model_x = {k: v.to(self.device) for k, v in model_x.items()}
            ys = ys.to(self.device)
            p_corr = p_corr.to(self.device).float()

            self.optimizer.zero_grad()
            y_preds, _, _ = self.model(model_x)
            loss = ips_loss(ys, y_preds, propensity_override=p_corr)
            loss = loss + self.reg_loss_fn(self.model)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            total += float(loss.item())
            n_batch += 1
        return total / max(n_batch, 1)

    @staticmethod
    def anchor_from_buffer(buffer, slices, n_bins):
        """(edges, anchors, std_logit, counts) from the buffered (score, click) of `slices`."""
        s = np.concatenate([buffer[i][0] for i in slices])
        c = np.concatenate([buffer[i][1] for i in slices]).astype(np.float32)
        _, _, edges, counts, _, _, anchors, _, std_logit = compute_binned_calibration(s, c, n_bins)
        return edges, anchors, std_logit, counts

    @staticmethod
    def mechanism_metrics(p_corr, click, purchase, n_bins, edges=None, bin_ids=None,
                          anchors=None, scores=None):
        """ESS / inverse-weight tail / Gamma_b, on the clicked rows (spec 3.3)."""
        m = click > 0
        w = 1.0 / np.clip(p_corr[m], 1e-6, 1.0)
        out = {
            "ess": float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12)),
            "ess_over_n": float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12) / max(w.size, 1)),
            "w_p99": float(np.percentile(w, 99)) if w.size else float("nan"),
            "w_mean": float(w.mean()) if w.size else float("nan"),
        }
        if anchors is not None and scores is not None and bin_ids is not None:
            # Gamma_b := exp|logit mean_score_b - logit anchor_b| -- the two-pillar seam (ticket 17)
            gam = []
            for b in range(n_bins):
                sel = bin_ids == b
                if not sel.any():
                    continue
                gam.append(abs(_logit(scores[sel].mean()) - _logit(np.asarray(anchors)[b])))
            if gam:
                out["gamma_median"] = float(np.exp(np.median(gam)))
                out["gamma_max"] = float(np.exp(np.max(gam)))
        return out

    # ------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate_prospective(self, loader):
        """Both CVR AUC conventions on `loader`.

        `MTLTrainer.evaluate` cannot be used for the primary metric: it has no click mask and
        scores task 0 over the whole exposure space (`mtl_trainer.py` evaluate), which is the
        non-standard convention the earlier submissions reported (spec section 0.4).
        """
        was_training = self.model.training
        self.model.eval()
        cvr_pred, cvr_label, click_label = [], [], []
        for raw_x_dict, ys in tqdm.tqdm(loader, desc="eval", smoothing=0, mininterval=1.0):
            model_x, _ = split_batch_inputs(raw_x_dict)
            model_x = {k: v.to(self.device) for k, v in model_x.items()}
            y_preds, _, _ = self.model(model_x)
            cvr_pred.append(y_preds[:, 0].detach().float().cpu().numpy())
            cvr_label.append(ys[:, 0].detach().cpu().numpy())
            click_label.append(ys[:, 1].detach().cpu().numpy())
        if was_training:
            self.model.train()
        return cvr_metrics(np.concatenate(cvr_pred), np.concatenate(cvr_label), np.concatenate(click_label))
