import os

import numpy as np
import torch
import torch.nn as nn
import tqdm

from ..basic.callback import EarlyStopper
from ..basic.loss_func import RegularizationLoss
from ..models.multi_task import ESMM
from ..utils.data import get_loss_func, get_metric_func
from .cab_utils import (
    BASELINE_CHOICES,
    BASELINES_NEEDING_CALIBRATION,
    apply_isotonic,
    apply_platt,
    apply_snips,
    apply_weight_clip,
    compute_binned_calibration,
    fit_isotonic_thresholds,
    fit_platt_scaling,
    interpolate_propensity,
    split_batch_inputs,
    weight_clip_threshold,
)
from .loss import *
from ..utils.mtl import MetaBalance, gradnorm, shared_task_layers


class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`.
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after last time validation auc improved (default = 2).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=2,
        device="cpu",
        gpus=None,
        model_path="./",
        model_logger=None,
        loss_type=None,
        apl_alpha=None,
        apl_method=None,
        apl_lambda=None,
        mcdp_n=None,
        propensity_scale=None,
        propensity_shift=None,
        propensity_noise=None,
        apl_learnable=False,
        meta_lr=1e-3,
        meta_update_interval=10,
        meta_inner_lr=None,
        meta_clip_norm=1.0,
        cab_enable=False,
        cab_bins=20,
        cab_beta_init=0.1,
        cab_eps=1e-6,
        cab_beta_dev_scale=0.0,
        cab_beta_dev_scale_limit=1.0,
        cab_tilt_enable=False,
        cab_tilt_lambda=None,
        propensity_baseline=None,
        baseline_bins=None,
        baseline_clip_quantile=0.99,
        baseline_clip_wmax=None,
        baseline_fit_max_rows=5000000,
        baseline_fit_seed=0,
    ):
        self.model = model
        self.loss_type = loss_type
        self.apl_learnable = apl_learnable
        self.meta_lr = meta_lr
        self.meta_update_interval = meta_update_interval
        self.meta_inner_lr = meta_inner_lr
        self.meta_clip_norm = meta_clip_norm
        self.cab_enable = cab_enable
        self.cab_bins = int(cab_bins)
        self.cab_eps = cab_eps
        self.cab_anchor_type = "anchor_shrinkage"
        self.cab_beta_mode = "anchor_absdev"
        self.cab_beta_dev_scale = float(cab_beta_dev_scale)
        self.cab_beta_dev_scale_limit = float(cab_beta_dev_scale_limit)
        self.cab_beta_feature_clip = 2.0
        self.cab_tilt_enable = bool(cab_enable and cab_tilt_enable)
        self.cab_tilt_lambda = float(cab_tilt_lambda) if self.cab_tilt_enable and cab_tilt_lambda is not None else None
        self.cab_sample_bin_ids = None
        self.cab_bin_residuals = None
        self.cab_bin_counts = None
        self.cab_bin_mean_propensity = None
        self.cab_bin_mean_click = None
        self.cab_bin_anchor = None
        self.cab_bin_edges = None
        self.cab_bin_mean_logit = None
        self.cab_bin_std_logit = None
        self.cab_ready = False
        self.global_step = 0
        self.best_cab_beta = None
        self.best_cab_beta_dev_scale = None
        self.best_cab_epoch = None
        self.best_cab_score = None

        # --- calibration / variance-control baselines (issue 22) -----------------
        # A baseline replaces the IPW denominator through the same override hook as CAPC;
        # the two are mutually exclusive so the CAB bin caches can be reused by ``hb``.
        self.propensity_baseline = str(propensity_baseline) if propensity_baseline else None
        if self.propensity_baseline is not None and self.propensity_baseline not in BASELINE_CHOICES:
            raise ValueError(f"propensity_baseline must be one of {BASELINE_CHOICES}, got {self.propensity_baseline}")
        if self.propensity_baseline is not None and self.cab_enable:
            raise ValueError("propensity_baseline and cab_enable are mutually exclusive arms")
        if self.propensity_baseline is not None and self.loss_type not in ["ips", "doubly_robust", "escm2_ips", "escm2_dr"]:
            raise ValueError("propensity_baseline requires an IPW-based loss_type")
        self.baseline_needs_calibration = self.propensity_baseline in BASELINES_NEEDING_CALIBRATION
        self.baseline_bins = int(baseline_bins) if baseline_bins else self.cab_bins
        if self.propensity_baseline == "hb":
            # ``hb`` reuses the CAB binning machinery, so the bin count must be the one it wants.
            self.cab_bins = self.baseline_bins
        self.baseline_clip_quantile = float(baseline_clip_quantile)
        self.baseline_clip_wmax = float(baseline_clip_wmax) if baseline_clip_wmax is not None else None
        self.baseline_fit_max_rows = int(baseline_fit_max_rows)
        self.baseline_fit_seed = int(baseline_fit_seed)
        self.baseline_platt_a = None
        self.baseline_platt_c = None
        self.baseline_iso_x = None
        self.baseline_iso_y = None
        self.baseline_clip_wmax_fitted = None
        self.baseline_ready = False
        # Anything that triggers the per-epoch calibration pass.
        self.calibration_pass_enable = bool(self.cab_enable or self.baseline_needs_calibration)

        if self.cab_enable and apl_method is not None:
            raise ValueError("CAPC currently does not support combining with APL methods")
        if self.cab_enable and self.apl_learnable:
            raise ValueError("CAPC currently does not support combining with APL meta-learning")
        if self.cab_enable and self.loss_type not in ["ips", "doubly_robust", "escm2_ips", "escm2_dr"]:
            raise ValueError("CAB requires an IPW-based loss_type")
        if self.cab_enable and self.cab_bins <= 0:
            raise ValueError("cab_bins must be positive")
        if self.cab_enable and self.cab_beta_dev_scale < 0:
            raise ValueError("cab_beta_dev_scale must be non-negative")
        if self.cab_enable and self.cab_beta_dev_scale > self.cab_beta_dev_scale_limit:
            raise ValueError("cab_beta_dev_scale must be <= cab_beta_dev_scale_limit")
        if self.cab_enable and self.cab_beta_dev_scale_limit <= 0:
            raise ValueError("cab_beta_dev_scale_limit must be positive")
        if cab_tilt_enable and not cab_enable:
            raise ValueError("CAB-Tilt requires cab_enable=True")
        if self.cab_tilt_enable and self.cab_tilt_lambda is None:
            raise ValueError("cab_tilt_lambda must be provided when CAB-Tilt is enabled")

        if self.cab_enable:
            beta_fixed = float(np.clip(cab_beta_init, 0.0, 1.0))
            self.cab_beta_fixed = torch.full((self.cab_bins,), beta_fixed, device=device, dtype=torch.float32)
            fixed_dev_scale = float(np.clip(self.cab_beta_dev_scale, 0.0, self.cab_beta_dev_scale_limit))
            self.cab_beta_dev_scale_fixed = torch.full((self.cab_bins,), fixed_dev_scale, device=device, dtype=torch.float32)
        else:
            self.cab_beta_fixed = None
            self.cab_beta_dev_scale_fixed = None
        
        # Initialize APL parameters as learnable if requested
        if apl_learnable:
            # Parameter range constraints
            self.apl_clamp_min = -5.0
            self.apl_clamp_max = 15.0

            if apl_alpha is not None:
                self.apl_alpha = nn.Parameter(torch.tensor(float(apl_alpha), device=device))
            else:
                self.apl_alpha = None
                
            if apl_lambda is not None:
                self.apl_lambda = nn.Parameter(torch.tensor(float(apl_lambda), device=device))
            else:
                self.apl_lambda = None
                
            # Create meta-optimizer for APL parameters
            meta_params = []
            if self.apl_alpha is not None:
                meta_params.append(self.apl_alpha)
            if self.apl_lambda is not None:
                meta_params.append(self.apl_lambda)
            
            if meta_params:
                self.apl_meta_optimizer = torch.optim.Adam(meta_params, lr=meta_lr)
            else:
                self.apl_learnable = False # No params to learn
                print("Warning: apl_learnable=True but no APL parameters (alpha/lambda) provided.")
        else:
            self.apl_alpha = apl_alpha
            self.apl_lambda = apl_lambda
            
        self.apl_method = apl_method
        self.mcdp_n = mcdp_n
        self.propensity_scale = propensity_scale
        self.propensity_shift = propensity_shift
        self.propensity_noise = propensity_noise
        if gpus is None:
            gpus = []
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None
        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
            elif adaptive_params["method"] == "gradnorm":
                self.adaptive_method = "gradnorm"
                self.alpha = adaptive_params.get("alpha", 0.16)
                share_layers = shared_task_layers(self.model)[0]
                # gradnorm calculate the gradients of each loss on the last
                # fully connected shared layer weight(dimension is 2)
                for i in range(len(share_layers)):
                    if share_layers[-i].ndim == 2:
                        self.last_share_layer = share_layers[-i]
                        break
                self.initial_task_loss = None
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=int(earlystop_patience))

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

        # todo
        if self.loss_type in ["doubly_robust", "escm2_dr", "doubly_robust_apl_heur", "escm2_dr_apl_heur"]:
            self.n_task -= 1

    def train_one_epoch(self, data_loader, val_dataloader=None):
        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)

        apl_val_iter = None
        if self.apl_learnable and val_dataloader is not None:
            apl_val_iter = iter(val_dataloader)

        for iter_i, (raw_x_dict, ys) in enumerate(tk0):
            current_global_step = self.global_step
            model_x, sample_ids = split_batch_inputs(raw_x_dict)
            model_x = {k: v.to(self.device) for k, v in model_x.items()}
            ys = ys.to(self.device)

            if (self.cab_enable or self.propensity_baseline == "hb") and sample_ids is None:
                raise ValueError("CAB / histogram-binning requires batches to carry sample ids")

            # APL meta-learning update
            if self.apl_learnable and apl_val_iter is not None and iter_i % self.meta_update_interval == 0:
                try:
                    val_batch = next(apl_val_iter)
                except StopIteration:
                    apl_val_iter = iter(val_dataloader)
                    val_batch = next(apl_val_iter)

                val_x_raw, val_y = val_batch
                val_x, _ = split_batch_inputs(val_x_raw)
                val_x = {k: v.to(self.device) for k, v in val_x.items()}
                val_y = val_y.to(self.device)

                meta_params = []
                if self.apl_alpha is not None:
                    meta_params.append(self.apl_alpha)
                if self.apl_lambda is not None:
                    meta_params.append(self.apl_lambda)

                if meta_params and hasattr(self, "apl_meta_optimizer"):
                    base_model = self.model.module if hasattr(self.model, "module") else self.model
                    inner_lr = self.meta_inner_lr if self.meta_inner_lr is not None else self._current_lr()

                    if inner_lr is not None:
                        from torch.func import functional_call

                        params_with_names = [(n, p) for n, p in base_model.named_parameters()]
                        param_list = [p for _, p in params_with_names]
                        buffers = {n: b for n, b in base_model.named_buffers()}

                        y_preds_train, ctr_mean, ctr_var = base_model(model_x)

                        current_apl_alpha = torch.nn.functional.softplus(self.apl_alpha) if self.apl_alpha is not None else None
                        current_apl_lambda = torch.nn.functional.softplus(self.apl_lambda) if self.apl_lambda is not None else None

                        if self.loss_type == "ips":
                            train_loss = ips_loss(ys, y_preds_train, current_apl_alpha, ctr_mean, ctr_var, self.apl_method, current_apl_lambda, propensity_scale=self.propensity_scale, propensity_shift=self.propensity_shift, propensity_noise=self.propensity_noise)
                        elif self.loss_type == "doubly_robust":
                            train_loss = doubly_robust_loss(ys, y_preds_train, current_apl_alpha, ctr_mean, ctr_var, self.apl_method, current_apl_lambda)
                        elif self.loss_type == "escm2_ips":
                            train_loss = escm2_ips_loss(ys, y_preds_train, current_apl_alpha, ctr_mean, ctr_var, self.apl_method, current_apl_lambda)
                        elif self.loss_type == "escm2_dr":
                            train_loss = escm2_dr_loss(ys, y_preds_train, current_apl_alpha, ctr_mean, ctr_var, self.apl_method, current_apl_lambda)
                        else:
                            train_loss = sum([self.loss_fns[i](y_preds_train[:, i], ys[:, i].float()) for i in range(self.n_task)]) / self.n_task
                        train_loss = train_loss + self.reg_loss_fn(base_model)

                        train_grads = torch.autograd.grad(train_loss, param_list, create_graph=True, allow_unused=True)
                        updated_params = {}
                        for (name, p), g in zip(params_with_names, train_grads):
                            updated_params[name] = p if g is None else (p - inner_lr * g)

                        was_training = base_model.training
                        if self.apl_method in ("mc_dropout", "mc_dropout_kl"):
                            base_model.train()
                        else:
                            base_model.eval()
                        y_preds_val, _, _ = functional_call(base_model, {**buffers, **updated_params}, (val_x,))
                        val_loss = self.loss_fns[0](y_preds_val[:, 0], val_y[:, 0].float())
                        if was_training:
                            base_model.train()
                        else:
                            base_model.eval()

                        hyper_grads = torch.autograd.grad(val_loss, meta_params, allow_unused=True)
                        self.apl_meta_optimizer.zero_grad()
                        for param, h_grad in zip(meta_params, hyper_grads):
                            if h_grad is None:
                                continue
                            if torch.isnan(h_grad).any() or torch.isinf(h_grad).any():
                                continue
                            param.grad = h_grad.detach()
                        if any((p.grad is not None) for p in meta_params):
                            if self.meta_clip_norm is not None and self.meta_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(meta_params, max_norm=self.meta_clip_norm)
                            self.apl_meta_optimizer.step()
                    with torch.no_grad():
                        if self.apl_alpha is not None:
                            self.apl_alpha.data.clamp_(min=self.apl_clamp_min, max=self.apl_clamp_max)
                        if self.apl_lambda is not None:
                            self.apl_lambda.data.clamp_(min=self.apl_clamp_min, max=self.apl_clamp_max)

            self.optimizer.zero_grad()

            y_preds, ctr_mean, ctr_var = self.model(model_x)
            loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]

            if self.apl_learnable:
                current_apl_alpha = torch.nn.functional.softplus(self.apl_alpha) if self.apl_alpha is not None else None
                current_apl_lambda = torch.nn.functional.softplus(self.apl_lambda) if self.apl_lambda is not None else None
            else:
                current_apl_alpha = self.apl_alpha
                current_apl_lambda = self.apl_lambda

            p_corr = None
            if self.cab_enable:
                p_corr = self._build_cab_corrected_propensity(y_preds[:, 1], sample_ids, detach_beta=False)
            elif self.propensity_baseline is not None:
                p_corr = self._build_baseline_propensity(y_preds[:, 1], sample_ids, ys[:, 1])

            if self.loss_type:
                if self.loss_type == "ips":
                    loss = ips_loss(
                        ys,
                        y_preds,
                        current_apl_alpha,
                        ctr_mean,
                        ctr_var,
                        self.apl_method,
                        current_apl_lambda,
                        propensity_scale=self.propensity_scale,
                        propensity_shift=self.propensity_shift,
                        propensity_noise=self.propensity_noise,
                        propensity_override=p_corr,
                        tilt_lambda=self.cab_tilt_lambda,
                    )
                elif self.loss_type == "escm2_ips":
                    loss = escm2_ips_loss(
                        ys,
                        y_preds,
                        current_apl_alpha,
                        ctr_mean,
                        ctr_var,
                        self.apl_method,
                        current_apl_lambda,
                        propensity_override=p_corr,
                        tilt_lambda=self.cab_tilt_lambda,
                    )
                elif self.loss_type == "doubly_robust":
                    loss = doubly_robust_loss(
                        ys,
                        y_preds,
                        current_apl_alpha,
                        ctr_mean,
                        ctr_var,
                        self.apl_method,
                        current_apl_lambda,
                        propensity_override=p_corr,
                        tilt_lambda=self.cab_tilt_lambda,
                    )
                elif self.loss_type == "escm2_dr":
                    loss = escm2_dr_loss(
                        ys,
                        y_preds,
                        current_apl_alpha,
                        ctr_mean,
                        ctr_var,
                        self.apl_method,
                        current_apl_lambda,
                        propensity_override=p_corr,
                        tilt_lambda=self.cab_tilt_lambda,
                    )
                else:
                    raise ValueError("loss_type doesn't support")
            else:
                if isinstance(self.model, ESMM):
                    ctr_pred = y_preds[:, 1]
                    ctcvr_pred = y_preds[:, 2]
                    ctr_label = ys[:, 1]
                    cvr_label = ys[:, 0]
                    ctcvr_label = ctr_label * cvr_label

                    ctr_loss = self.loss_fns[1](ctr_pred, ctr_label.float()).sum()
                    ctcvr_loss = self.loss_fns[1](ctcvr_pred, ctcvr_label.float()).sum()
                    loss = (ctr_loss + ctcvr_loss) / ys.shape[0]
                else:
                    if self.adaptive_method is not None:
                        if self.adaptive_method == "uwl":
                            loss = 0
                            for loss_i, w_i in zip(loss_list, self.loss_weight):
                                w_i = torch.clamp(w_i, min=0)
                                loss += 2 * loss_i * torch.exp(-w_i) + w_i
                    else:
                        loss = sum(loss_list) / self.n_task

            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss
            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            elif self.adaptive_method == "gradnorm":
                self.optimizer.zero_grad()
                if self.initial_task_loss is None:
                    self.initial_task_loss = [l.item() for l in loss_list]
                gradnorm(loss_list, self.loss_weight, self.last_share_layer, self.initial_task_loss, self.alpha)
                self.optimizer.step()
                loss_weight_sum = sum([w.item() for w in self.loss_weight])
                normalize_coeff = len(self.loss_weight) / loss_weight_sum
                for w in self.loss_weight:
                    w.data = w.data * normalize_coeff
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += np.array([l.item() for l in loss_list])
            self.global_step += 1

        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        loss_list = [total_loss[i] / (iter_i + 1) for i in range(self.n_task)]
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])

        return loss_list

    def fit(self, train_dataloader, val_dataloader, mode='base', seed=0, save_model=False, cab_calibration_loader=None):
        total_log = []
        self.global_step = 0
        self.best_cab_beta = None
        self.best_cab_beta_dev_scale = None
        self.best_cab_epoch = None
        self.best_cab_score = None

        # Log hyperparameters once
        for logger in self._iter_loggers():
            hp = {
                'n_epoch': self.n_epoch,
                'learning_rate': self._current_lr(),
                'adaptive_method': self.adaptive_method,
                'loss_type': self.loss_type,
                'apl_method': self.apl_method,
                'apl_learnable': self.apl_learnable,
                'meta_lr': self.meta_lr,
                'meta_update_interval': self.meta_update_interval,
                'meta_inner_lr': self.meta_inner_lr,
                'meta_clip_norm': self.meta_clip_norm,
                'cab_enable': self.cab_enable,
                'cab_bins': self.cab_bins,
                'cab_anchor_type': self.cab_anchor_type,
                'cab_beta_mode': self.cab_beta_mode,
                'cab_beta_dev_scale': self.cab_beta_dev_scale,
                'cab_beta_dev_scale_limit': self.cab_beta_dev_scale_limit,
                'cab_tilt_enable': self.cab_tilt_enable,
                'cab_tilt_lambda': self.cab_tilt_lambda,
                'propensity_baseline': self.propensity_baseline or 'none',
            }
            if self.propensity_baseline is not None:
                hp['baseline_bins'] = self.baseline_bins
                hp['baseline_clip_quantile'] = self.baseline_clip_quantile
                hp['baseline_clip_wmax'] = self.baseline_clip_wmax if self.baseline_clip_wmax is not None else -1.0
            if self.apl_alpha is not None:
                hp['apl_alpha_raw_init'] = float(self.apl_alpha.item()) if hasattr(self.apl_alpha, "item") else float(self.apl_alpha)
                hp['apl_alpha_init'] = float(torch.nn.functional.softplus(self.apl_alpha).item()) if self.apl_learnable else float(self.apl_alpha)
            if self.apl_lambda is not None:
                hp['apl_lambda_raw_init'] = float(self.apl_lambda.item()) if hasattr(self.apl_lambda, "item") else float(self.apl_lambda)
                hp['apl_lambda_init'] = float(torch.nn.functional.softplus(self.apl_lambda).item()) if self.apl_learnable else float(self.apl_lambda)
            if self.cab_enable:
                beta_init = self._current_cab_beta(detach=True)
                hp['cab_beta_init_mean'] = float(beta_init.mean().item())
                hp['cab_beta_init_min'] = float(beta_init.min().item())
                hp['cab_beta_init_max'] = float(beta_init.max().item())
                cab_dev_scale = self._current_cab_beta_dev_scale(detach=True)
                if cab_dev_scale is not None:
                    hp['cab_beta_dev_scale_current_mean'] = float(cab_dev_scale.mean().item())
                    hp['cab_beta_dev_scale_current_min'] = float(cab_dev_scale.min().item())
                    hp['cab_beta_dev_scale_current_max'] = float(cab_dev_scale.max().item())
            logger.log_hyperparams(hp)

        for epoch_i in range(self.n_epoch):
            if self.calibration_pass_enable:
                self._refresh_cab_statistics(cab_calibration_loader if cab_calibration_loader is not None else train_dataloader)

            _log_per_epoch = self.train_one_epoch(train_dataloader, val_dataloader)

            # Collect metrics
            logs = {f'train/task_{task_id}_loss': loss_val for task_id, loss_val in enumerate(_log_per_epoch)}
            lr_value = self._current_lr()
            if lr_value is not None:
                logs['learning_rate'] = lr_value
            logs['global_step'] = int(self.global_step)

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            scores = self.evaluate(self.model, val_dataloader)
            print('epoch:', epoch_i, 'validation scores: ', scores)

            for task_id, score in enumerate(scores):
                logs[f'val/task_{task_id}_score'] = score
                _log_per_epoch.append(score)
            logs['auc'] = scores[self.earlystop_taskid]

            if self.loss_weight:
                for task_id, weight in enumerate(self.loss_weight):
                    logs[f'loss_weight/task_{task_id}'] = weight.item()
            if self.apl_learnable:
                if self.apl_alpha is not None:
                    logs["apl_alpha"] = torch.nn.functional.softplus(self.apl_alpha).item()
                    logs["apl_alpha_raw"] = self.apl_alpha.item()
                if self.apl_lambda is not None:
                    logs["apl_lambda"] = torch.nn.functional.softplus(self.apl_lambda).item()
                    logs["apl_lambda_raw"] = self.apl_lambda.item()
            if self.cab_enable:
                cab_beta = self._current_cab_beta(detach=True)
                logs["cab_beta_mean"] = float(cab_beta.mean().item())
                logs["cab_beta_min"] = float(cab_beta.min().item())
                logs["cab_beta_max"] = float(cab_beta.max().item())
                cab_dev_scale = self._current_cab_beta_dev_scale(detach=True)
                if cab_dev_scale is not None:
                    logs["cab_beta_dev_scale_mean"] = float(cab_dev_scale.mean().item())
                    logs["cab_beta_dev_scale_min"] = float(cab_dev_scale.min().item())
                    logs["cab_beta_dev_scale_max"] = float(cab_dev_scale.max().item())
                if self.cab_bin_residuals is not None:
                    logs["cab_residual_mean_abs"] = float(self.cab_bin_residuals.abs().mean().item())
                    logs["cab_residual_max_abs"] = float(self.cab_bin_residuals.abs().max().item())
                if self.cab_bin_anchor is not None:
                    logs["cab_anchor_mean"] = float(self.cab_bin_anchor.mean().item())
                    logs["cab_anchor_min"] = float(self.cab_bin_anchor.min().item())
                    logs["cab_anchor_max"] = float(self.cab_bin_anchor.max().item())
                if self.cab_bin_counts is not None:
                    logs["cab_bin_count_min"] = float(self.cab_bin_counts.min().item())
                    logs["cab_bin_count_max"] = float(self.cab_bin_counts.max().item())
                if self.cab_bin_mean_propensity is not None and self.cab_bin_anchor is not None and self.cab_bin_residuals is not None:
                    for bin_idx in range(self.cab_bins):
                        logs[f"cab_bin_{bin_idx:02d}_beta"] = float(cab_beta[bin_idx].item())
                        if cab_dev_scale is not None:
                            logs[f"cab_bin_{bin_idx:02d}_beta_dev_scale"] = float(cab_dev_scale[bin_idx].item())
                        logs[f"cab_bin_{bin_idx:02d}_residual"] = float(self.cab_bin_residuals[bin_idx].item())
                        logs[f"cab_bin_{bin_idx:02d}_count"] = float(self.cab_bin_counts[bin_idx].item())
                        logs[f"cab_bin_{bin_idx:02d}_mean_propensity"] = float(self.cab_bin_mean_propensity[bin_idx].item())
                        logs[f"cab_bin_{bin_idx:02d}_anchor"] = float(self.cab_bin_anchor[bin_idx].item())

            total_log.append(_log_per_epoch)

            # Log metrics once per epoch
            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_i)

            if self.apl_learnable:
                msg = []
                if self.apl_alpha is not None:
                    msg.append(f"apl_alpha={torch.nn.functional.softplus(self.apl_alpha).item():.6f}")
                if self.apl_lambda is not None:
                    msg.append(f"apl_lambda={torch.nn.functional.softplus(self.apl_lambda).item():.6f}")
                if msg:
                    print("meta params:", ", ".join(msg), flush=True)
            if self.cab_enable:
                cab_beta = self._current_cab_beta(detach=True)
                msg = [
                    f"cab_beta_mean={cab_beta.mean().item():.6f}",
                    f"cab_beta_min={cab_beta.min().item():.6f}",
                    f"cab_beta_max={cab_beta.max().item():.6f}",
                    f"cab_anchor_type={self.cab_anchor_type}",
                    f"cab_beta_mode={self.cab_beta_mode}",
                ]
                cab_dev_scale = self._current_cab_beta_dev_scale(detach=True)
                if cab_dev_scale is not None:
                    msg.append(f"cab_beta_dev_scale_mean={cab_dev_scale.mean().item():.6f}")
                if self.cab_bin_residuals is not None:
                    msg.append(f"cab_residual_max_abs={self.cab_bin_residuals.abs().max().item():.6f}")
                if self.cab_bin_anchor is not None:
                    msg.append(f"cab_anchor_mean={self.cab_bin_anchor.mean().item():.6f}")
                if self.cab_tilt_enable:
                    msg.append(f"cab_tilt_lambda={self.cab_tilt_lambda:.6f}")
                print("cab params:", ", ".join(msg), flush=True)

            if self.cab_enable and scores[self.earlystop_taskid] > self.early_stopper.best_auc:
                self.best_cab_beta = self._current_cab_beta(detach=True).detach().cpu().clone()
                self.best_cab_epoch = epoch_i
                self.best_cab_score = float(scores[self.earlystop_taskid])
                cab_dev_scale = self._current_cab_beta_dev_scale(detach=True)
                if cab_dev_scale is not None:
                    self.best_cab_beta_dev_scale = cab_dev_scale.detach().cpu().clone()
                else:
                    self.best_cab_beta_dev_scale = None

            if self.early_stopper.stop_training(scores[self.earlystop_taskid], self.model.state_dict()):
                print('validation best auc of main task %d: %.6f' % (self.earlystop_taskid, self.early_stopper.best_auc))
                break

        if self.cab_enable and self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
            self._restore_best_cab_state()
            print(
                "restored best cab checkpoint:",
                f"epoch={self.best_cab_epoch}",
                f"auc={self.best_cab_score:.6f}" if self.best_cab_score is not None else "auc=unknown",
                flush=True,
            )

        if save_model:
            model_path = os.path.join(self.model_path, "model_{}_{}.pth".format(mode, seed))
            torch.save(self.model.state_dict(), model_path)  # save best auc model
            if self.cab_enable:
                cab_state_path = os.path.join(self.model_path, "model_{}_{}_cab_state.pth".format(mode, seed))
                torch.save(
                    {
                        "cab_enable": self.cab_enable,
                        "cab_bins": self.cab_bins,
                        "cab_tilt_enable": self.cab_tilt_enable,
                        "cab_tilt_lambda": self.cab_tilt_lambda,
                        "cab_beta_mode": self.cab_beta_mode,
                        "cab_beta_dev_scale": self.cab_beta_dev_scale,
                        "cab_beta_dev_scale_limit": self.cab_beta_dev_scale_limit,
                        "best_cab_epoch": self.best_cab_epoch,
                        "best_cab_score": self.best_cab_score,
                        "best_cab_beta": self.best_cab_beta,
                        "best_cab_beta_dev_scale": self.best_cab_beta_dev_scale,
                    },
                    cab_state_path,
                )

        if self.cab_enable and self.best_cab_beta is not None:
            beta_values = ", ".join(f"{v:.6f}" for v in self.best_cab_beta.tolist())
            print(
                "best cab beta:",
                f"epoch={self.best_cab_epoch}",
                f"auc={self.best_cab_score:.6f}",
                f"mean={self.best_cab_beta.mean().item():.6f}",
                f"min={self.best_cab_beta.min().item():.6f}",
                f"max={self.best_cab_beta.max().item():.6f}",
                f"values=[{beta_values}]",
                flush=True,
            )
            if self.best_cab_beta_dev_scale is not None:
                dev_scale_values = ", ".join(f"{v:.6f}" for v in self.best_cab_beta_dev_scale.tolist())
                print(
                    "best cab beta deviation scale:",
                    f"mean={self.best_cab_beta_dev_scale.mean().item():.6f}",
                    f"min={self.best_cab_beta_dev_scale.min().item():.6f}",
                    f"max={self.best_cab_beta_dev_scale.max().item():.6f}",
                    f"values=[{dev_scale_values}]",
                    flush=True,
                )
        for logger in self._iter_loggers():
            logger.finish()

        return total_log

    def _iter_loggers(self):
        """Return logger instances as a list.

        Returns
        -------
        list
            Active logger instances. Empty when ``model_logger`` is ``None``.
        """
        if self.model_logger is None:
            return []
        if isinstance(self.model_logger, (list, tuple)):
            return list(self.model_logger)
        return [self.model_logger]

    def _current_lr(self):
        """Fetch current learning rate regardless of adaptive method."""
        if self.adaptive_method == "metabalance":
            return self.share_optimizer.param_groups[0]['lr'] if hasattr(self, 'share_optimizer') else None
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return None

    def _current_cab_beta(self, detach=False, bin_ids=None):
        if not self.cab_enable:
            return None
        beta = self.cab_beta_fixed
        beta = beta.detach() if detach else beta
        if bin_ids is not None:
            if beta.device != bin_ids.device:
                beta = beta.to(bin_ids.device)
            beta = beta[bin_ids]
        return beta

    def _current_cab_beta_dev_scale(self, detach=False, bin_ids=None):
        if not self.cab_enable:
            return None
        dev_scale = self.cab_beta_dev_scale_fixed
        dev_scale = dev_scale.detach() if detach else dev_scale
        if bin_ids is not None:
            if dev_scale.device != bin_ids.device:
                dev_scale = dev_scale.to(bin_ids.device)
            dev_scale = dev_scale[bin_ids]
        return dev_scale

    def _restore_best_cab_state(self):
        if not self.cab_enable or self.best_cab_beta is None:
            return
        if self.cab_beta_fixed is not None:
            self.cab_beta_fixed = self.best_cab_beta.to(self.cab_beta_fixed.device).clone()
        if self.cab_beta_dev_scale_fixed is not None and self.best_cab_beta_dev_scale is not None:
            self.cab_beta_dev_scale_fixed = self.best_cab_beta_dev_scale.to(self.cab_beta_dev_scale_fixed.device).clone()

    def _lookup_cab_bin_ids(self, sample_ids, device=None):
        if sample_ids is None:
            raise ValueError("CAB requires sample ids in each batch")
        if self.cab_sample_bin_ids is None:
            raise RuntimeError("CAB bin assignments are not ready; call _refresh_cab_statistics first")
        batch_ids = sample_ids.detach().cpu().long()
        batch_bins = self.cab_sample_bin_ids[batch_ids].long()
        if device is not None:
            batch_bins = batch_bins.to(device)
        return batch_bins

    def _lookup_cab_residual(self, sample_ids, device):
        if self.cab_bin_residuals is None:
            raise RuntimeError("CAB statistics are not ready; call _refresh_cab_statistics first")
        batch_bins = self._lookup_cab_bin_ids(sample_ids, device=None)
        return self.cab_bin_residuals[batch_bins].to(device)

    def _lookup_cab_bin_stat(self, stat_tensor, bin_ids, device):
        if stat_tensor is None:
            raise RuntimeError("Requested CAB bin statistic is not ready")
        values = stat_tensor[bin_ids.detach().cpu()]
        if device is not None:
            values = values.to(device)
        return values

    def _compute_cab_beta_feature(self, propensity, bin_ids):
        if self.cab_bin_std_logit is None or self.cab_bin_anchor is None:
            raise RuntimeError("CAB adaptive beta statistics are not ready; call _refresh_cab_statistics first")
        std_logit = self._lookup_cab_bin_stat(self.cab_bin_std_logit, bin_ids, propensity.device)
        anchor = self._lookup_cab_bin_stat(self.cab_bin_anchor, bin_ids, propensity.device)
        anchor = torch.clamp(anchor, min=self.cab_eps, max=1.0 - self.cab_eps)
        propensity_logit = torch.logit(torch.clamp(propensity, min=self.cab_eps, max=1.0 - self.cab_eps))
        anchor_logit = torch.logit(anchor)
        z = (propensity_logit - anchor_logit) / torch.clamp(std_logit, min=1e-3)
        return torch.clamp(z, min=-self.cab_beta_feature_clip, max=self.cab_beta_feature_clip)

    def _current_cab_sample_beta(self, propensity, bin_ids, detach=False):
        base_beta = self._current_cab_beta(detach=detach, bin_ids=bin_ids)
        dev_scale = self._current_cab_beta_dev_scale(detach=detach, bin_ids=bin_ids)
        beta_feature = self._compute_cab_beta_feature(propensity, bin_ids)
        beta_raw = torch.logit(torch.clamp(base_beta, min=1e-4, max=1.0 - 1e-4))
        return torch.sigmoid(beta_raw + dev_scale * torch.abs(beta_feature))

    def _build_cab_corrected_propensity(self, ctr_pred, sample_ids, detach_beta=False):
        propensity = apply_propensity_perturbation(
            ctr_pred,
            propensity_scale=self.propensity_scale,
            propensity_shift=self.propensity_shift,
            propensity_noise=self.propensity_noise,
        )
        propensity = torch.clamp(propensity, min=self.cab_eps, max=1.0 - self.cab_eps).detach()
        bin_ids = self._lookup_cab_bin_ids(sample_ids, device=propensity.device)
        anchor = self._lookup_cab_bin_stat(self.cab_bin_anchor, bin_ids, propensity.device)
        anchor = torch.clamp(anchor, min=self.cab_eps, max=1.0 - self.cab_eps)
        beta = self._current_cab_sample_beta(propensity, bin_ids, detach=detach_beta)
        corrected = interpolate_propensity(propensity, anchor, beta)
        return torch.clamp(corrected, min=self.cab_eps, max=1.0 - self.cab_eps)

    def _build_baseline_propensity(self, ctr_pred, sample_ids, click_label):
        """Replace the IPW denominator with a calibration / variance-control baseline.

        Returns a detached tensor: ``loss.py:_get_ips_weight_base`` does NOT detach when an
        override is supplied, so every arm has to detach at the source to keep the same
        gradient semantics as vanilla IPS (see the assertion in
        ``tests/test_propensity_baselines.py``).
        """
        propensity = apply_propensity_perturbation(
            ctr_pred,
            propensity_scale=self.propensity_scale,
            propensity_shift=self.propensity_shift,
            propensity_noise=self.propensity_noise,
        )
        propensity = torch.clamp(propensity, min=self.cab_eps, max=1.0 - self.cab_eps).detach()

        name = self.propensity_baseline
        if name == "snips":
            out = apply_snips(propensity, click_label.detach().float(), eps=self.cab_eps)
        elif name == "clip":
            w_max = self.baseline_clip_wmax if self.baseline_clip_wmax is not None else self.baseline_clip_wmax_fitted
            if w_max is None:
                raise RuntimeError("weight-clipping threshold is not ready; call _refresh_cab_statistics first")
            out = apply_weight_clip(propensity, w_max, eps=self.cab_eps)
        elif name == "hb":
            bin_ids = self._lookup_cab_bin_ids(sample_ids, device=propensity.device)
            anchor = self._lookup_cab_bin_stat(self.cab_bin_anchor, bin_ids, propensity.device)
            out = torch.clamp(anchor, min=self.cab_eps, max=1.0 - self.cab_eps)
        elif name == "platt":
            if self.baseline_platt_a is None:
                raise RuntimeError("Platt parameters are not ready; call _refresh_cab_statistics first")
            out = apply_platt(propensity, self.baseline_platt_a, self.baseline_platt_c, eps=self.cab_eps)
        elif name == "isotonic":
            if self.baseline_iso_x is None:
                raise RuntimeError("Isotonic knots are not ready; call _refresh_cab_statistics first")
            x = self.baseline_iso_x.to(propensity.device)
            y = self.baseline_iso_y.to(propensity.device)
            out = apply_isotonic(propensity, x, y, eps=self.cab_eps)
            # Issue 27 guard: isotonic can map some propensities to ~0 -> inverse weight ~1/eps
            # (1e6), which NaNs the run. When an absolute weight cap is supplied (the same
            # --baseline_clip_wmax knob the clip arm uses), clamp the inverse weight so the
            # collapse trend stays visible in the results without detonating the matrix.
            # Intentionally NOT a quantile cap -- that would mask the collapse (issue 27 (a)).
            if self.baseline_clip_wmax is not None:
                out = apply_weight_clip(out, self.baseline_clip_wmax, eps=self.cab_eps)
        else:
            raise ValueError(f"unknown propensity_baseline {name}")
        return out.detach()

    def _fit_propensity_baseline(self, ctr_scores, click_labels):
        """Fit the baseline off the same calibration pass CAPC uses (train split, per epoch)."""
        name = self.propensity_baseline
        if name == "clip":
            if self.baseline_clip_wmax is None:
                self.baseline_clip_wmax_fitted = weight_clip_threshold(ctr_scores, self.baseline_clip_quantile, eps=self.cab_eps)
                print(f"baseline[clip] w_max(q={self.baseline_clip_quantile})={self.baseline_clip_wmax_fitted:.4f}", flush=True)
            else:
                print(f"baseline[clip] w_max(fixed)={self.baseline_clip_wmax:.4f}", flush=True)
        elif name == "platt":
            self.baseline_platt_a, self.baseline_platt_c = fit_platt_scaling(
                ctr_scores, click_labels, max_rows=self.baseline_fit_max_rows, seed=self.baseline_fit_seed, eps=self.cab_eps)
            print(f"baseline[platt] a={self.baseline_platt_a:.6f} c={self.baseline_platt_c:.6f}", flush=True)
        elif name == "isotonic":
            x, y = fit_isotonic_thresholds(
                ctr_scores, click_labels, max_rows=self.baseline_fit_max_rows, seed=self.baseline_fit_seed)
            self.baseline_iso_x = torch.from_numpy(x)
            self.baseline_iso_y = torch.from_numpy(y)
            print(f"baseline[isotonic] knots={x.shape[0]} range=[{float(y.min()):.6f}, {float(y.max()):.6f}]", flush=True)
        elif name == "hb":
            # A bin with zero clicks gets anchor == cab_eps, i.e. a weight of 1/eps = 1e6, which
            # silently detonates the arm. On real data every bin has clicks (Ali-CCP K=50:
            # anchor_range min 0.0096), so this is a loud guard, not a routine path.
            empty_bins = int((self.cab_bin_counts > 0).logical_and(self.cab_bin_mean_click <= 0).sum().item())
            if empty_bins > 0:
                raise RuntimeError(
                    f"histogram-binning baseline: {empty_bins}/{self.cab_bins} bins have zero clicks, so their "
                    f"anchor is clamped to eps={self.cab_eps} and the inverse weight becomes {1.0 / self.cab_eps:.0f}. "
                    f"Lower --baseline_bins (currently {self.cab_bins}) instead of training on exploding weights.")
            print(f"baseline[hb] K={self.cab_bins} anchor_range="
                  f"[{self.cab_bin_anchor.min().item():.6f}, {self.cab_bin_anchor.max().item():.6f}]", flush=True)
        self.baseline_ready = True

    def _refresh_cab_statistics(self, calibration_loader):
        if not self.calibration_pass_enable:
            return

        dataset_size = len(calibration_loader.dataset)
        ctr_scores = np.zeros(dataset_size, dtype=np.float32)
        click_labels = np.zeros(dataset_size, dtype=np.float32)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            tk0 = tqdm.tqdm(calibration_loader, desc="cab-calib", smoothing=0, mininterval=1.0)
            for raw_x_dict, ys in tk0:
                model_x, sample_ids = split_batch_inputs(raw_x_dict)
                if sample_ids is None:
                    raise ValueError("CAB calibration loader must provide sample ids")
                model_x = {k: v.to(self.device) for k, v in model_x.items()}
                ys = ys.to(self.device)
                y_preds, _, _ = self.model(model_x)
                propensity = apply_propensity_perturbation(
                    y_preds[:, 1],
                    propensity_scale=self.propensity_scale,
                    propensity_shift=self.propensity_shift,
                    propensity_noise=self.propensity_noise,
                )
                propensity = torch.clamp(propensity, min=self.cab_eps, max=1.0 - self.cab_eps)
                sample_ids_np = sample_ids.detach().cpu().long().numpy()
                ctr_scores[sample_ids_np] = propensity.detach().cpu().numpy()
                click_labels[sample_ids_np] = ys[:, 1].detach().cpu().numpy()
        if was_training:
            self.model.train()

        (
            bin_ids,
            residuals,
            bin_edges,
            counts,
            mean_scores,
            mean_clicks,
            anchors,
            mean_logits,
            std_logits,
        ) = compute_binned_calibration(ctr_scores, click_labels, self.cab_bins)
        bin_dtype = torch.uint8 if self.cab_bins <= 256 else torch.int32
        self.cab_sample_bin_ids = torch.from_numpy(bin_ids.astype(np.uint8 if self.cab_bins <= 256 else np.int32))
        self.cab_sample_bin_ids = self.cab_sample_bin_ids.to(dtype=bin_dtype)
        self.cab_bin_residuals = torch.from_numpy(residuals.astype(np.float32))
        self.cab_bin_counts = torch.from_numpy(counts.astype(np.float32))
        self.cab_bin_mean_propensity = torch.from_numpy(mean_scores.astype(np.float32))
        self.cab_bin_mean_click = torch.from_numpy(mean_clicks.astype(np.float32))
        self.cab_bin_anchor = torch.from_numpy(np.clip(anchors, self.cab_eps, 1.0 - self.cab_eps).astype(np.float32))
        self.cab_bin_edges = torch.from_numpy(bin_edges.astype(np.float32))
        self.cab_bin_mean_logit = torch.from_numpy(mean_logits.astype(np.float32))
        self.cab_bin_std_logit = torch.from_numpy(std_logits.astype(np.float32))
        self.cab_ready = True

        print(
            "CAB refresh:",
            f"bins={self.cab_bins}",
            f"residual_range=[{self.cab_bin_residuals.min().item():.6f}, {self.cab_bin_residuals.max().item():.6f}]",
            f"mean_abs={self.cab_bin_residuals.abs().mean().item():.6f}",
            f"anchor_range=[{self.cab_bin_anchor.min().item():.6f}, {self.cab_bin_anchor.max().item():.6f}]",
            flush=True,
        )

        if self.propensity_baseline is not None:
            self._fit_propensity_baseline(ctr_scores, click_labels)

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict, _ = split_batch_inputs(x_dict)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                ys = ys.to(self.device)
                y_preds, _, _ = model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return scores

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict, _ = split_batch_inputs(x_dict)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds, _, _ = model(x_dict)
                predicts.extend(y_preds.tolist())
        return predicts

    def export_onnx(self, output_path, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained multi-task model to ONNX format.

        This method exports multi-task learning models (e.g., MMOE, PLE, ESMM, SharedBottom)
        to ONNX format for deployment. The exported model will have multiple outputs
        corresponding to each task.

        Note:
            The ONNX model will output a tensor of shape [batch_size, n_task] where
            n_task is the number of tasks in the multi-task model.

        Args:
            output_path (str): Path to save the ONNX model file.
            dummy_input (dict, optional): Example input dict {feature_name: tensor}.
                If not provided, dummy inputs will be generated automatically.
            batch_size (int): Batch size for auto-generated dummy input (default: 2).
            seq_length (int): Sequence length for SequenceFeature (default: 10).
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = MTLTrainer(mmoe_model, task_types=["classification", "classification"], ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("mmoe.onnx")

            >>> # Export on specific device
            >>> trainer.export_onnx("mmoe.onnx", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        exporter = ONNXExporter(model, device=export_device)
        return exporter.export(
            output_path=output_path,
            dummy_input=dummy_input,
            batch_size=batch_size,
            seq_length=seq_length,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verbose=verbose,
            onnx_export_kwargs=onnx_export_kwargs,
        )

    def visualization(self, input_data=None, batch_size=2, seq_length=10, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Visualize the model's computation graph.

        This method generates a visual representation of the model architecture,
        showing layer connections, tensor shapes, and nested module structures.
        It automatically extracts feature information from the model.

        Parameters
        ----------
        input_data : dict, optional
            Example input dict {feature_name: tensor}.
            If not provided, dummy inputs will be generated automatically.
        batch_size : int, default=2
            Batch size for auto-generated dummy input.
        seq_length : int, default=10
            Sequence length for SequenceFeature.
        depth : int, default=3
            Visualization depth, higher values show more detail.
            Set to -1 to show all layers.
        show_shapes : bool, default=True
            Whether to display tensor shapes.
        expand_nested : bool, default=True
            Whether to expand nested modules.
        save_path : str, optional
            Path to save the graph image (.pdf, .svg, .png).
            If None, displays in Jupyter or opens system viewer.
        graph_name : str, default="model"
            Name for the graph.
        device : str, optional
            Device for model execution. If None, defaults to 'cpu'.
        dpi : int, default=300
            Resolution in dots per inch for output image.
            Higher values produce sharper images suitable for papers.
        **kwargs : dict
            Additional arguments passed to torchview.draw_graph().

        Returns
        -------
        ComputationGraph
            A torchview ComputationGraph object.

        Raises
        ------
        ImportError
            If torchview or graphviz is not installed.

        Notes
        -----
        Default Display Behavior:
            When `save_path` is None (default):
            - In Jupyter/IPython: automatically displays the graph inline
            - In Python script: opens the graph with system default viewer

        Examples
        --------
        >>> trainer = MTLTrainer(model, task_types=["classification", "classification"])
        >>> trainer.fit(train_dl, val_dl)
        >>>
        >>> # Auto-display in Jupyter (no save_path needed)
        >>> trainer.visualization(depth=4)
        >>>
        >>> # Save to high-DPI PNG for papers
        >>> trainer.visualization(save_path="model.png", dpi=300)
        """
        from ..utils.visualization import TORCHVIEW_AVAILABLE, visualize_model

        if not TORCHVIEW_AVAILABLE:
            raise ImportError(
                "Visualization requires torchview. "
                "Install with: pip install torch-rechub[visualization]\n"
                "Also ensure graphviz is installed on your system:\n"
                "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                "  - macOS: brew install graphviz\n"
                "  - Windows: choco install graphviz"
            )

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        viz_device = device if device is not None else 'cpu'

        return visualize_model(
            model,
            input_data=input_data,
            batch_size=batch_size,
            seq_length=seq_length,
            depth=depth,
            show_shapes=show_shapes,
            expand_nested=expand_nested,
            save_path=save_path,
            graph_name=graph_name,
            device=viz_device,
            dpi=dpi,
            **kwargs
        )
