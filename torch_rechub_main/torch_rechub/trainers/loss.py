import torch
import torch.nn as nn

bce_loss_fn = nn.BCELoss(reduction='none')


def apply_propensity_perturbation(propensity, propensity_scale=None, propensity_shift=None, propensity_noise=None):
    if propensity_scale is not None:
        propensity = propensity ** propensity_scale
    if propensity_shift is not None:
        propensity = propensity * (1.0 + propensity_shift)
    if propensity_noise is not None:
        noise = torch.randn_like(propensity) * propensity_noise
        propensity = propensity * (1.0 + noise)
    return propensity


def _get_ips_weight_base(propensity_clamped, propensity_override=None):
    propensity_base = propensity_clamped if propensity_override is not None else propensity_clamped.detach()
    return 1.0 / propensity_base


def _get_tilted_weighted_loss(ips_weight_base, cvr_loss, tilt_lambda, click_mask, batch_size, eps, detach_ips_weight=True):
    q = apply_apl_kl(
        ips_weight_base,
        cvr_loss,
        tilt_lambda,
        click_mask,
        eps,
        detach_ips_weight=detach_ips_weight,
    )
    return (q * cvr_loss * batch_size).sum()


def _get_tilted_ips(ips_weight_base, cvr_loss, tilt_lambda, click_mask, batch_size, eps, detach_ips_weight=True):
    q = apply_apl_kl(
        ips_weight_base,
        cvr_loss,
        tilt_lambda,
        click_mask,
        eps,
        detach_ips_weight=detach_ips_weight,
    )
    return q * batch_size


def ips_loss(ys, y_preds, apl_alpha=None, ctr_mean=None, ctr_var=None, apl_method=None, apl_lambda=None, eps=1e-6, propensity_scale=None, propensity_shift=None, propensity_noise=None, propensity_override=None, tilt_lambda=None):
    """
    IPS (Inverse Propensity Scoring) loss for CVR estimation.
    
    Args:
        ys: Ground truth labels, shape (batch_size, n_task)
            ys[:, 0] = CVR label (purchase), only valid when clicked
            ys[:, 1] = CTR label (click)
        y_preds: Model predictions (probabilities after sigmoid), shape (batch_size, n_task)
            y_preds[:, 0] = CVR prediction
            y_preds[:, 1] = CTR prediction
        apl_alpha: Alpha for APL weighting (heuristic/mc_dropout methods)
        ctr_mean: Mean of CTR predictions from MC Dropout (optional)
        ctr_var: Variance of CTR predictions from MC Dropout (optional)
        apl_method: APL method: None, 'heuristic', 'mc_dropout', or 'kl'
        apl_lambda: Lambda for KL method
        eps: Small value to prevent division by zero
        propensity_scale: Scale factor for propensity score (synthetic bias injection)
        propensity_shift: Shift value for propensity score (synthetic bias injection)
        propensity_noise: Noise variance for propensity score (synthetic bias injection)
    
    Returns:
        (weighted CVR loss + CTR loss) / batch_size
    """
    click_label = ys[:, 1]
    cvr_label = ys[:, 0]
    
    ctr_pred = y_preds[:, 1]
    cvr_pred = y_preds[:, 0]
    
    ctr_pred = apply_propensity_perturbation(
        ctr_pred,
        propensity_scale=propensity_scale,
        propensity_shift=propensity_shift,
        propensity_noise=propensity_noise,
    )
        
    ctr_pred_clamped = torch.clamp(ctr_pred, min=eps, max=1.0 - eps)
    cvr_pred_clamped = torch.clamp(cvr_pred, min=eps, max=1.0 - eps)
    propensity_clamped = torch.clamp(propensity_override, min=eps, max=1.0 - eps) if propensity_override is not None else ctr_pred_clamped
    
    click_mask = click_label.float()
    batch_size = click_mask.shape[0]
    
    cvr_bce = bce_loss_fn(cvr_pred_clamped, cvr_label.float())
    ctr_loss = bce_loss_fn(ctr_pred_clamped, click_label.float()).sum()
    
    if apl_method == 'kl' and apl_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif apl_method == 'mc_dropout' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None:
        ips_weight = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    elif apl_method == 'heuristic' and apl_alpha is not None:
        ips_weight = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    elif apl_method == 'heuristic_kl' and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif apl_method == 'mc_dropout_kl' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif tilt_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        weighted_cvr_loss = _get_tilted_weighted_loss(
            ips_weight_base,
            cvr_bce,
            tilt_lambda,
            click_mask,
            batch_size,
            eps,
            detach_ips_weight=(propensity_override is None),
        )
    else:
        ips_weight = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    
    total_loss = (weighted_cvr_loss + ctr_loss) / batch_size
    return total_loss



def doubly_robust_loss(ys, y_preds, apl_alpha=None, ctr_mean=None, ctr_var=None, apl_method=None, apl_lambda=None, eps=1e-6, propensity_override=None, tilt_lambda=None):
    """
    Doubly Robust Loss for CVR estimation.
    
    Args:
        ys: Ground truth labels, shape (batch_size, n_task)
            ys[:, 0] = CVR label (purchase), only valid when clicked
            ys[:, 1] = CTR label (click)
        y_preds: Model predictions (probabilities after sigmoid), shape (batch_size, n_task)
            y_preds[:, 0] = CVR prediction
            y_preds[:, 1] = CTR prediction
            y_preds[:, 2] = CVR imp tower: predict CVR loss
        apl_alpha: Alpha for APL weighting (heuristic/mc_dropout methods)
        ctr_mean: Mean of CTR predictions from MC Dropout (optional)
        ctr_var: Variance of CTR predictions from MC Dropout (optional)
        apl_method: APL method: None, 'heuristic', 'mc_dropout', or 'kl'
        apl_lambda: Lambda for KL method
        eps: Small value to prevent division by zero
    
    Returns:
        (weighted CVR loss + CTR loss) / batch_size
    """
    click_label = ys[:, 1]
    cvr_label = ys[:, 0]

    ctr_pred = y_preds[:, 1]
    cvr_pred = y_preds[:, 0]
    cvr_imp_pred = y_preds[:, 2]

    ctr_pred_clamped = torch.clamp(ctr_pred, min=eps, max=1.0 - eps)
    cvr_pred_clamped = torch.clamp(cvr_pred, min=eps, max=1.0 - eps)
    cvr_imp_pred_clamped = torch.clamp(cvr_imp_pred, min=eps, max=1.0 - eps)
    propensity_clamped = torch.clamp(propensity_override, min=eps, max=1.0 - eps) if propensity_override is not None else ctr_pred_clamped

    cvr_loss = bce_loss_fn(cvr_pred_clamped, cvr_label.float())
    ctr_loss = bce_loss_fn(ctr_pred_clamped, click_label.float()).sum()
    
    click_mask = click_label.float()
    batch_size = click_label.shape[0]
    
    if apl_method == 'kl' and apl_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif apl_method == 'mc_dropout' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None:
        ips = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
    elif apl_method == 'heuristic' and apl_alpha is not None:
        ips = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
    elif apl_method == 'heuristic_kl' and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif apl_method == 'mc_dropout_kl' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif tilt_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        ips = _get_tilted_ips(
            ips_weight_base,
            cvr_loss,
            tilt_lambda,
            click_mask,
            batch_size,
            eps,
            detach_ips_weight=(propensity_override is None),
        )
    else:
        ips = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)

    e = cvr_loss - cvr_imp_pred_clamped
    dr_part_1 = cvr_imp_pred_clamped
    dr_part_2 = click_mask * e * ips

    dr_loss = (dr_part_1 + dr_part_2).sum()

    e_detached = cvr_loss.detach() - cvr_imp_pred_clamped
    imp_loss = (ips * (e_detached**2)).sum()

    return (0.5 * dr_loss + 0.5 * imp_loss + ctr_loss) / batch_size




def escm2_ips_loss(ys, y_preds, apl_alpha=None, ctr_mean=None, ctr_var=None, apl_method=None, apl_lambda=None, eps=1e-6, propensity_override=None, tilt_lambda=None):
    """
    IPS (Inverse Propensity Scoring) loss for CVR estimation with ESCM2.
    
    Args:
        ys: Ground truth labels, shape (batch_size, n_task)
            ys[:, 0] = CVR label (purchase), only valid when clicked
            ys[:, 1] = CTR label (click)
        y_preds: Model predictions (probabilities after sigmoid), shape (batch_size, n_task)
            y_preds[:, 0] = CVR prediction
            y_preds[:, 1] = CTR prediction
        apl_alpha: Alpha for APL weighting (heuristic/mc_dropout methods)
        ctr_mean: Mean of CTR predictions from MC Dropout (optional)
        ctr_var: Variance of CTR predictions from MC Dropout (optional)
        apl_method: APL method: None, 'heuristic', 'mc_dropout', or 'kl'
        apl_lambda: Lambda for KL method
        eps: Small value to prevent division by zero
    
    Returns:
        (weighted CVR loss + CTR loss) / batch_size
    """
    click_label = ys[:, 1]
    cvr_label = ys[:, 0]
    ctcvr_label = click_label.float() * cvr_label
    
    ctr_pred = y_preds[:, 1]
    cvr_pred = y_preds[:, 0]
    ctcvr_pred = ctr_pred * cvr_pred
    
    ctr_pred_clamped = torch.clamp(ctr_pred, min=eps, max=1.0 - eps)
    cvr_pred_clamped = torch.clamp(cvr_pred, min=eps, max=1.0 - eps)
    ctcvr_pred_clamped = torch.clamp(ctcvr_pred, min=eps, max=1.0 - eps)
    propensity_clamped = torch.clamp(propensity_override, min=eps, max=1.0 - eps) if propensity_override is not None else ctr_pred_clamped
    
    click_mask = click_label.float()
    batch_size = click_mask.shape[0]
    
    cvr_bce = bce_loss_fn(cvr_pred_clamped, cvr_label.float())
    ctr_loss = bce_loss_fn(ctr_pred_clamped, click_label.float()).sum()
    ctcvr_loss = bce_loss_fn(ctcvr_pred_clamped, ctcvr_label.float()).sum()
    
    if apl_method == 'kl' and apl_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif apl_method == 'mc_dropout' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None:
        ips_weight = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    elif apl_method == 'heuristic' and apl_alpha is not None:
        ips_weight = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    elif apl_method == 'heuristic_kl' and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif apl_method == 'mc_dropout_kl' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_bce, apl_lambda, click_mask, eps)
        weighted_cvr_loss = (q * cvr_bce * batch_size).sum()
    elif tilt_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        weighted_cvr_loss = _get_tilted_weighted_loss(
            ips_weight_base,
            cvr_bce,
            tilt_lambda,
            click_mask,
            batch_size,
            eps,
            detach_ips_weight=(propensity_override is None),
        )
    else:
        ips_weight = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        weighted_cvr_loss = (cvr_bce * ips_weight * click_mask).sum()
    
    total_loss = (0.1 * weighted_cvr_loss + ctr_loss + ctcvr_loss) / batch_size
    return total_loss



    
def escm2_dr_loss(ys, y_preds, apl_alpha=None, ctr_mean=None, ctr_var=None, apl_method=None, apl_lambda=None, eps=1e-6, propensity_override=None, tilt_lambda=None):
    """
    Doubly Robust Loss for CVR estimation with ESCM2.
    
    Args:
        ys: Ground truth labels, shape (batch_size, n_task)
            ys[:, 0] = CVR label (purchase), only valid when clicked
            ys[:, 1] = CTR label (click)
        y_preds: Model predictions (probabilities after sigmoid), shape (batch_size, n_task)
            y_preds[:, 0] = CVR prediction
            y_preds[:, 1] = CTR prediction
            y_preds[:, 2] = CVR imp tower: predict CVR loss
        apl_alpha: Alpha for APL weighting (heuristic/mc_dropout methods)
        ctr_mean: Mean of CTR predictions from MC Dropout (optional)
        ctr_var: Variance of CTR predictions from MC Dropout (optional)
        apl_method: APL method: None, 'heuristic', 'mc_dropout', or 'kl'
        apl_lambda: Lambda for KL method
        eps: Small value to prevent division by zero
    
    Returns:
        (weighted CVR loss + CTR loss) / batch_size
    """
    click_label = ys[:, 1]
    cvr_label = ys[:, 0]
    ctcvr_label = click_label.float() * cvr_label

    ctr_pred = y_preds[:, 1]
    cvr_pred = y_preds[:, 0]
    cvr_imp_pred = y_preds[:, 2]
    ctcvr_pred = ctr_pred * cvr_pred

    ctr_pred_clamped = torch.clamp(ctr_pred, min=eps, max=1.0 - eps)
    cvr_pred_clamped = torch.clamp(cvr_pred, min=eps, max=1.0 - eps)
    cvr_imp_pred_clamped = torch.clamp(cvr_imp_pred, min=eps, max=1.0 - eps)
    ctcvr_pred_clamped = torch.clamp(ctcvr_pred, min=eps, max=1.0 - eps)
    propensity_clamped = torch.clamp(propensity_override, min=eps, max=1.0 - eps) if propensity_override is not None else ctr_pred_clamped

    cvr_loss = bce_loss_fn(cvr_pred_clamped, cvr_label.float())
    ctr_loss = bce_loss_fn(ctr_pred_clamped, click_label.float()).sum()
    ctcvr_loss = bce_loss_fn(ctcvr_pred_clamped, ctcvr_label.float()).sum()
    
    click_mask = click_label.float()
    batch_size = click_label.shape[0]
    
    if apl_method == 'kl' and apl_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif apl_method == 'mc_dropout' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None:
        ips = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
    elif apl_method == 'heuristic' and apl_alpha is not None:
        ips = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
    elif apl_method == 'heuristic_kl' and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif apl_method == 'mc_dropout_kl' and ctr_mean is not None and ctr_var is not None and apl_alpha is not None and apl_lambda is not None:
        ips_weight_base = apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps)
        q = apply_apl_kl(ips_weight_base, cvr_loss, apl_lambda, click_mask, eps)
        ips = q * batch_size
    elif tilt_lambda is not None:
        ips_weight_base = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
        ips = _get_tilted_ips(
            ips_weight_base,
            cvr_loss,
            tilt_lambda,
            click_mask,
            batch_size,
            eps,
            detach_ips_weight=(propensity_override is None),
        )
    else:
        ips = _get_ips_weight_base(propensity_clamped, propensity_override=propensity_override)
    
    e = cvr_loss - cvr_imp_pred_clamped
    dr_part_1 = cvr_imp_pred_clamped
    dr_part_2 = click_mask * e * ips

    dr_loss = (dr_part_1 + dr_part_2).sum()

    e_detached = cvr_loss.detach() - cvr_imp_pred_clamped
    imp_loss = (ips * (e_detached**2)).sum()

    DR_loss = dr_loss + imp_loss

    return (0.1 * DR_loss + ctcvr_loss + ctr_loss) / batch_size



def apply_apl_heuristic(ctr_pred_clamped, apl_alpha, eps=1e-6, max_weight=100.0):
    mu = ctr_pred_clamped.detach()
    sigma = torch.sqrt(mu * (1.0 - mu))
    p_adv = torch.clamp(mu - apl_alpha * sigma, min=eps, max=1.0)
    ips_weight = 1.0 / p_adv
    ips_weight = torch.clamp(ips_weight, max=max_weight)
    # print(f"[APL Heuristic] apl_alpha={apl_alpha}, mu_range=[{mu.min().item():.4f}, {mu.max().item():.4f}], sigma_range=[{sigma.min().item():.4f}, {sigma.max().item():.4f}], weight_range=[{ips_weight.min().item():.4f}, {ips_weight.max().item():.4f}]", flush=True)
    return ips_weight

def apply_apl_mc_dr_dropout(ctr_mean, ctr_var, apl_alpha, eps=1e-6, max_weight=100.0):
    """
    Apply APL (Adversarial Propensity Learning) weighting using MC Dropout uncertainty.
    
    Args:
        ctr_mean: Mean of CTR predictions from MC Dropout, shape (batch_size,)
        ctr_var: Variance of CTR predictions from MC Dropout, shape (batch_size,)
        apl_alpha: Alpha parameter for adversarial propensity
        eps: Small value to prevent division by zero
        max_weight: Maximum IPS weight to prevent explosion
    
    Returns:
        ips_weight: IPS weight based on adversarial propensity
    """
    mu = ctr_mean.detach()
    sigma = torch.sqrt(ctr_var.detach() + eps)
    p_adv = torch.clamp(mu - apl_alpha * sigma, min=eps, max=1.0)
    ips_weight = 1.0 / p_adv
    ips_weight = torch.clamp(ips_weight, max=max_weight)
    # print(f"[APL MC Dropout] apl_alpha={apl_alpha}, mu_range=[{mu.min().item():.4f}, {mu.max().item():.4f}], sigma_range=[{sigma.min().item():.4f}, {sigma.max().item():.4f}], weight_range=[{ips_weight.min().item():.4f}, {ips_weight.max().item():.4f}]", flush=True)
    return ips_weight

def apply_apl_kl(ips_weight, cvr_loss, apl_lambda, click_mask, eps=1e-12, detach_ips_weight=True, detach_loss=True):
    """
    Apply APL (Adversarial Propensity Learning) weighting using KL divergence method.
    
    This method finds an adversarial weight distribution q that maximizes the weighted loss
    under a KL divergence constraint from the original IPS weights.
    
    Mathematical Derivation:
    ------------------------
    Original problem: max_q sum_i(q_i * loss_i)  s.t. KL(q || mu) <= delta
    
    Where:
    - mu_i = IPS weight for sample i (proportional to 1/ctr_pred_i)
    - loss_i = CVR loss for sample i
    - q = adversarial weight distribution
    
    Lagrangian: L = sum_i(q_i * loss_i) - lambda * (KL(q || mu) - delta)
    
    Solving gives analytical solution:
    q_i = mu_i * exp(lambda * loss_i) / sum_j(mu_j * exp(lambda * loss_j))
    
    Dimension Analysis:
    ------------------
    - ips_weight: (batch_size,) - original IPS weights, typically sum ≈ batch_size for clicked samples
    - cvr_loss: (batch_size,) - per-sample BCE loss
    - click_mask: (batch_size,) - 1 for clicked samples, 0 otherwise
    - mu: (batch_size,) - masked IPS weights
    - lam_losses: (batch_size,) - lambda * loss for numerical stability
    - q_tilde: (batch_size,) - unnormalized weights
    - q: (batch_size,) - normalized distribution, sum(q) = 1
    
    Args:
        ips_weight: Original IPS weights (1/ctr_pred), shape (batch_size,)
        cvr_loss: Per-sample CVR loss (BCE), shape (batch_size,)
        apl_lambda: Lambda parameter controlling adversarial strength
        click_mask: Binary mask for clicked samples (1 if clicked, 0 otherwise)
        eps: Small value to prevent division by zero
    
    Returns:
        q: Normalized adversarial weight distribution, shape (batch_size,)
    """
    # mu: masked IPS weights, shape (batch_size,)
    # Only clicked samples have non-zero weights
    mu = (ips_weight.detach() if detach_ips_weight else ips_weight) * click_mask.float()
    
    # lam_losses: lambda * loss for each sample, shape (batch_size,)
    # Only clicked samples contribute to the adversarial weighting
    lam_losses = apl_lambda * (cvr_loss.detach() if detach_loss else cvr_loss) * click_mask.float()
    
    # Numerical stability: subtract max before exp to prevent overflow
    # lam_max: scalar
    lam_max = lam_losses.max()
    
    # q_tilde: unnormalized adversarial weights, shape (batch_size,)
    # q_tilde_i = mu_i * exp(lambda * loss_i - max_loss)
    q_tilde = mu * torch.exp(lam_losses - lam_max)
    
    # q_sum: normalization constant, scalar
    q_sum = q_tilde.sum().clamp(min=eps)
    
    # q: normalized distribution, shape (batch_size,)
    # sum(q) = 1, so we need to multiply by batch_size when using as IPS weights
    q = q_tilde / q_sum
    
    # print(f"[APL KL] apl_lambda={apl_lambda}, q_range=[{q.min().item():.6f}, {q.max().item():.6f}], q_sum={q_sum.item():.4f}", flush=True)
    return q
