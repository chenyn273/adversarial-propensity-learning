
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../torch_rechub_main')))

import torch
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path

from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.basic.tracking import TensorBoardXLogger
from utils import (
    ALI_CCP_CONFIG,
    seed_everything,
    calculate_metrics,
    get_ali_ccp_data_loader
)


def sanitize_run_tag(run_tag):
    if not run_tag:
        return None
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', run_tag.strip())
    sanitized = sanitized.strip('_')
    return sanitized or None


def load_completed_seed_results(resume_log_path):
    if not resume_log_path:
        return [], set()

    path = Path(resume_log_path)
    if not path.exists():
        return [], set()

    pattern = re.compile(
        r"Seed\s+(\d+)\s+CVR Results - AUC:\s*([0-9.]+),\s*LogLoss:\s*([0-9.]+),\s*KS:\s*([0-9.]+)"
    )

    recovered = []
    recovered_seeds = set()
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        seed = int(m.group(1))
        if seed in recovered_seeds:
            continue
        recovered_seeds.add(seed)
        recovered.append({
            'seed': seed,
            'auc': float(m.group(2)),
            'logloss': float(m.group(3)),
            'ks': float(m.group(4)),
        })

    return recovered, recovered_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=str, default='ips', choices=['ips'], help='Loss type (only ips for this experiment)')
    parser.add_argument('--apl_alpha', type=float, default=None, help='Alpha for APL heuristic/mc_dropout')
    parser.add_argument('--apl_method', type=str, default=None, choices=[None, 'heuristic', 'mc_dropout', 'kl', 'heuristic_kl', 'mc_dropout_kl'], help='APL method')
    parser.add_argument('--apl_lambda', type=float, default=None, help='Lambda for APL KL method')
    parser.add_argument('--mcdp_n', type=int, default=None, help='Number of MC Dropout samples')
    
    # New arguments for bias injection
    parser.add_argument('--propensity_scale', type=float, default=None, help='Scale factor for propensity score (e.g., 0.8, 1.2)')
    parser.add_argument('--propensity_shift', type=float, default=None, help='Shift value for propensity score (e.g., 0.1, -0.1)')
    parser.add_argument('--propensity_noise', type=float, default=None, help='Noise variance for propensity score (e.g., 0.1)')
    
    # Meta-learning arguments
    parser.add_argument('--apl_learnable', action='store_true', help='Enable meta-learning for APL parameters')
    parser.add_argument('--meta_lr', type=float, default=1e-2, help='Learning rate for meta-optimizer')
    parser.add_argument('--meta_update_interval', type=int, default=2, help='Meta update interval in steps')
    parser.add_argument('--meta_inner_lr', type=float, default=None, help='Inner loop learning rate (defaults to training lr when None)')
    parser.add_argument('--meta_clip_norm', type=float, default=1.0, help='Meta gradient clip norm (<=0 disables)')
    parser.add_argument('--cab_enable', action='store_true', help='Enable fixed CAPC propensity correction')
    parser.add_argument('--cab_bins', type=int, default=20, help='Number of CAB calibration bins')
    parser.add_argument('--cab_beta_init', type=float, default=0.1, help='Initial CAB interpolation coefficient beta')
    parser.add_argument('--cab_calibration_batch_size', type=int, default=8192, help='Calibration refresh batch size for CAB')
    parser.add_argument('--cab_beta_dev_scale', type=float, default=0.0, help='Fixed non-negative scale b for absolute-deviation adaptive CAPC strength; 0 recovers bin-shared beta')
    parser.add_argument('--cab_beta_dev_scale_limit', type=float, default=1.0, help='Maximum allowed absolute-deviation scale for adaptive CAPC beta')
    parser.add_argument('--cab_tilt_enable', action='store_true', help='Enable CAB-Tilt on top of corrected propensities')
    parser.add_argument('--cab_tilt_lambda', type=float, default=3.0, help='Tilt strength for CAB-Tilt')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional suffix to isolate logs/checkpoints for a code version or run')
    parser.add_argument('--resume_log', type=str, default=None, help='Optional existing log path used to recover already completed seeds and continue only the remaining seeds')

    args = parser.parse_args()
    loss_type = args.loss_type
    apl_alpha = args.apl_alpha
    apl_method = args.apl_method
    apl_lambda = args.apl_lambda
    mcdp_n = args.mcdp_n
    propensity_scale = args.propensity_scale
    propensity_shift = args.propensity_shift
    propensity_noise = args.propensity_noise
    apl_learnable = args.apl_learnable
    meta_lr = args.meta_lr
    meta_update_interval = args.meta_update_interval
    meta_inner_lr = args.meta_inner_lr
    meta_clip_norm = args.meta_clip_norm
    cab_enable = args.cab_enable
    cab_bins = args.cab_bins
    cab_beta_init = args.cab_beta_init
    cab_calibration_batch_size = args.cab_calibration_batch_size
    cab_beta_dev_scale = args.cab_beta_dev_scale
    cab_beta_dev_scale_limit = args.cab_beta_dev_scale_limit
    cab_tilt_enable = args.cab_tilt_enable
    cab_tilt_lambda = args.cab_tilt_lambda
    run_tag = sanitize_run_tag(args.run_tag)
    resume_log = args.resume_log

    if cab_enable and apl_method is not None:
        raise ValueError("CAPC and APL are not meant to be enabled together in the same run")
    if cab_enable and apl_learnable:
        raise ValueError("CAPC and APL meta-learning cannot be enabled together")
    if cab_tilt_enable and not cab_enable:
        raise ValueError("CAB-Tilt requires --cab_enable")

    if apl_method in ['mc_dropout', 'mc_dropout_kl'] and mcdp_n is None:
        mcdp_n = 5

    if apl_learnable:
        if apl_method in ['heuristic', 'mc_dropout', 'heuristic_kl', 'mc_dropout_kl'] and apl_alpha is None:
            apl_alpha = 1.0
        if apl_method in ['kl', 'heuristic_kl', 'mc_dropout_kl'] and apl_lambda is None:
            apl_lambda = 3.0
    
    CONFIG = ALI_CCP_CONFIG
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    print(f"Loss type: {loss_type}", flush=True)
    print(f"APL method: {apl_method}", flush=True)
    print(f"APL alpha: {apl_alpha}", flush=True)
    print(f"APL lambda: {apl_lambda}", flush=True)
    print(f"MC Dropout n: {mcdp_n}", flush=True)
    print(f"Propensity Scale: {propensity_scale}", flush=True)
    print(f"Propensity Shift: {propensity_shift}", flush=True)
    print(f"Propensity Noise: {propensity_noise}", flush=True)
    print(f"APL Learnable: {apl_learnable}", flush=True)
    print(f"Meta LR: {meta_lr}", flush=True)
    print(f"Meta Update Interval: {meta_update_interval}", flush=True)
    print(f"Meta Inner LR: {meta_inner_lr}", flush=True)
    print(f"Meta Clip Norm: {meta_clip_norm}", flush=True)
    print(f"CAB Enabled: {cab_enable}", flush=True)
    print(f"CAB Bins: {cab_bins}", flush=True)
    print(f"CAB Beta Init: {cab_beta_init}", flush=True)
    print(f"CAB Calibration Batch Size: {cab_calibration_batch_size}", flush=True)
    print("CAB Anchor Type: anchor_shrinkage (fixed)", flush=True)
    print("CAB Beta Mode: anchor_absdev (fixed)", flush=True)
    print(f"CAB Beta Deviation Scale: {cab_beta_dev_scale}", flush=True)
    print(f"CAB Beta Deviation Scale Limit: {cab_beta_dev_scale_limit}", flush=True)
    print(f"CAB Tilt Enabled: {cab_tilt_enable}", flush=True)
    print(f"CAB Tilt Lambda: {cab_tilt_lambda}", flush=True)
    print(f"Run tag: {run_tag}", flush=True)
    print(f"Resume log: {resume_log}", flush=True)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data/ali-ccp')
    
    print(f"Loading data from {data_path}", flush=True)
    cab_loaders = None
    loader_outputs = get_ali_ccp_data_loader(
        data_path=data_path,
        batch_size=CONFIG.BATCH_SIZE,
        embed_dim=CONFIG.EMBED_DIM,
        loss_type=loss_type,
        cab_enable=cab_enable,
        cab_calibration_batch_size=cab_calibration_batch_size,
    )
    if cab_enable:
        train_dl, val_dl, test_dl, features, task_types, y_test, cab_loaders = loader_outputs
    else:
        train_dl, val_dl, test_dl, features, task_types, y_test = loader_outputs
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test batches: {len(test_dl)}", flush=True)
    print(f"Feature count: {len(features)}", flush=True)
    print(f"Label shape: {y_test.shape}", flush=True)

    recovered_results, recovered_seeds = load_completed_seed_results(resume_log)
    if recovered_seeds:
        print(f"Recovered completed seeds from resume log: {sorted(recovered_seeds)}", flush=True)
    results = list(recovered_results)

    for seed in CONFIG.SEEDS:
        if seed in recovered_seeds:
            print(f"Skipping Seed: {seed} (already present in resume log)", flush=True)
            continue
        print(f"\nRunning with Seed: {seed}", flush=True)
        seed_everything(seed)
        
        model = MMOE(features, task_types, n_expert=CONFIG.N_EXPERTS, 
                     expert_params={"dims": CONFIG.EXPERT_DIMS, "dropout": CONFIG.DROPOUT}, 
                     tower_params_list=[{"dims": CONFIG.TOWER_DIMS, "dropout": CONFIG.DROPOUT}] * len(task_types),
                     mcdp_n=mcdp_n)
        
        exp_name = f"mmoe_aliccp_bias"
        if loss_type:
            exp_name += f"_{loss_type}"
        if apl_method:
            exp_name += f"_{apl_method}"
        if apl_alpha is not None:
            exp_name += f"_alpha{apl_alpha}"
        if apl_lambda is not None:
            exp_name += f"_lambda{apl_lambda}"
        if mcdp_n is not None:
            exp_name += f"_mcdp{mcdp_n}"
        if propensity_scale is not None:
            exp_name += f"_scale{propensity_scale}"
        if propensity_shift is not None:
            exp_name += f"_shift{propensity_shift}"
        if propensity_noise is not None:
            exp_name += f"_noise{propensity_noise}"
        if cab_enable:
            exp_name += f"_cab_b{cab_bins}_beta{cab_beta_init}"
            exp_name += "_anchor_shrinkage"
            exp_name += "_anchor_absdev"
            if cab_beta_dev_scale != 0.0:
                exp_name += f"_dscale{cab_beta_dev_scale}"
            if cab_beta_dev_scale_limit != 1.0:
                exp_name += f"_dlim{cab_beta_dev_scale_limit}"
            if cab_tilt_enable:
                exp_name += f"_tilt{cab_tilt_lambda}"
        if run_tag:
            exp_name += f"_{run_tag}"
        exp_name += f"_seed{seed}"
        
        save_dir = os.path.join(os.path.dirname(__file__), 'saved', exp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        tb = None
        if os.environ.get("CAPC_ENABLE_TENSORBOARD", "0") == "1":
            tb = TensorBoardXLogger(log_dir=os.path.join(save_dir, 'runs'))
            
        trainer = MTLTrainer(model, task_types=task_types, 
                             optimizer_params={"lr": CONFIG.LR, "weight_decay": CONFIG.WEIGHT_DECAY}, 
                             n_epoch=CONFIG.EPOCHS, earlystop_patience=CONFIG.EARLY_STOP_PATIENCE, 
                             earlystop_taskid=0,
                             loss_type=loss_type,
                             device=device, model_path=save_dir,
                             model_logger=tb,
                             apl_alpha=apl_alpha,
                             apl_method=apl_method,
                             apl_lambda=apl_lambda,
                             mcdp_n=mcdp_n,
                             propensity_scale=propensity_scale,
                             propensity_shift=propensity_shift,
                             propensity_noise=propensity_noise,
                             apl_learnable=apl_learnable,
                             meta_lr=meta_lr,
                             meta_update_interval=meta_update_interval,
                             meta_inner_lr=meta_inner_lr,
                             meta_clip_norm=meta_clip_norm,
                             cab_enable=cab_enable,
                             cab_bins=cab_bins,
                             cab_beta_init=cab_beta_init,
                             cab_beta_dev_scale=cab_beta_dev_scale,
                             cab_beta_dev_scale_limit=cab_beta_dev_scale_limit,
                             cab_tilt_enable=cab_tilt_enable,
                             cab_tilt_lambda=cab_tilt_lambda)
        
        print(f"Start training{' (' + loss_type + ')' if loss_type else ''}...", flush=True)
        trainer.fit(
            train_dl,
            val_dl,
            save_model=False,
            cab_calibration_loader=cab_loaders["calibration_dataloader"] if cab_loaders is not None else None,
        )
        
        print("Start evaluation...", flush=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for x_dict, _ in test_dl:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                y_p, _, _ = model(x_dict)
                preds.extend(y_p.cpu().numpy())
        preds = np.array(preds)
        
        y_true_cvr = y_test[:, 0]
        y_pred_cvr = preds[:, 0]
        
        auc, ll, ks = calculate_metrics(y_true_cvr, y_pred_cvr)
        print(f"Seed {seed} CVR Results - AUC: {auc:.4f}, LogLoss: {ll:.4f}, KS: {ks:.4f}")
        
        results.append({
            'seed': seed,
            'auc': auc,
            'logloss': ll,
            'ks': ks
        })

    if not results:
        raise RuntimeError("No seed results available for aggregation")

    df_res = pd.DataFrame(results).sort_values("seed").reset_index(drop=True)
    print("\n" + "="*30)
    print(f"Final Aggregated Results (CVR){' - ' + loss_type.upper() if loss_type else ''}")
    print("="*30)
    print(df_res)
    print("-" * 30)
    print(f"CVR AUC:     {df_res['auc'].mean():.4f} ± {df_res['auc'].std():.4f}")
    print(f"CVR LogLoss: {df_res['logloss'].mean():.4f} ± {df_res['logloss'].std():.4f}")
    print(f"CVR KS:      {df_res['ks'].mean():.4f} ± {df_res['ks'].std():.4f}")
    print("="*30)

if __name__ == '__main__':
    main()
