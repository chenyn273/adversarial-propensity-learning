import os
import sys
import time
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../torch_rechub_main')))

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.trainers.cab_utils import CAB_SAMPLE_ID_KEY
from torch_rechub.utils.data import DataGenerator, TorchDataset

# ==========================================
# Configuration
# ==========================================

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Ali-CCP Configuration
ALI_CCP_CONFIG = Config(
    EMBED_DIM=32,
    TOWER_DIMS=[320, 200, 80],
    EXPERT_DIMS=[64],
    N_EXPERTS=8,
    DROPOUT=0.1,
    BATCH_SIZE=1024,
    LR=0.001,
    WEIGHT_DECAY=1e-4,
    EPOCHS=50,
    EARLY_STOP_PATIENCE=2,
    SEEDS=[0, 1, 2, 3, 4]
)

# AliExpress Configuration
ALI_EXPRESS_CONFIG = Config(
    EMBED_DIM=32,
    TOWER_DIMS=[64, 64, 32],
    EXPERT_DIMS=[64],
    N_EXPERTS=8,
    DROPOUT=0.1,
    BATCH_SIZE=1024,
    LR=0.001,
    WEIGHT_DECAY=1e-4,
    EPOCHS=50,
    EARLY_STOP_PATIENCE=2,
    SEEDS=[0, 1, 2, 3, 4]
)

# ==========================================
# Helper Functions
# ==========================================

def seed_everything(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_metrics(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
    except ValueError:
        # Handle cases with single class in batch/test set
        auc, ll, ks = 0.5, 0.0, 0.0
    return auc, ll, ks

def _build_cache_key(file_paths, extra):
    parts = []
    for path in sorted(file_paths):
        stat = os.stat(path)
        parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
    extra_str = json.dumps(extra, sort_keys=True, ensure_ascii=False)
    raw = "|".join(parts) + "|" + extra_str
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _acquire_lock(lock_path):
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.5)

def _release_lock(lock_path):
    if os.path.exists(lock_path):
        os.remove(lock_path)

def _read_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_meta(meta_path, meta):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def _save_memmap(cache_dir, arrays):
    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(cache_dir, f"{name}.npy"), arr, allow_pickle=False)
    return list(arrays.keys())

def _load_memmap(cache_dir, array_names):
    return {name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r") for name in array_names}


def _build_cab_loaders(x_train, y_train, x_val, y_val, batch_size, cab_calibration_batch_size=None, num_workers=0):
    train_size = len(y_train)
    x_train_with_id = dict(x_train)
    x_train_with_id[CAB_SAMPLE_ID_KEY] = np.arange(train_size, dtype=np.int32)

    calibration_batch_size = cab_calibration_batch_size if cab_calibration_batch_size is not None else max(batch_size * 4, batch_size)

    calibration_loader = DataLoader(
        TorchDataset(x_train_with_id, y_train),
        batch_size=calibration_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return x_train_with_id, {
        "calibration_dataloader": calibration_loader,
    }

# ==========================================
# Data Loaders
# ==========================================

def get_ali_ccp_data_loader(data_path, batch_size, embed_dim, loss_type=None, cab_enable=False, cab_calibration_batch_size=None, num_workers=0):
    """
    Load and preprocess Ali-CCP dataset.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size for dataloader
        embed_dim: Embedding dimension for sparse features
        loss_type: Loss type, None (default), 'ips', or 'doubly_robust'
    """
    print(f"Loading Ali-CCP data from {data_path}")
    cache_dir = os.path.join(data_path, ".apl_cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(data_path, "ali_ccp_train.csv")
    val_path = os.path.join(data_path, "ali_ccp_val.csv")
    test_path = os.path.join(data_path, "ali_ccp_test.csv")
    cache_key = _build_cache_key([train_path, val_path, test_path], {"dataset": "ali_ccp", "cache_format": "memmap_v1"})
    cache_root = os.path.join(cache_dir, cache_key)
    meta_path = os.path.join(cache_root, "meta.json")
    lock_path = os.path.join(cache_dir, f"{cache_key}.lock")

    if os.path.exists(meta_path):
        meta = _read_meta(meta_path)
        cached = _load_memmap(cache_root, meta["array_names"])
        used_cols = meta["used_cols"]
        sparse_cols = meta["sparse_cols"]
        dense_cols = meta["dense_cols"]
        vocab_sizes = meta["vocab_sizes"]
        x_train = {name: cached[f"x_train__{name}"] for name in used_cols}
        x_val = {name: cached[f"x_val__{name}"] for name in used_cols}
        x_test = {name: cached[f"x_test__{name}"] for name in used_cols}
        y_train = cached["y_train"]
        y_val = cached["y_val"]
        y_test = cached["y_test"]
    else:
        _acquire_lock(lock_path)
        try:
            if os.path.exists(meta_path):
                meta = _read_meta(meta_path)
                cached = _load_memmap(cache_root, meta["array_names"])
                used_cols = meta["used_cols"]
                sparse_cols = meta["sparse_cols"]
                dense_cols = meta["dense_cols"]
                vocab_sizes = meta["vocab_sizes"]
                x_train = {name: cached[f"x_train__{name}"] for name in used_cols}
                x_val = {name: cached[f"x_val__{name}"] for name in used_cols}
                x_test = {name: cached[f"x_test__{name}"] for name in used_cols}
                y_train = cached["y_train"]
                y_val = cached["y_val"]
                y_test = cached["y_test"]
            else:
                df_train = pd.read_csv(train_path)
                df_val = pd.read_csv(val_path)
                df_test = pd.read_csv(test_path)
                data = pd.concat([df_train, df_val, df_test], axis=0)
                data.rename(columns={"purchase": "cvr_label", "click": "ctr_label"}, inplace=True)
                train_idx = len(df_train)
                val_idx = len(df_train) + len(df_val)
                col_names = data.columns.values.tolist()
                dense_cols = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
                sparse_cols = [col for col in col_names if col not in dense_cols and col not in ["cvr_label", "ctr_label"]]
                used_cols = sparse_cols + dense_cols
                vocab_sizes = {col: int(data[col].max() + 1) for col in sparse_cols}
                label_cols = ["cvr_label", "ctr_label"]
                x_train = {name: data[name].values[:train_idx] for name in used_cols}
                x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
                x_test = {name: data[name].values[val_idx:] for name in used_cols}
                y_train = data[label_cols].values[:train_idx]
                y_val = data[label_cols].values[train_idx:val_idx]
                y_test = data[label_cols].values[val_idx:]
                arrays = {"y_train": y_train, "y_val": y_val, "y_test": y_test}
                for name in used_cols:
                    arrays[f"x_train__{name}"] = x_train[name]
                    arrays[f"x_val__{name}"] = x_val[name]
                    arrays[f"x_test__{name}"] = x_test[name]
                array_names = _save_memmap(cache_root, arrays)
                meta = {
                    "used_cols": used_cols,
                    "sparse_cols": sparse_cols,
                    "dense_cols": dense_cols,
                    "vocab_sizes": vocab_sizes,
                    "array_names": array_names
                }
                _write_meta(meta_path, meta)
        finally:
            _release_lock(lock_path)
    
    features = [SparseFeature(col, vocab_sizes[col], embed_dim=embed_dim) for col in sparse_cols] + [DenseFeature(col) for col in dense_cols]

    task_types = ["classification", "classification"]
    
    if loss_type in ["doubly_robust", "escm2_dr", "doubly_robust_apl_heur", "escm2_dr_apl_heur"]:
        # cvr, ctr, imp
        task_types = ["classification", "classification", "regression"]
    
    cab_loaders = None
    train_x = x_train
    if cab_enable:
        train_x, cab_loaders = _build_cab_loaders(
            x_train,
            y_train,
            x_val,
            y_val,
            batch_size=batch_size,
            cab_calibration_batch_size=cab_calibration_batch_size,
            num_workers=num_workers,
        )

    dg = DataGenerator(train_x, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(
        x_val=x_val, y_val=y_val, 
        x_test=x_test, y_test=y_test, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    if cab_enable:
        return train_dataloader, val_dataloader, test_dataloader, features, task_types, y_test, cab_loaders
    return train_dataloader, val_dataloader, test_dataloader, features, task_types, y_test

def get_aliexpress_data_loader(data_path, batch_size, embed_dim, region, loss_type=None, cab_enable=False, cab_calibration_batch_size=None, num_workers=0):
    """
    Load and preprocess AliExpress dataset.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size for dataloader
        embed_dim: Embedding dimension for sparse features
        region: Region code (ES, FR, NL, US)
        loss_type: Loss type, None (default), 'ips', or 'doubly_robust'
    """
    cache_dir = os.path.join(data_path, ".apl_cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(data_path, region, f"aliexpress_{region}_train.csv")
    val_path = os.path.join(data_path, region, f"aliexpress_{region}_val.csv")
    test_path = os.path.join(data_path, region, f"aliexpress_{region}_test.csv")
    cache_key = _build_cache_key([train_path, val_path, test_path], {"dataset": "aliexpress", "region": region, "cache_format": "memmap_v1"})
    cache_root = os.path.join(cache_dir, cache_key)
    meta_path = os.path.join(cache_root, "meta.json")
    lock_path = os.path.join(cache_dir, f"{cache_key}.lock")

    if os.path.exists(meta_path):
        meta = _read_meta(meta_path)
        cached = _load_memmap(cache_root, meta["array_names"])
        input_cols = meta["input_cols"]
        sparse_cols = meta["sparse_cols"]
        dense_cols = meta["dense_cols"]
        vocab_sizes = meta["vocab_sizes"]
        x_train = {name: cached[f"x_train__{name}"] for name in input_cols}
        x_val = {name: cached[f"x_val__{name}"] for name in input_cols}
        x_test = {name: cached[f"x_test__{name}"] for name in input_cols}
        y_train = cached["y_train"]
        y_val = cached["y_val"]
        y_test = cached["y_test"]
    else:
        _acquire_lock(lock_path)
        try:
            if os.path.exists(meta_path):
                meta = _read_meta(meta_path)
                cached = _load_memmap(cache_root, meta["array_names"])
                input_cols = meta["input_cols"]
                sparse_cols = meta["sparse_cols"]
                dense_cols = meta["dense_cols"]
                vocab_sizes = meta["vocab_sizes"]
                x_train = {name: cached[f"x_train__{name}"] for name in input_cols}
                x_val = {name: cached[f"x_val__{name}"] for name in input_cols}
                x_test = {name: cached[f"x_test__{name}"] for name in input_cols}
                y_train = cached["y_train"]
                y_val = cached["y_val"]
                y_test = cached["y_test"]
            else:
                df_train = pd.read_csv(train_path)
                df_val = pd.read_csv(val_path)
                df_test = pd.read_csv(test_path)
                data = pd.concat([df_train, df_val, df_test], axis=0)
                train_idx = len(df_train)
                val_idx = len(df_train) + len(df_val)
                u_cols = [c for c in data.columns if c.startswith("u_")]
                i_cols = [c for c in data.columns if c.startswith("i_")]
                u_dense = [f"u_{i}" for i in range(8, 31)]
                i_int = {"i_18", "i_33", "i_34", "i_35", "i_39", "i_40", "i_41"}
                dense_cols = u_dense + [c for c in i_cols if c not in i_int]
                sparse_cols = [c for c in u_cols + i_cols if c not in dense_cols]
                input_cols = u_cols + i_cols
                vocab_sizes = {col: int(data[col].max() + 1) for col in sparse_cols}
                label_cols = ["purchase", "click"]
                x_train = {name: data[name].values[:train_idx] for name in input_cols}
                x_val = {name: data[name].values[train_idx:val_idx] for name in input_cols}
                x_test = {name: data[name].values[val_idx:] for name in input_cols}
                y_train = data[label_cols].values[:train_idx]
                y_val = data[label_cols].values[train_idx:val_idx]
                y_test = data[label_cols].values[val_idx:]
                arrays = {"y_train": y_train, "y_val": y_val, "y_test": y_test}
                for name in input_cols:
                    arrays[f"x_train__{name}"] = x_train[name]
                    arrays[f"x_val__{name}"] = x_val[name]
                    arrays[f"x_test__{name}"] = x_test[name]
                array_names = _save_memmap(cache_root, arrays)
                meta = {
                    "input_cols": input_cols,
                    "sparse_cols": sparse_cols,
                    "dense_cols": dense_cols,
                    "vocab_sizes": vocab_sizes,
                    "array_names": array_names
                }
                _write_meta(meta_path, meta)
        finally:
            _release_lock(lock_path)

    features = [SparseFeature(col, vocab_sizes[col], embed_dim=embed_dim) for col in sparse_cols] + [DenseFeature(col) for col in dense_cols]
    task_types = ["classification", "classification"]
    
    if loss_type in ["doubly_robust", "escm2_dr", "doubly_robust_apl_heur", "escm2_dr_apl_heur"]:
        # cvr, ctr, imp
        task_types = ["classification", "classification", "regression"]
    
    cab_loaders = None
    train_x = x_train
    if cab_enable:
        train_x, cab_loaders = _build_cab_loaders(
            x_train,
            y_train,
            x_val,
            y_val,
            batch_size=batch_size,
            cab_calibration_batch_size=cab_calibration_batch_size,
            num_workers=num_workers,
        )

    dg = DataGenerator(train_x, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(
        x_val=x_val, y_val=y_val, 
        x_test=x_test, y_test=y_test, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    if cab_enable:
        return train_dataloader, val_dataloader, test_dataloader, features, task_types, y_test, cab_loaders
    return train_dataloader, val_dataloader, test_dataloader, features, task_types, y_test
