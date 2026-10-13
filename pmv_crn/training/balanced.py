from __future__ import annotations
import numpy as np
from pmv_crn.training import engine as tmpmodel
from pmv_crn.models.variants import OptimizedDualBranchCMRNet
_original_split_indices = tmpmodel.split_indices

def _balanced_subset(indices: np.ndarray, labels: np.ndarray, seed: int, split_offset: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    y = labels[indices].astype(np.int64)
    pos = indices[y == 1]
    neg = indices[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.sort(indices)
    n = min(len(pos), len(neg))
    rng = np.random.default_rng(int(seed) + split_offset)
    pos_keep = rng.choice(pos, size=n, replace=False) if len(pos) > n else pos
    neg_keep = rng.choice(neg, size=n, replace=False) if len(neg) > n else neg
    return np.sort(np.concatenate([pos_keep, neg_keep]).astype(np.int64))

def balanced_split_indices(labels, window, train_ratio, val_ratio, seed, max_normal_segment_len, purge_windows, split_mode, block_size):
    train_idx, val_idx, test_idx = _original_split_indices(labels, window, train_ratio, val_ratio, seed, max_normal_segment_len, purge_windows, split_mode, block_size)
    labels = np.asarray(labels, dtype=np.int64)
    return (_balanced_subset(train_idx, labels, seed, 11), _balanced_subset(val_idx, labels, seed, 17), _balanced_subset(test_idx, labels, seed, 23))
tmpmodel.OptimizedDualBranchCMRNet = OptimizedDualBranchCMRNet
tmpmodel.split_indices = balanced_split_indices
