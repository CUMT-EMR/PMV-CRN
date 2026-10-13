from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pmv_crn.data.representations import standardize_channels
from pmv_crn.data.splits import blocked_random_stratified_split, purge_near_splits

def balanced_subset(indices: np.ndarray, labels: np.ndarray, seed: int, split_offset: int) -> np.ndarray:
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

def make_windows(values: np.ndarray, labels: np.ndarray, indices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_mean = values.mean(axis=0, keepdims=True)
    global_std = values.std(axis=0, keepdims=True) + 1e-06
    xs = []
    ys = []
    for idx in indices:
        start = int(idx) - window + 1
        raw = values[start:int(idx) + 1]
        local = standardize_channels(raw)
        absolute = ((raw - global_mean) / global_std).astype(np.float32)
        xs.append(np.concatenate([local, absolute], axis=1))
        ys.append(int(labels[int(idx)]))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), indices.astype(np.int64))

class WindowDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (self.x[idx], self.y[idx])

def prepare_data(args):
    df = pd.read_csv(args.input_csv)
    if args.max_points is not None and args.max_points > 0:
        df = df.iloc[:args.max_points].copy()
    values = df[['AE', 'EMR', 'GAS']].to_numpy(np.float32)
    labels = df['LABEL'].to_numpy(np.int64)
    all_splits = blocked_random_stratified_split(labels, args.window, args.train_ratio, args.val_ratio, args.seed, args.block_size)
    splits = purge_near_splits(all_splits, purge=args.purge_windows * args.window)
    if args.sample_stride > 1:
        out = []
        for idx in splits:
            keep = (idx - (args.window - 1)) % args.sample_stride == 0
            out.append(idx[keep])
        splits = tuple(out)
    train_idx, val_idx, test_idx = splits
    if args.balance_eval_splits:
        train_idx = balanced_subset(train_idx, labels, args.seed, 11)
        val_idx = balanced_subset(val_idx, labels, args.seed, 17)
        test_idx = balanced_subset(test_idx, labels, args.seed, 23)
    return (*make_windows(values, labels, train_idx, args.window)[:2], *make_windows(values, labels, val_idx, args.window)[:2], *make_windows(values, labels, test_idx, args.window)[:2], (train_idx, val_idx, test_idx))
