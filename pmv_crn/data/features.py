from __future__ import annotations
import numpy as np
STAT_FEATURE_DIM = 3 * 11 + 3

def make_future_any_labels(labels: np.ndarray, horizon: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros_like(labels, dtype=np.int64)
    for i in range(len(labels)):
        end = min(len(labels), i + max(1, horizon) + 1)
        out[i] = int(labels[i:end].max() > 0)
    return out

def window_stat_features(raw_window: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray) -> np.ndarray:
    x = np.asarray(raw_window, dtype=np.float32)
    d = np.diff(x, axis=0)
    t = np.arange(len(x), dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-06)
    feats = []
    for c in range(x.shape[1]):
        s = x[:, c]
        ds = d[:, c] if len(d) else np.zeros(1, dtype=np.float32)
        slope = float(np.polyfit(t, s, deg=1)[0]) if len(s) > 2 else 0.0
        feats += [float(s[-1]), float(s.mean()), float(s.std()), float(s.min()), float(s.max()), float(np.median(s)), float(np.percentile(s, 75) - np.percentile(s, 25)), float(np.mean(np.abs(ds))), float(np.max(np.abs(ds))), slope, float((s[-1] - global_mean[c]) / (global_std[c] + 1e-06))]
    corr = np.corrcoef(x.T)
    feats += [0.0 if np.isnan(corr[0, 1]) else float(corr[0, 1]), 0.0 if np.isnan(corr[0, 2]) else float(corr[0, 2]), 0.0 if np.isnan(corr[1, 2]) else float(corr[1, 2])]
    return np.asarray(feats, dtype=np.float32)

def compute_stat_normalizer(values: np.ndarray, train_idx: np.ndarray, window: int, global_mean: np.ndarray, global_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    for idx in train_idx:
        start = int(idx) - window + 1
        feats.append(window_stat_features(values[start:int(idx) + 1], global_mean, global_std))
    arr = np.vstack(feats).astype(np.float32)
    return (arr.mean(axis=0), arr.std(axis=0) + 1e-06)
