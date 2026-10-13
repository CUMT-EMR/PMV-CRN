from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def metrics_from_prob(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float('nan')
        pr_auc = float('nan')
    else:
        auc = float(roc_auc_score(y_true, prob))
        pr_auc = float(average_precision_score(y_true, prob))
    return {'accuracy': float(accuracy_score(y_true, pred)), 'precision': float(precision_score(y_true, pred, zero_division=0)), 'recall': float(recall_score(y_true, pred, zero_division=0)), 'f1': float(f1_score(y_true, pred, zero_division=0)), 'f2': float(fbeta_score_np(y_true, pred, beta=2.0)), 'auc': auc, 'pr_auc': pr_auc, 'threshold': float(threshold), 'true_positive_rate': float((y_true == 1).mean()) if len(y_true) else 0.0, 'pred_positive_rate': float((pred == 1).mean()) if len(pred) else 0.0, 'tp': float(((pred == 1) & (y_true == 1)).sum()), 'tn': float(((pred == 0) & (y_true == 0)).sum()), 'fp': float(((pred == 1) & (y_true == 0)).sum()), 'fn': float(((pred == 0) & (y_true == 1)).sum())}

def fbeta_score_np(y_true: np.ndarray, pred: np.ndarray, beta: float=2.0) -> float:
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return 0.0 if denom <= 0 else float((1.0 + beta2) * precision * recall / denom)

def rolling_mean_prob(prob: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return prob
    return pd.Series(prob).rolling(window, min_periods=1, center=False).mean().to_numpy(dtype=float)

def best_threshold(y_true: np.ndarray, prob: np.ndarray, threshold_min: float=0.05, threshold_max: float=0.95, min_precision: float=0.0) -> float:
    best_score = -1.0
    best_tiebreak = -1.0
    best_th = float(threshold_min)
    fallback_score = -1.0
    fallback_tiebreak = -1.0
    fallback_th = best_th
    threshold_min = float(np.clip(threshold_min, 0.001, 0.999))
    threshold_max = float(np.clip(threshold_max, threshold_min, 0.999))
    for th in np.linspace(threshold_min, threshold_max, 91):
        m = metrics_from_prob(y_true, prob, float(th))
        score = m['f1']
        tiebreak = 0.5 * m['precision'] + 0.5 * m['recall']
        if score > fallback_score or (np.isclose(score, fallback_score) and tiebreak > fallback_tiebreak):
            fallback_score = score
            fallback_tiebreak = tiebreak
            fallback_th = float(th)
        if m['precision'] < min_precision:
            continue
        if score > best_score or (np.isclose(score, best_score) and tiebreak > best_tiebreak):
            best_score = score
            best_tiebreak = tiebreak
            best_th = float(th)
    return best_th if best_score >= 0 else fallback_th
