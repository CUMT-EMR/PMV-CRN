from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    pr_auc = average_precision_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    return {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0), 'auc': auc, 'pr_auc': pr_auc, 'threshold': threshold, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def ce_loss_from_prob(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob.astype(float), 1e-07, 1.0 - 1e-07)
    y = y.astype(float)
    return float((-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob))).mean())

def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    best = 0.5
    best_score = -1.0
    for th in np.linspace(0.01, 0.99, 99):
        m = metrics(y, prob, float(th))
        score = m['f1'] + 0.25 * m['accuracy']
        if score > best_score:
            best_score = score
            best = float(th)
    return best

@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    return (np.concatenate(ys), np.concatenate(probs))

@torch.no_grad()
def collect_torch_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []
    embs = []
    for x, y in loader:
        x = x.to(device)
        emb = model.encode(x) if hasattr(model, 'encode') else model(x)
        logits = model.head(emb) if hasattr(model, 'head') and hasattr(model, 'encode') else model(x)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
        embs.append(emb.detach().cpu().numpy())
        ys.append(y.numpy())
    return (np.concatenate(ys), np.concatenate(probs), np.concatenate(embs))
