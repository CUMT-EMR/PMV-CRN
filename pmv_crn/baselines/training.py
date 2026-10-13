from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pmv_crn.baselines.data import WindowDataset
from pmv_crn.baselines.evaluation import best_threshold, ce_loss_from_prob, collect_torch_embeddings, metrics, predict_torch

def train_torch_model(name: str, model: nn.Module, x_train, y_train, x_val, y_val, x_test, y_test, args, out_dir: Path | None=None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    device = torch.device('cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu')
    model = model.to(device)
    train_ds = WindowDataset(x_train, y_train)
    val_ds = WindowDataset(x_val, y_val)
    test_ds = WindowDataset(x_test, y_test)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    target_pos = args.sampler_positive_ratio
    mass = np.asarray([1.0 - target_pos, target_pos], dtype=np.float32)
    weights = [float(mass[int(y)] / max(1.0, counts[int(y)])) for y in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    best_state = None
    best_val = -1.0
    bad = 0
    history_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        vy, vp = predict_torch(model, val_loader, device)
        th = best_threshold(vy, vp)
        val_m = metrics(vy, vp, th)
        epoch_row = {'model': name, 'epoch': epoch, 'train_batch_ce_loss': float(np.mean(batch_losses)) if batch_losses else np.nan, 'threshold': th}
        for split, loader in [('train', eval_train_loader), ('val', val_loader), ('test', test_loader)]:
            sy, sp = predict_torch(model, loader, device)
            sm = metrics(sy, sp, th)
            epoch_row[f'{split}_ce_loss'] = ce_loss_from_prob(sy, sp)
            for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc']:
                epoch_row[f'{split}_{key}'] = sm[key]
        history_rows.append(epoch_row)
        score = val_m['f1'] + val_m['pr_auc']
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    rows = []
    vy, vp = predict_torch(model, val_loader, device)
    th = best_threshold(vy, vp)
    for split, loader in [('train', eval_train_loader), ('val', val_loader), ('test', test_loader)]:
        y, p = predict_torch(model, loader, device)
        rows.append({'model': name, 'split': split, **metrics(y, p, th)})
        if out_dir is not None and args.save_embeddings:
            ey, ep, emb = collect_torch_embeddings(model, loader, device)
            emb_df = pd.DataFrame(emb, columns=[f'emb_{i}' for i in range(emb.shape[1])])
            emb_df.insert(0, 'prob', ep)
            emb_df.insert(0, 'true_label', ey)
            emb_df.insert(0, 'sample_order', np.arange(len(emb_df)))
            emb_df.insert(0, 'split', split)
            emb_dir = out_dir / 'embeddings'
            emb_dir.mkdir(parents=True, exist_ok=True)
            emb_df.to_csv(emb_dir / f'{name}_{split}_embeddings.csv', index=False, encoding='utf-8-sig')
    return (rows, history_rows)

def train_sklearn(name: str, estimator, x_train, y_train, x_val, y_val, x_test, y_test) -> list[dict[str, object]]:
    xtr = x_train.reshape(len(x_train), -1)
    xv = x_val.reshape(len(x_val), -1)
    xt = x_test.reshape(len(x_test), -1)
    estimator.fit(xtr, y_train)

    def prob(est, x):
        if hasattr(est, 'predict_proba'):
            return est.predict_proba(x)[:, 1]
        return est.decision_function(x)
    vp = prob(estimator, xv)
    th = best_threshold(y_val, vp)
    return [{'model': name, 'split': 'train', **metrics(y_train, prob(estimator, xtr), th)}, {'model': name, 'split': 'val', **metrics(y_val, vp, th)}, {'model': name, 'split': 'test', **metrics(y_test, prob(estimator, xt), th)}]
