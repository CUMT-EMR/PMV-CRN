from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pmv_crn.evaluation.metrics import best_threshold, metrics_from_prob, rolling_mean_prob
from pmv_crn.models.network import OptimizedDualBranchCMRNet
from pmv_crn.training.batch import forward_model_batch
from pmv_crn.training.losses import optimized_dual_branch_cmr_loss

@torch.no_grad()
def predict(model: OptimizedDualBranchCMRNet, loader: DataLoader, device: torch.device, threshold: float, smooth_window: int, loss_kwargs: Dict[str, object] | None=None) -> tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    loss_values: list[float] = []
    loss_component_values: Dict[str, list[float]] = {}
    for batch in loader:
        x_ts = batch['x_ts'].to(device)
        x_img = batch['x_img'].to(device)
        ts_concepts = batch['ts_concepts'].to(device)
        img_concepts = batch['img_concepts'].to(device)
        stat_features = batch['stat_features'].to(device)
        y_tensor = batch['y'].to(device)
        out = forward_model_batch(model, batch, device)
        if loss_kwargs is not None:
            losses = optimized_dual_branch_cmr_loss(out, y_tensor, model=model, concept_targets_ts=ts_concepts if float(loss_kwargs.get('lambda_concept', 0.0)) > 0 else None, concept_targets_img=img_concepts if float(loss_kwargs.get('lambda_concept', 0.0)) > 0 else None, class_weights=loss_kwargs.get('class_weights'), loss_type=str(loss_kwargs.get('loss_type', 'ce')), focal_alpha=float(loss_kwargs.get('focal_alpha', 0.75)), focal_gamma=float(loss_kwargs.get('focal_gamma', 2.0)), lambda_aux=float(loss_kwargs.get('lambda_aux', 0.0)), lambda_contrast=float(loss_kwargs.get('lambda_contrast', 0.0)), lambda_rule_consistency=float(loss_kwargs.get('lambda_rule_consistency', 0.0)), lambda_concept=float(loss_kwargs.get('lambda_concept', 0.0)), lambda_rule_sparse=float(loss_kwargs.get('lambda_rule_sparse', 0.0)), lambda_proto=float(loss_kwargs.get('lambda_proto', 0.0)), lambda_img_aux=float(loss_kwargs.get('lambda_img_aux', 0.0)), lambda_image_metric=float(loss_kwargs.get('lambda_image_metric', 0.0)), lambda_image_proto=float(loss_kwargs.get('lambda_image_proto', 0.0)), lambda_image_view_diversity=float(loss_kwargs.get('lambda_image_view_diversity', 0.0)), lambda_img_rate=float(loss_kwargs.get('lambda_img_rate', 0.0)), lambda_gate_balance=float(loss_kwargs.get('lambda_gate_balance', 0.0)), gate_target_image_share=float(loss_kwargs.get('gate_target_image_share', 0.35)), metric_temperature=float(loss_kwargs.get('metric_temperature', 0.2)))
            batch_n = int(y_tensor.numel())
            loss_values.append(float(losses['loss'].detach().cpu()) * batch_n)
            for loss_name, loss_value in losses.items():
                loss_component_values.setdefault(loss_name, []).append(float(loss_value.detach().cpu()) * batch_n)
        prob = torch.softmax(out['logits'], dim=-1)[:, 1].detach().cpu().numpy()
        ts_prob = torch.softmax(out['ts_logits'], dim=-1)[:, 1].detach().cpu().numpy()
        img_prob = torch.softmax(out['img_logits'], dim=-1)[:, 1].detach().cpu().numpy()
        gate = out['view_gate'].detach().cpu().numpy()
        y = y_tensor.detach().cpu().numpy()
        raw_y = batch['raw_y'].detach().cpu().numpy()
        ts_con = out['ts_concepts'].detach().cpu().numpy()
        img_con = out['img_concepts'].detach().cpu().numpy()
        ts_rule = out['ts_rule_weights'].detach().cpu().numpy()
        img_rule = out['img_rule_weights'].detach().cpu().numpy()
        for i in range(len(y)):
            row = {'point_index': batch['point_index'][i], 'date': batch['date'][i], 'AE': batch['AE'][i], 'EMR': batch['EMR'][i], 'GAS': batch['GAS'][i], 'true_label': int(y[i]), 'raw_label': int(raw_y[i]), 'prob': float(prob[i]), 'ts_prob': float(ts_prob[i]), 'img_prob': float(img_prob[i]), 'temporal_gate': float(gate[i, 0]), 'image_gate': float(gate[i, 1]), 'threshold': float(threshold)}
            for j in range(ts_con.shape[1]):
                row[f'ts_concept_{j}'] = float(ts_con[i, j])
            for j in range(img_con.shape[1]):
                row[f'img_concept_{j}'] = float(img_con[i, j])
            for j in range(ts_rule.shape[1]):
                row[f'ts_rule_{j}'] = float(ts_rule[i, j])
            for j in range(img_rule.shape[1]):
                row[f'img_rule_{j}'] = float(img_rule[i, j])
            rows.append(row)
    df = pd.DataFrame(rows).sort_values('point_index').reset_index(drop=True)
    df['prob_smooth'] = rolling_mean_prob(df['prob'].to_numpy(), smooth_window)
    df['pred_label'] = (df['prob_smooth'] >= threshold).astype(int)
    m = metrics_from_prob(df['true_label'].to_numpy(), df['prob_smooth'].to_numpy(), threshold)
    y_true = df['true_label'].to_numpy()
    for branch_name, prob_col in [('ts', 'ts_prob'), ('img', 'img_prob')]:
        branch_prob = df[prob_col].to_numpy()
        branch_th = best_threshold(y_true, branch_prob, threshold_min=0.001, threshold_max=0.999)
        branch_m = metrics_from_prob(y_true, branch_prob, branch_th)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'auc', 'pr_auc', 'threshold', 'pred_positive_rate']:
            m[f'{branch_name}_{key}'] = branch_m[key]
    m['temporal_gate_mean'] = float(df['temporal_gate'].mean()) if len(df) else float('nan')
    m['image_gate_mean'] = float(df['image_gate'].mean()) if len(df) else float('nan')
    raw_m = metrics_from_prob(df['raw_label'].to_numpy(), df['prob_smooth'].to_numpy(), threshold)
    for k, v in raw_m.items():
        m[f'raw_{k}'] = v
    m['n_points'] = float(len(df))
    if loss_values:
        m['loss'] = float(np.sum(loss_values) / max(1.0, m['n_points']))
        for loss_name, values in loss_component_values.items():
            m[f'loss_{loss_name}'] = float(np.sum(values) / max(1.0, m['n_points']))
    return (m, df)
