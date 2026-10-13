from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from pmv_crn.models.network import OptimizedDualBranchCMRNet

def supervised_cross_view_contrastive_loss(z_ts: torch.Tensor, z_img: torch.Tensor, y: torch.Tensor, temperature: float=0.2) -> torch.Tensor:
    if z_ts.size(0) <= 2:
        return z_ts.new_tensor(0.0)
    z_ts = F.normalize(z_ts, dim=-1)
    z_img = F.normalize(z_img, dim=-1)
    y = y.view(-1, 1)
    pos_mask = torch.eq(y, y.T).float().to(z_ts.device)
    if pos_mask.sum() <= 0:
        return z_ts.new_tensor(0.0)
    logits = z_ts @ z_img.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob_ts = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_ts = -(pos_mask * log_prob_ts).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)
    logits_t = logits.T
    log_prob_img = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
    loss_img = -(pos_mask.T * log_prob_img).sum(dim=1) / pos_mask.T.sum(dim=1).clamp_min(1.0)
    return 0.5 * (loss_ts.mean() + loss_img.mean())

def prototype_loss(z: torch.Tensor, y: torch.Tensor, temperature: float=0.2) -> torch.Tensor:
    classes = torch.unique(y.long())
    if classes.numel() < 2:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=-1)
    prototypes = []
    targets = torch.zeros_like(y.long())
    for proto_idx, cls in enumerate(classes):
        mask = y.long() == cls
        prototypes.append(z[mask].mean(dim=0))
        targets[mask] = proto_idx
    prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
    logits = z @ prototypes.T / temperature
    return F.cross_entropy(logits, targets)

def supervised_embedding_loss(z: torch.Tensor, y: torch.Tensor, temperature: float=0.2) -> torch.Tensor:
    if z.size(0) <= 2:
        return z.new_tensor(0.0)
    y = y.long()
    classes = torch.unique(y)
    if classes.numel() < 2:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = (y[:, None] == y[None, :]) & ~eye
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return z.new_tensor(0.0)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    logits_mask = ~eye
    log_prob = sim - torch.logsumexp(sim.masked_fill(~logits_mask, -1000000000.0), dim=1, keepdim=True)
    supcon = -(pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    return supcon[valid].mean()

def rule_consistency_loss(ts_y_rule: torch.Tensor, img_y_rule: torch.Tensor) -> torch.Tensor:
    ts_prob = ts_y_rule / ts_y_rule.sum(dim=-1, keepdim=True).clamp_min(1e-06)
    img_prob = img_y_rule / img_y_rule.sum(dim=-1, keepdim=True).clamp_min(1e-06)
    return F.mse_loss(ts_prob, img_prob)

def view_gate_balance_loss(view_gate: torch.Tensor, target_image_share: float=0.35) -> torch.Tensor:
    target = view_gate.new_tensor([1.0 - target_image_share, target_image_share])
    mean_gate = view_gate.mean(dim=0)
    entropy = -(view_gate.clamp_min(1e-06) * view_gate.clamp_min(1e-06).log()).sum(dim=-1).mean()
    max_entropy = view_gate.new_tensor(float(np.log(view_gate.size(-1))))
    return F.mse_loss(mean_gate, target) + 0.1 * (max_entropy - entropy)

def view_diversity_loss(view_weights: torch.Tensor) -> torch.Tensor:
    if view_weights.size(-1) <= 1:
        return view_weights.new_tensor(0.0)
    entropy = -(view_weights.clamp_min(1e-06) * view_weights.clamp_min(1e-06).log()).sum(dim=-1).mean()
    max_entropy = view_weights.new_tensor(float(np.log(view_weights.size(-1))))
    return max_entropy - entropy

def branch_positive_rate_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)[:, 1]
    target_rate = y_true.float().mean().clamp(0.02, 0.98)
    return F.mse_loss(prob.mean(), target_rate)

def focal_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float=0.75, gamma: float=2.0, weight: Optional[torch.Tensor]=None) -> torch.Tensor:
    ce = F.cross_entropy(logits, y.long(), weight=weight, reduction='none')
    pt = torch.exp(-ce).clamp(1e-06, 1.0)
    alpha_t = torch.where(y.long() == 1, torch.full_like(pt, alpha), torch.full_like(pt, 1.0 - alpha))
    return (alpha_t * (1.0 - pt).pow(gamma) * ce).mean()

def optimized_dual_branch_cmr_loss(outputs: Dict[str, torch.Tensor], y_true: torch.Tensor, model: Optional[OptimizedDualBranchCMRNet]=None, concept_targets_ts: Optional[torch.Tensor]=None, concept_targets_img: Optional[torch.Tensor]=None, class_weights: Optional[torch.Tensor]=None, loss_type: str='ce', focal_alpha: float=0.75, focal_gamma: float=2.0, lambda_aux: float=0.3, lambda_contrast: float=0.02, lambda_rule_consistency: float=0.02, lambda_concept: float=0.02, lambda_rule_sparse: float=0.001, lambda_proto: float=0.01, lambda_img_aux: float=0.0, lambda_image_metric: float=0.0, lambda_image_proto: float=0.0, lambda_image_view_diversity: float=0.0, lambda_img_rate: float=0.0, lambda_gate_balance: float=0.0, gate_target_image_share: float=0.35, metric_temperature: float=0.2) -> Dict[str, torch.Tensor]:
    mode = str(outputs.get('model_mode', 'full'))
    if class_weights is not None:
        class_weights = class_weights.to(outputs['logits'].device)
    if loss_type == 'focal':
        cls_loss = focal_loss(outputs['logits'], y_true, focal_alpha, focal_gamma, class_weights)
        ts_cls_loss = focal_loss(outputs['ts_logits'], y_true, focal_alpha, focal_gamma, class_weights)
        img_cls_loss = focal_loss(outputs['img_logits'], y_true, focal_alpha, focal_gamma, class_weights)
    else:
        cls_loss = F.cross_entropy(outputs['logits'], y_true.long(), weight=class_weights)
        ts_cls_loss = F.cross_entropy(outputs['ts_logits'], y_true.long(), weight=class_weights)
        img_cls_loss = F.cross_entropy(outputs['img_logits'], y_true.long(), weight=class_weights)
    aux_loss = 0.5 * (ts_cls_loss + img_cls_loss)
    contrast = supervised_cross_view_contrastive_loss(outputs['ts_embedding'], outputs['img_embedding'], y_true, metric_temperature)
    proto = prototype_loss(outputs['fused_embedding'], y_true, metric_temperature)
    image_proto = prototype_loss(outputs['img_embedding'], y_true, metric_temperature)
    image_metric = supervised_embedding_loss(outputs['img_embedding'], y_true, metric_temperature)
    image_view_diversity = view_diversity_loss(outputs['image_view_weights']) if 'image_view_weights' in outputs else outputs['logits'].new_tensor(0.0)
    img_rate = branch_positive_rate_loss(outputs['img_logits'], y_true)
    rule_cons = rule_consistency_loss(outputs['ts_y_rule'], outputs['img_y_rule'])
    if mode == 'no_image_branch':
        img_cls_loss = outputs['logits'].new_tensor(0.0)
        image_metric = outputs['logits'].new_tensor(0.0)
        image_proto = outputs['logits'].new_tensor(0.0)
        image_view_diversity = outputs['logits'].new_tensor(0.0)
        img_rate = outputs['logits'].new_tensor(0.0)
        rule_cons = outputs['logits'].new_tensor(0.0)
    concept_loss = outputs['logits'].new_tensor(0.0)
    if concept_targets_ts is not None:
        concept_loss = concept_loss + F.binary_cross_entropy(outputs['ts_concepts'], concept_targets_ts.float())
    if concept_targets_img is not None:
        concept_loss = concept_loss + F.binary_cross_entropy(outputs['img_concepts'], concept_targets_img.float())
    if model is not None and mode == 'no_image_branch':
        reg = model.temporal_cmr.regularization_loss()
    else:
        reg = model.regularization_loss() if model is not None else outputs['logits'].new_tensor(0.0)
    gate_balance = view_gate_balance_loss(outputs['view_gate'], gate_target_image_share)
    total = cls_loss + lambda_aux * aux_loss + lambda_img_aux * img_cls_loss + lambda_contrast * contrast + lambda_proto * proto + lambda_image_metric * image_metric + lambda_image_proto * image_proto + lambda_image_view_diversity * image_view_diversity + lambda_img_rate * img_rate + lambda_rule_consistency * rule_cons + lambda_concept * concept_loss + lambda_rule_sparse * reg + lambda_gate_balance * gate_balance
    return {'loss': total, 'cls_loss': cls_loss, 'aux_loss': aux_loss, 'contrast_loss': contrast, 'prototype_loss': proto, 'image_prototype_loss': image_proto, 'image_metric_loss': image_metric, 'image_view_diversity_loss': image_view_diversity, 'image_positive_rate_loss': img_rate, 'rule_consistency_loss': rule_cons, 'concept_loss': concept_loss, 'rule_regularization': reg, 'gate_balance_loss': gate_balance}
