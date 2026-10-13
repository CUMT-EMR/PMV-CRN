from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from pmv_crn.models.common import BINARY_CLASS_NAMES, BranchCMROutput, IMAGE_CONCEPT_NAMES, TEMPORAL_CONCEPT_NAMES
from pmv_crn.models.concept_rule import CMRRuleMemory
from pmv_crn.models.fusion import CrossAttentionFusion
from pmv_crn.models.image import PrecursorImageEncoder, SensorGroupedImageEncoder
from pmv_crn.models.temporal import MultiScaleTemporalEncoder

class OptimizedDualBranchCMRNet(nn.Module):

    def __init__(self, ts_channels: int=6, img_channels: int=4, n_classes: int=2, d_model: int=96, n_heads: int=4, ts_rules: int=8, img_rules: int=8, dropout: float=0.15, concept_prior_blend: float=0.15, stat_feature_dim: int=0, model_mode: str='full', fixed_temporal_weight: float=0.7, gate_prior_mix: float=0.0):
        super().__init__()
        self.temporal_concept_names = TEMPORAL_CONCEPT_NAMES
        self.image_concept_names = IMAGE_CONCEPT_NAMES
        self.class_names = BINARY_CLASS_NAMES[:n_classes]
        self.stat_feature_dim = int(stat_feature_dim)
        self.model_mode = str(model_mode)
        self.fixed_temporal_weight = float(fixed_temporal_weight)
        self.gate_prior_mix = float(gate_prior_mix)
        self.temporal_encoder = MultiScaleTemporalEncoder(ts_channels, d_model, dropout=dropout)
        self.image_encoder = SensorGroupedImageEncoder(img_channels, d_model, dropout=dropout) if img_channels >= 9 and self.model_mode in {'sensor_image', 'sensor_image_only'} else PrecursorImageEncoder(img_channels, d_model, dropout=dropout)
        self.temporal_cmr = CMRRuleMemory(d_model, len(TEMPORAL_CONCEPT_NAMES), n_classes, ts_rules, dropout=dropout, prior_blend=concept_prior_blend)
        self.image_cmr = CMRRuleMemory(d_model, len(IMAGE_CONCEPT_NAMES), n_classes, img_rules, dropout=dropout, prior_blend=concept_prior_blend)
        self.ts_aux_head = nn.Sequential(nn.Linear(d_model + n_classes, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.img_aux_head = nn.Sequential(nn.Linear(d_model + n_classes, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.img_stat_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes))
        self.fusion = CrossAttentionFusion(d_model, n_heads, dropout)
        fused_stat_dim = 0 if self.model_mode == 'no_stat_features' else self.stat_feature_dim
        self.fused_head = nn.Sequential(nn.Linear(d_model + 4 * n_classes + fused_stat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x_ts: torch.Tensor, x_img: torch.Tensor, ts_concept_prior: Optional[torch.Tensor]=None, img_concept_prior: Optional[torch.Tensor]=None, stat_features: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        ts_tokens, ts_emb = self.temporal_encoder(x_ts)
        img_tokens, img_emb, image_view_weights, image_stat_embedding = self.image_encoder(x_img)
        ts_rule = self.temporal_cmr(ts_emb, ts_concept_prior)
        img_rule = self.image_cmr(img_emb, img_concept_prior)
        ts_logits = self.ts_aux_head(torch.cat([ts_emb, ts_rule.y_rule], dim=-1))
        img_logits_main = self.img_aux_head(torch.cat([img_emb, img_rule.y_rule], dim=-1))
        img_logits_stat = self.img_stat_head(image_stat_embedding)
        img_logits = 0.65 * img_logits_main + 0.35 * img_logits_stat
        if self.model_mode == 'no_image_branch':
            img_tokens_for_fusion = torch.zeros_like(img_tokens)
            img_rule_y_for_head = torch.zeros_like(img_rule.y_rule)
            img_prob_for_head = torch.zeros_like(torch.softmax(img_logits, dim=-1))
        else:
            img_tokens_for_fusion = img_tokens
            img_rule_y_for_head = img_rule.y_rule
            img_prob_for_head = torch.softmax(img_logits, dim=-1)
        fusion = self.fusion(ts_tokens, img_tokens_for_fusion, no_gate=self.model_mode == 'no_gate_fusion', gate_prior_mix=self.gate_prior_mix if self.model_mode == 'full' else 0.0)
        if self.model_mode == 'no_image_branch':
            fusion['gate'] = torch.stack([torch.ones_like(ts_logits[:, 0]), torch.zeros_like(ts_logits[:, 0])], dim=-1)
        head_inputs = [fusion['fused'], ts_rule.y_rule, img_rule_y_for_head, torch.softmax(ts_logits, dim=-1), img_prob_for_head]
        if self.stat_feature_dim > 0 and self.model_mode != 'no_stat_features':
            if stat_features is None:
                stat_features = torch.zeros(fusion['fused'].size(0), self.stat_feature_dim, device=fusion['fused'].device, dtype=fusion['fused'].dtype)
            head_inputs.append(stat_features.to(device=fusion['fused'].device, dtype=fusion['fused'].dtype))
        logits = self.fused_head(torch.cat(head_inputs, dim=-1))
        if self.model_mode == 'temporal_only':
            logits = ts_logits
            fusion['fused'] = ts_emb
            fusion['gate'] = torch.stack([torch.ones_like(ts_logits[:, 0]), torch.zeros_like(ts_logits[:, 0])], dim=-1)
        elif self.model_mode in {'image_only', 'sensor_image_only'}:
            logits = img_logits
            fusion['fused'] = img_emb
            fusion['gate'] = torch.stack([torch.zeros_like(img_logits[:, 0]), torch.ones_like(img_logits[:, 0])], dim=-1)
        elif self.model_mode == 'fixed_weight_fusion':
            alpha = float(np.clip(self.fixed_temporal_weight, 0.0, 1.0))
            ts_prob = torch.softmax(ts_logits, dim=-1)
            img_prob = torch.softmax(img_logits, dim=-1)
            prob = alpha * ts_prob + (1.0 - alpha) * img_prob
            logits = torch.log(prob.clamp_min(1e-06))
            fusion['gate'] = torch.stack([torch.full_like(ts_logits[:, 0], alpha), torch.full_like(ts_logits[:, 0], 1.0 - alpha)], dim=-1)
        elif self.model_mode == 'no_image_branch':
            img_emb = torch.zeros_like(img_emb)
            img_logits = torch.zeros_like(img_logits)
            img_logits_main = torch.zeros_like(img_logits_main)
            img_logits_stat = torch.zeros_like(img_logits_stat)
            img_rule = BranchCMROutput(torch.zeros_like(img_rule.concepts), torch.zeros_like(img_rule.rule_logits), torch.zeros_like(img_rule.rule_weights), torch.zeros_like(img_rule.rule_probs), torch.zeros_like(img_rule.y_rule))
        return {'logits': logits, 'ts_logits': ts_logits, 'img_logits': img_logits, 'y_prob': torch.softmax(logits, dim=-1), 'ts_embedding': ts_emb, 'img_embedding': img_emb, 'fused_embedding': fusion['fused'], 'view_gate': fusion['gate'], 'image_view_weights': image_view_weights, 'image_stat_embedding': image_stat_embedding, 'img_logits_main': img_logits_main, 'img_logits_stat': img_logits_stat, 'attn_ts_to_img': fusion['attn_ts_to_img'], 'attn_img_to_ts': fusion['attn_img_to_ts'], 'ts_concepts': ts_rule.concepts, 'img_concepts': img_rule.concepts, 'ts_rule_weights': ts_rule.rule_weights, 'img_rule_weights': img_rule.rule_weights, 'ts_rule_probs': ts_rule.rule_probs, 'img_rule_probs': img_rule.rule_probs, 'ts_y_rule': ts_rule.y_rule, 'img_y_rule': img_rule.y_rule, 'model_mode': self.model_mode}

    def regularization_loss(self) -> torch.Tensor:
        return self.temporal_cmr.regularization_loss() + self.image_cmr.regularization_loss()
