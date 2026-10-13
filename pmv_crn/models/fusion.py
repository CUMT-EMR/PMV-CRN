from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model: int=96, n_heads: int=4, dropout: float=0.15):
        super().__init__()
        self.ts_queries_img = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.img_queries_ts = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ts_norm = nn.LayerNorm(d_model)
        self.img_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))
        self.out = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, ts_tokens: torch.Tensor, img_tokens: torch.Tensor, no_gate: bool=False, gate_prior_mix: float=0.0, gate_prior: tuple[float, float]=(0.7, 0.3)) -> Dict[str, torch.Tensor]:
        ts_aligned, attn_ts_to_img = self.ts_queries_img(ts_tokens, img_tokens, img_tokens, need_weights=True)
        img_aligned, attn_img_to_ts = self.img_queries_ts(img_tokens, ts_tokens, ts_tokens, need_weights=True)
        ts_aligned = self.ts_norm(ts_tokens + ts_aligned).mean(dim=1)
        img_aligned = self.img_norm(img_tokens + img_aligned).mean(dim=1)
        gate_in = torch.cat([ts_aligned, img_aligned, torch.abs(ts_aligned - img_aligned), ts_aligned * img_aligned], dim=-1)
        gate = F.softmax(self.gate(gate_in), dim=-1)
        if no_gate:
            gate = torch.full_like(gate, 0.5)
        elif gate_prior_mix > 0:
            prior = gate.new_tensor(gate_prior).view(1, 2)
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-06)
            rho = float(np.clip(gate_prior_mix, 0.0, 1.0))
            gate = (1.0 - rho) * gate + rho * prior
        fused = torch.cat([gate[:, :1] * ts_aligned, gate[:, 1:] * img_aligned], dim=-1)
        return {'fused': self.out(fused), 'gate': gate, 'attn_ts_to_img': attn_ts_to_img, 'attn_img_to_ts': attn_img_to_ts}
