from __future__ import annotations
import torch
import torch.nn as nn
from pmv_crn.models.common import _group_count

class PatchTransformerImageEncoder(nn.Module):

    def __init__(self, in_channels: int=9, d_model: int=96, dropout: float=0.15, n_heads: int=4):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=4, stride=4, bias=False), nn.GroupNorm(_group_count(d_model), d_model), nn.GELU())
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.stat_proj = nn.Sequential(nn.Linear(in_channels * 5, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.fuse_proj = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def _image_stats(self, x_img: torch.Tensor) -> torch.Tensor:
        flat = x_img.flatten(2)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1, unbiased=False)
        maxv = flat.max(dim=-1).values
        minv = flat.min(dim=-1).values
        high = (flat > 0.7).float().mean(dim=-1)
        return torch.cat([mean, std, maxv, maxv - minv, high], dim=-1)

    def forward(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_img = x_img.float()
        feat = self.patch(x_img)
        tokens = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x_img.size(0), -1, -1)
        tokens = self.encoder(torch.cat([cls, tokens], dim=1))
        stat_emb = self.stat_proj(self._image_stats(x_img))
        pooled = self.fuse_proj(torch.cat([tokens[:, 0], stat_emb], dim=-1))
        view_weights = x_img.new_full((x_img.size(0), 3), 1.0 / 3.0)
        return (self.norm(tokens[:, 1:]), self.norm(pooled), view_weights, stat_emb)

class InceptionImageEncoder(nn.Module):

    def __init__(self, in_channels: int=9, d_model: int=96, dropout: float=0.15):
        super().__init__()
        c = max(24, d_model // 3)
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, c, kernel_size=k, padding=k // 2, bias=False), nn.GroupNorm(_group_count(c), c), nn.GELU(), nn.Dropout2d(dropout * 0.5)) for k in (3, 5, 9)])
        self.pool_branch = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_channels, c, kernel_size=1, bias=False), nn.GroupNorm(_group_count(c), c), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(4 * c, d_model, kernel_size=1, bias=False), nn.GroupNorm(_group_count(d_model), d_model), nn.GELU(), nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False), nn.GroupNorm(_group_count(d_model), d_model), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=False), nn.GroupNorm(_group_count(d_model), d_model), nn.GELU())
        self.stat_proj = nn.Sequential(nn.Linear(in_channels * 5, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.fuse_proj = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def _image_stats(self, x_img: torch.Tensor) -> torch.Tensor:
        flat = x_img.flatten(2)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1, unbiased=False)
        maxv = flat.max(dim=-1).values
        minv = flat.min(dim=-1).values
        high = (flat > 0.7).float().mean(dim=-1)
        return torch.cat([mean, std, maxv, maxv - minv, high], dim=-1)

    def forward(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_img = x_img.float()
        multi = torch.cat([branch(x_img) for branch in self.branches] + [self.pool_branch(x_img)], dim=1)
        feat = self.fuse(multi)
        tokens = feat.flatten(2).transpose(1, 2)
        cnn_pooled = tokens.mean(dim=1)
        stat_emb = self.stat_proj(self._image_stats(x_img))
        pooled = self.fuse_proj(torch.cat([cnn_pooled, stat_emb], dim=-1))
        view_weights = x_img.new_full((x_img.size(0), 3), 1.0 / 3.0)
        return (self.norm(tokens), self.norm(pooled), view_weights, stat_emb)
