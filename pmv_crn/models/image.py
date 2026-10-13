from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pmv_crn.models.common import _group_count

class _ResidualBlock2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, dropout: float=0.0):
        super().__init__()
        groups1 = _group_count(in_channels)
        groups2 = _group_count(out_channels)
        self.block = nn.Sequential(nn.GroupNorm(groups1, in_channels), nn.GELU(), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), nn.GroupNorm(groups2, out_channels), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)

class _SqueezeExcite2d(nn.Module):

    def __init__(self, channels: int, reduction: int=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, kernel_size=1), nn.GELU(), nn.Conv2d(hidden, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)

class _MultiScaleResidualBlock2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, dropout: float=0.0):
        super().__init__()
        branch_channels = max(8, out_channels // 4)
        groups_in = max(1, min(8, in_channels))
        self.pre = nn.Sequential(nn.GroupNorm(groups_in, in_channels), nn.GELU())
        self.local_3x3 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=1, bias=False), nn.GroupNorm(_group_count(branch_channels), branch_channels), nn.GELU())
        self.regional_5x5 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=5, stride=stride, padding=2, bias=False), nn.GroupNorm(_group_count(branch_channels), branch_channels), nn.GELU())
        self.dilated_band = nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False), nn.GroupNorm(_group_count(branch_channels), branch_channels), nn.GELU())
        self.context_pool = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=stride, padding=1), nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False), nn.GroupNorm(_group_count(branch_channels), branch_channels), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False), nn.GroupNorm(_group_count(out_channels), out_channels), nn.GELU(), nn.Dropout2d(dropout), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.se = _SqueezeExcite2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        multi_scale = torch.cat([self.local_3x3(z), self.regional_5x5(z), self.dilated_band(z), self.context_pool(z)], dim=1)
        return self.se(self.skip(x) + self.fuse(multi_scale))

class _ImageViewCNN(nn.Module):

    def __init__(self, in_channels: int, d_model: int, dropout: float):
        super().__init__()
        c1 = max(24, d_model // 3)
        c2 = max(48, d_model // 2)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False), nn.GroupNorm(_group_count(c1), c1), nn.GELU())
        self.backbone = nn.Sequential(_MultiScaleResidualBlock2d(c1, c1, stride=1, dropout=dropout * 0.5), _MultiScaleResidualBlock2d(c1, c2, stride=2, dropout=dropout), _MultiScaleResidualBlock2d(c2, c2, stride=1, dropout=dropout), _MultiScaleResidualBlock2d(c2, d_model, stride=2, dropout=dropout), _MultiScaleResidualBlock2d(d_model, d_model, stride=1, dropout=dropout), nn.GroupNorm(_group_count(d_model), d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(self.stem(x.float()))
        tokens = feat.flatten(2).transpose(1, 2)
        pooled = tokens.mean(dim=1)
        return (tokens, pooled)

class PrecursorImageEncoder(nn.Module):

    def __init__(self, in_channels: int=9, d_model: int=96, dropout: float=0.15):
        super().__init__()
        self.in_channels = int(in_channels)
        if in_channels >= 12:
            view_indices = [[0, 1, 2, 4], [3, 8], [5, 6, 7, 9, 10, 11]]
        elif in_channels >= 9:
            view_indices = [[0, 1, 2, 4], [3, 8], [5, 6, 7]]
        else:
            splits = np.array_split(np.arange(in_channels), 3)
            view_indices = [part.tolist() for part in splits if len(part) > 0]
        self.register_buffer('_dummy', torch.empty(0), persistent=False)
        self.view_indices = [torch.tensor(idx, dtype=torch.long) for idx in view_indices]
        self.view_encoders = nn.ModuleList([_ImageViewCNN(len(idx), d_model, dropout=dropout) for idx in view_indices])
        self.view_attention = nn.Sequential(nn.Linear(d_model, max(16, d_model // 2)), nn.GELU(), nn.Dropout(dropout), nn.Linear(max(16, d_model // 2), 1))
        self.stat_proj = nn.Sequential(nn.Linear(self.in_channels * 5, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.fuse_proj = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.token_proj = nn.Linear(d_model, d_model)
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
        view_tokens = []
        view_embs = []
        for indices, encoder in zip(self.view_indices, self.view_encoders):
            idx = indices.to(device=x_img.device)
            tokens, pooled = encoder(x_img.index_select(1, idx))
            view_tokens.append(tokens)
            view_embs.append(pooled)
        emb_stack = torch.stack(view_embs, dim=1)
        learned_weights = F.softmax(self.view_attention(emb_stack).squeeze(-1), dim=-1)
        uniform_weights = torch.full_like(learned_weights, 1.0 / learned_weights.size(-1))
        view_weights = 0.5 * learned_weights + 0.5 * uniform_weights
        cnn_pooled = (view_weights.unsqueeze(-1) * emb_stack).sum(dim=1)
        stat_emb = self.stat_proj(self._image_stats(x_img))
        pooled = self.fuse_proj(torch.cat([cnn_pooled, stat_emb], dim=-1))
        weighted_tokens = [tokens * view_weights[:, i].view(-1, 1, 1) for i, tokens in enumerate(view_tokens)]
        tokens = torch.cat(weighted_tokens, dim=1)
        stat_token = stat_emb.unsqueeze(1)
        tokens = torch.cat([tokens, stat_token], dim=1)
        tokens = self.norm(self.token_proj(tokens))
        pooled = self.norm(pooled)
        return (tokens, pooled, view_weights, stat_emb)

class SensorGroupedImageEncoder(PrecursorImageEncoder):

    def __init__(self, in_channels: int=9, d_model: int=96, dropout: float=0.15):
        nn.Module.__init__(self)
        self.in_channels = int(in_channels)
        if in_channels >= 9:
            view_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        else:
            splits = np.array_split(np.arange(in_channels), 3)
            view_indices = [part.tolist() for part in splits if len(part) > 0]
        self.register_buffer('_dummy', torch.empty(0), persistent=False)
        self.view_indices = [torch.tensor(idx, dtype=torch.long) for idx in view_indices]
        self.view_encoders = nn.ModuleList([_ImageViewCNN(len(idx), d_model, dropout=dropout) for idx in view_indices])
        self.view_attention = nn.Sequential(nn.Linear(d_model, max(16, d_model // 2)), nn.GELU(), nn.Dropout(dropout), nn.Linear(max(16, d_model // 2), 1))
        self.stat_proj = nn.Sequential(nn.Linear(self.in_channels * 5, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.fuse_proj = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.token_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
