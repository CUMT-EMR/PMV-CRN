from __future__ import annotations
import torch
import torch.nn as nn

class MultiScaleTemporalEncoder(nn.Module):

    def __init__(self, in_channels: int=3, d_model: int=96, kernels: tuple[int, ...]=(3, 5, 9), gru_layers: int=1, dropout: float=0.15):
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even for bidirectional GRU.'
        branch_dim = max(8, d_model // len(kernels))
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels, branch_dim, kernel_size=k, padding=k // 2, bias=False), nn.BatchNorm1d(branch_dim), nn.GELU(), nn.Dropout(dropout)) for k in kernels])
        self.proj = nn.Linear(branch_dim * len(kernels), d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model // 2, num_layers=gru_layers, batch_first=True, bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_ts.float().transpose(1, 2)
        feats = [branch(x) for branch in self.branches]
        tokens = torch.cat(feats, dim=1).transpose(1, 2)
        tokens = self.dropout(self.proj(tokens))
        tokens, _ = self.gru(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return (tokens, pooled)
