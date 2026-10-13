from __future__ import annotations
import torch
import torch.nn as nn

class BPNN(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 96), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(96, 2))

    def forward(self, x):
        return self.head(self.encode(x))

    def encode(self, x):
        return self.features(x)

class CNN1D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 48, 5, padding=2), nn.BatchNorm1d(48), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(48, 96, 5, padding=2), nn.BatchNorm1d(96), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 2))

    def forward(self, x):
        z = x.transpose(1, 2)
        return self.head(self.encode(x))

    def encode(self, x):
        z = x.transpose(1, 2)
        return torch.flatten(self.net(z), 1)

class RNNClassifier(nn.Module):

    def __init__(self, channels: int, kind: str):
        super().__init__()
        rnn_cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = rnn_cls(channels, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.encode(x))

    def encode(self, x):
        out, _ = self.rnn(x)
        return out.mean(dim=1)

class TransformerClassifier(nn.Module):

    def __init__(self, channels: int, medformer_like: bool=False):
        super().__init__()
        d_model = 96
        self.patch = 4 if medformer_like else 1
        self.proj = nn.Linear(channels * self.patch, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=192, dropout=0.15, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2), nn.Linear(d_model, 2))

    def forward(self, x):
        return self.head(self.encode(x))

    def encode(self, x):
        if self.patch > 1:
            b, t, c = x.shape
            trim = t // self.patch * self.patch
            x = x[:, :trim].reshape(b, trim // self.patch, c * self.patch)
        z = self.proj(x)
        z = self.enc(z)
        return z.mean(dim=1)
