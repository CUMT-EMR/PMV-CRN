from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pmv_crn.data.features import window_stat_features
from pmv_crn.data.representations import make_image_concept_prior, make_precursor_image_maps, make_temporal_concept_prior, standardize_channels

class RecursivePointDataset(Dataset):

    def __init__(self, df: pd.DataFrame, indices: Sequence[int], labels: np.ndarray, window: int, img_size: int, image_map_type: str, global_mean: np.ndarray, global_std: np.ndarray, stat_mean: np.ndarray, stat_std: np.ndarray, use_absolute_ts: bool=True, cache_in_memory: bool=True):
        self.df = df.reset_index(drop=True)
        self.values = self.df[['AE', 'EMR', 'GAS']].to_numpy(dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.raw_labels = self.df['LABEL'].to_numpy(dtype=np.int64)
        self.dates = self.df['date'].astype(str).to_numpy() if 'date' in self.df.columns else np.arange(len(self.df)).astype(str)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = int(window)
        self.img_size = int(img_size)
        self.image_map_type = str(image_map_type)
        self.global_mean = np.asarray(global_mean, dtype=np.float32)
        self.global_std = np.asarray(global_std, dtype=np.float32)
        self.stat_mean = np.asarray(stat_mean, dtype=np.float32)
        self.stat_std = np.asarray(stat_std, dtype=np.float32)
        self.use_absolute_ts = bool(use_absolute_ts)
        self.cache_in_memory = bool(cache_in_memory)
        self.cache: list[Dict[str, object]] | None = None
        if self.cache_in_memory:
            self.cache = [self._make_item(int(point_idx)) for point_idx in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def _make_item(self, point_idx: int) -> Dict[str, object]:
        raw_window = self.values[point_idx - self.window + 1:point_idx + 1]
        x_local = standardize_channels(raw_window).astype(np.float32)
        raw_z = ((raw_window - self.global_mean) / (self.global_std + 1e-06)).astype(np.float32)
        raw_z = np.clip(raw_z, -8.0, 8.0)
        x_ts = np.concatenate([x_local, raw_z], axis=1).astype(np.float32) if self.use_absolute_ts else x_local
        x_img = make_precursor_image_maps(x_local, self.img_size, raw_z, map_type=self.image_map_type)
        stat = window_stat_features(raw_window, self.global_mean, self.global_std)
        stat = ((stat - self.stat_mean) / (self.stat_std + 1e-06)).astype(np.float32)
        item = {'x_ts': torch.from_numpy(x_ts), 'x_img': torch.from_numpy(x_img), 'ts_concepts': torch.from_numpy(make_temporal_concept_prior(x_local)), 'img_concepts': torch.from_numpy(make_image_concept_prior(x_img, self.image_map_type)), 'stat_features': torch.from_numpy(stat), 'y': torch.tensor(int(self.labels[point_idx]), dtype=torch.long), 'raw_y': torch.tensor(int(self.raw_labels[point_idx]), dtype=torch.long), 'point_index': point_idx, 'date': self.dates[point_idx], 'AE': float(self.values[point_idx, 0]), 'EMR': float(self.values[point_idx, 1]), 'GAS': float(self.values[point_idx, 2])}
        return item

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.cache is not None:
            return self.cache[idx]
        return self._make_item(int(self.indices[idx]))

def collate_points(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out = {'x_ts': torch.stack([item['x_ts'] for item in batch]), 'x_img': torch.stack([item['x_img'] for item in batch]), 'ts_concepts': torch.stack([item['ts_concepts'] for item in batch]), 'img_concepts': torch.stack([item['img_concepts'] for item in batch]), 'stat_features': torch.stack([item['stat_features'] for item in batch]), 'y': torch.stack([item['y'] for item in batch]), 'raw_y': torch.stack([item['raw_y'] for item in batch]), 'point_index': [item['point_index'] for item in batch], 'date': [item['date'] for item in batch], 'AE': [item['AE'] for item in batch], 'EMR': [item['EMR'] for item in batch], 'GAS': [item['GAS'] for item in batch]}
    return out
