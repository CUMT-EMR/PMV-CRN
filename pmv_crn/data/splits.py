from __future__ import annotations
import numpy as np
import pandas as pd

def label_segments(labels: np.ndarray, start_index: int, max_segment_len: int=0) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    n = len(labels)
    s = int(start_index)
    while s < n:
        lab = int(labels[s])
        e = s
        while e + 1 < n and int(labels[e + 1]) == lab:
            e += 1
        if max_segment_len and max_segment_len > 0 and (lab == 0):
            cs = s
            while cs <= e:
                ce = min(e, cs + max_segment_len - 1)
                segments.append((cs, ce, lab))
                cs = ce + 1
        else:
            segments.append((s, e, lab))
        s = e + 1
    return segments

def allocate_segments_by_class(segments: list[tuple[int, int, int]], train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    split_parts: list[list[np.ndarray]] = [[], [], []]
    ratios = np.asarray([train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio)], dtype=np.float64)
    ratios = ratios / ratios.sum()
    for lab in [1, 0]:
        class_segments = [seg for seg in segments if seg[2] == lab]
        rng.shuffle(class_segments)
        total = float(sum((e - s + 1 for s, e, _ in class_segments)))
        targets = total * ratios
        counts = np.zeros(3, dtype=np.float64)
        assigned: list[list[tuple[int, int, int]]] = [[], [], []]
        if len(class_segments) >= 3:
            for split_id in [0, 1, 2]:
                seg = class_segments.pop(0)
                assigned[split_id].append(seg)
                counts[split_id] += seg[1] - seg[0] + 1
        for seg in class_segments:
            split_id = int(np.argmax(targets - counts))
            assigned[split_id].append(seg)
            counts[split_id] += seg[1] - seg[0] + 1
        for split_id, segs in enumerate(assigned):
            for s, e, _ in segs:
                split_parts[split_id].append(np.arange(s, e + 1, dtype=np.int64))
    out = []
    for parts in split_parts:
        out.append(np.sort(np.concatenate(parts)) if parts else np.asarray([], dtype=np.int64))
    return (out[0], out[1], out[2])

def purge_near_splits(indices: tuple[np.ndarray, np.ndarray, np.ndarray], purge: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if purge <= 0:
        return indices
    train_idx, val_idx, test_idx = indices

    def _purge(source: np.ndarray, blockers: np.ndarray) -> np.ndarray:
        if len(source) == 0 or len(blockers) == 0:
            return source
        keep = np.ones(len(source), dtype=bool)
        blockers = np.sort(blockers)
        for i, idx in enumerate(source):
            pos = np.searchsorted(blockers, idx)
            left_ok = pos == 0 or abs(idx - blockers[pos - 1]) > purge
            right_ok = pos == len(blockers) or abs(idx - blockers[pos]) > purge
            keep[i] = left_ok and right_ok
        return source[keep]
    train_idx = _purge(train_idx, np.concatenate([val_idx, test_idx]))
    return (train_idx, val_idx, test_idx)

def split_indices(labels: np.ndarray, window: int, train_ratio: float, val_ratio: float, seed: int, max_normal_segment_len: int, purge_windows: int, split_mode: str='event', block_size: int=1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if split_mode == 'random_stratified':
        return random_stratified_split(labels, window, train_ratio, val_ratio, seed)
    if split_mode == 'blocked_random_stratified':
        splits = blocked_random_stratified_split(labels, window, train_ratio, val_ratio, seed, block_size)
        return purge_near_splits(splits, purge=max(0, purge_windows * window))
    segments = label_segments(labels, window - 1, max_segment_len=max_normal_segment_len)
    splits = allocate_segments_by_class(segments, train_ratio, val_ratio, seed)
    return purge_near_splits(splits, purge=max(0, purge_windows * window))

def split_leakage_summary(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, window: int) -> pd.DataFrame:
    rows = []
    train_sorted = np.sort(np.asarray(train_idx, dtype=np.int64))

    def _nearest_distance(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.int64)
        if len(query) == 0 or len(train_sorted) == 0:
            return np.asarray([], dtype=np.float32)
        pos = np.searchsorted(train_sorted, query)
        left = np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
        right = np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
        has_left = pos > 0
        has_right = pos < len(train_sorted)
        left[has_left] = np.abs(query[has_left] - train_sorted[pos[has_left] - 1])
        right[has_right] = np.abs(query[has_right] - train_sorted[pos[has_right]])
        return np.minimum(left, right).astype(np.float32)
    for split, idx in [('val', val_idx), ('test', test_idx)]:
        dist = _nearest_distance(idx)
        if len(dist) == 0:
            rows.append({'split': split, 'points': 0, 'min_train_distance': np.nan})
            continue
        rows.append({'split': split, 'points': int(len(idx)), 'min_train_distance': float(dist.min()), 'median_train_distance': float(np.median(dist)), 'share_within_1_step': float((dist <= 1).mean()), 'share_within_window': float((dist < window).mean()), 'mean_window_overlap_ratio': float(np.maximum(0.0, 1.0 - dist / max(1, window)).mean())})
    return pd.DataFrame(rows)

def random_stratified_split(labels: np.ndarray, window: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_idx = np.arange(window - 1, len(labels), dtype=np.int64)
    split_parts: list[list[np.ndarray]] = [[], [], []]
    ratios = np.asarray([train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio)], dtype=np.float64)
    ratios = ratios / ratios.sum()
    for lab in [0, 1]:
        idx = all_idx[labels[all_idx] == lab].copy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * ratios[0]))
        n_val = int(round(len(idx) * ratios[1]))
        train = idx[:n_train]
        val = idx[n_train:n_train + n_val]
        test = idx[n_train + n_val:]
        split_parts[0].append(train)
        split_parts[1].append(val)
        split_parts[2].append(test)
    out = []
    for parts in split_parts:
        merged = np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)
        rng.shuffle(merged)
        out.append(merged.astype(np.int64))
    return (out[0], out[1], out[2])

def blocked_random_stratified_split(labels: np.ndarray, window: int, train_ratio: float, val_ratio: float, seed: int, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    start_idx = window - 1
    block_size = max(int(block_size), int(window))
    blocks: list[tuple[int, int, int]] = []
    s = start_idx
    while s < len(labels):
        e = min(len(labels) - 1, s + block_size - 1)
        lab = int(labels[s:e + 1].max() > 0)
        blocks.append((s, e, lab))
        s = e + 1
    ratios = np.asarray([train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio)], dtype=np.float64)
    ratios = ratios / ratios.sum()
    split_blocks: list[list[tuple[int, int, int]]] = [[], [], []]

    def _block_weight(block: tuple[int, int, int]) -> float:
        s, e, lab = block
        if lab == 1:
            return float((labels[s:e + 1] == 1).sum())
        return float(e - s + 1)
    for lab in [1, 0]:
        class_blocks = [b for b in blocks if b[2] == lab]
        rng.shuffle(class_blocks)
        total = float(sum((_block_weight(block) for block in class_blocks)))
        targets = total * ratios
        counts = np.zeros(3, dtype=np.float64)
        if len(class_blocks) >= 3:
            for split_id in [0, 1, 2]:
                block = class_blocks.pop(0)
                split_blocks[split_id].append(block)
                counts[split_id] += _block_weight(block)
        for block in class_blocks:
            split_id = int(np.argmax(targets - counts))
            split_blocks[split_id].append(block)
            counts[split_id] += _block_weight(block)
    out = []
    for parts in split_blocks:
        idx_parts = [np.arange(s, e + 1, dtype=np.int64) for s, e, _ in parts]
        merged = np.concatenate(idx_parts) if idx_parts else np.asarray([], dtype=np.int64)
        rng.shuffle(merged)
        out.append(merged.astype(np.int64))
    return (out[0], out[1], out[2])
