from __future__ import annotations
from typing import Optional
import numpy as np

def standardize_channels(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / (std + 1e-06)).astype(np.float32)

def resample_1d(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(length, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)

def minmax_1d(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-06:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def _moving_average(x: np.ndarray, k: int=7) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(k, dtype=np.float32) / k
    out = np.stack([np.convolve(xp[:, c], kernel, mode='valid') for c in range(x.shape[1])], axis=1)
    return out.astype(np.float32)

def _similarity_matrix(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=-1))
    sim = np.exp(-dist / (np.std(dist) + 1e-06))
    return sim.astype(np.float32)

def _gaf_sum_diff(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(minmax_1d(signal) * 2.0 - 1.0, -1.0, 1.0)
    phi = np.arccos(x)
    gasf = np.cos(phi[:, None] + phi[None, :])
    gadf = np.sin(phi[:, None] - phi[None, :])
    return (((gasf + 1.0) * 0.5).astype(np.float32), ((gadf + 1.0) * 0.5).astype(np.float32))

def _markov_transition_field(signal: np.ndarray, bins: int=8) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    quantiles = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    quantiles[0] -= 1e-06
    quantiles[-1] += 1e-06
    states = np.digitize(x, quantiles[1:-1], right=False)
    trans = np.ones((bins, bins), dtype=np.float32) * 0.001
    for left, right in zip(states[:-1], states[1:]):
        trans[int(left), int(right)] += 1.0
    trans = trans / trans.sum(axis=1, keepdims=True).clip(min=1e-06)
    return trans[states[:, None], states[None, :]].astype(np.float32)

def _trajectory_raster(signal: np.ndarray, size: int) -> np.ndarray:
    x = minmax_1d(signal)
    rows = np.round((1.0 - x) * (size - 1)).astype(np.float32)
    rr = np.arange(size, dtype=np.float32)[:, None]
    sigma = max(1.0, size * 0.035)
    img = np.exp(-(rr - rows[None, :]) ** 2 / (2.0 * sigma ** 2))
    return img.astype(np.float32)

def _resize_2d(img: np.ndarray, size: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.shape == (size, size):
        return img.astype(np.float32)
    y_old = np.linspace(0.0, 1.0, img.shape[0], dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, img.shape[1], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y_new = np.linspace(0.0, 1.0, size, dtype=np.float32)
    rows = np.stack([np.interp(x_new, x_old, row) for row in img], axis=0)
    cols = np.stack([np.interp(y_new, y_old, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    return cols.astype(np.float32)

def _stft_texture(signal: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    win = max(8, min(32, len(x) // 4))
    hop = max(2, win // 4)
    window = np.hanning(win).astype(np.float32)
    specs = []
    for start in range(0, max(1, len(x) - win + 1), hop):
        seg = x[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)), mode='edge')
        mag = np.abs(np.fft.rfft(seg * window))
        specs.append(mag)
    spec = np.stack(specs, axis=1) if specs else np.zeros((win // 2 + 1, 1), dtype=np.float32)
    spec = np.log1p(spec)
    return _resize_2d(minmax_1d(spec.ravel()).reshape(spec.shape), size)

def _cwt_texture(signal: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    scales = np.linspace(1.0, max(2.0, len(x) / 8.0), num=max(8, size // 2), dtype=np.float32)
    t = np.arange(-len(x) // 2, len(x) // 2, dtype=np.float32)
    coeffs = []
    for scale in scales:
        u = t / scale
        wavelet = (1.0 - u * u) * np.exp(-0.5 * u * u)
        wavelet = wavelet - wavelet.mean()
        norm = np.sqrt(np.sum(wavelet * wavelet)) + 1e-06
        coeffs.append(np.abs(np.convolve(x, wavelet / norm, mode='same')))
    cwt = np.stack(coeffs, axis=0)
    cwt = np.log1p(cwt)
    return _resize_2d(minmax_1d(cwt.ravel()).reshape(cwt.shape), size)

def make_precursor_image_maps(x_ts_local: np.ndarray, img_size: int=32, x_ts_abs: Optional[np.ndarray]=None, map_type: str='precursor9') -> np.ndarray:
    small = np.stack([resample_1d(x_ts_local[:, i], img_size) for i in range(x_ts_local.shape[1])], axis=1)
    small = standardize_channels(small)
    if x_ts_abs is None:
        abs_small = small
    else:
        abs_small = np.stack([resample_1d(x_ts_abs[:, i], img_size) for i in range(x_ts_abs.shape[1])], axis=1)
        abs_small = np.clip(abs_small, -8.0, 8.0).astype(np.float32)
    rec = _similarity_matrix(small)
    grad = np.gradient(small, axis=0).astype(np.float32)
    trend = _similarity_matrix(grad)
    energy_feat = _moving_average(small ** 2, k=max(3, img_size // 16 * 2 + 1))
    energy = _similarity_matrix(energy_feat)
    ae = minmax_1d(small[:, 0])
    emr = minmax_1d(small[:, 1])
    gas = minmax_1d(small[:, 2])
    ii, jj = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')
    delay = jj - ii
    lag0 = max(2.0, img_size * 0.12)
    sigma = max(2.0, img_size * 0.08)
    delay_mask = np.exp(-(delay - lag0) ** 2 / (2 * sigma ** 2)) * (delay > 0)
    ae_gas = 1.0 - np.abs(ae[:, None] - gas[None, :])
    emr_gas = 1.0 - np.abs(emr[:, None] - gas[None, :])
    lag = 0.5 * (ae_gas + emr_gas) * delay_mask
    lag = lag / (lag.max() + 1e-06)
    abs_level = _similarity_matrix(abs_small)
    abs_ae = minmax_1d(abs_small[:, 0])
    abs_emr = minmax_1d(abs_small[:, 1])
    abs_gas = minmax_1d(abs_small[:, 2])
    coupling_vec = np.stack([1.0 - np.abs(abs_ae - abs_emr), 1.0 - np.abs(abs_ae - abs_gas), 1.0 - np.abs(abs_emr - abs_gas)], axis=1).mean(axis=1)
    coupling = _similarity_matrix(coupling_vec[:, None])
    boundary_1d = np.abs(np.gradient(rec, axis=0)) + np.abs(np.gradient(trend, axis=1))
    boundary = minmax_1d(boundary_1d.ravel()).reshape(img_size, img_size)
    local_energy = minmax_1d(energy.ravel()).reshape(img_size, img_size)
    contrast = np.abs(local_energy - np.median(local_energy)).astype(np.float32)
    contrast = contrast / (contrast.max() + 1e-06)
    cross_source = np.maximum(abs_ae[:, None], abs_emr[:, None]) * abs_gas[None, :] * delay_mask
    cross_source = cross_source / (cross_source.max() + 1e-06)
    map_type = str(map_type).lower()
    if map_type == 'sensor_gaf_mtf9':
        maps = []
        for c in range(3):
            gasf_c, gadf_c = _gaf_sum_diff(abs_small[:, c])
            mtf_c = _markov_transition_field(abs_small[:, c], bins=8)
            maps.extend([gasf_c, gadf_c, mtf_c])
        return np.clip(np.stack(maps, axis=0), 0.0, 1.0).astype(np.float32)
    if map_type == 'sensor_trajectory9':
        maps = []
        for c in range(3):
            level = minmax_1d(abs_small[:, c])
            diff = minmax_1d(np.gradient(abs_small[:, c]).astype(np.float32))
            energy_1d = minmax_1d(_moving_average(abs_small[:, c:c + 1] ** 2, k=max(3, img_size // 16 * 2 + 1))[:, 0])
            maps.extend([_trajectory_raster(level, img_size), _trajectory_raster(diff, img_size), _trajectory_raster(energy_1d, img_size)])
        return np.clip(np.stack(maps, axis=0), 0.0, 1.0).astype(np.float32)
    if map_type in {'texture9', 'obvious9', 'gaf_mtf9', 'hybrid9'}:
        fused = standardize_channels(abs_small).mean(axis=1)
        fused_local = standardize_channels(small).mean(axis=1)
        grad_fused = np.gradient(fused_local).astype(np.float32)
        gasf, gadf = _gaf_sum_diff(fused)
        _, trend_gadf = _gaf_sum_diff(grad_fused)
        mtf = _markov_transition_field(fused, bins=8)
        cwt = _cwt_texture(fused, img_size)
        stft = _stft_texture(fused, img_size)
        multi_rec = np.maximum(rec, abs_level)
        if map_type == 'hybrid9':
            source_level = np.maximum(abs_ae, abs_emr)
            source_traj = _trajectory_raster(source_level, img_size)
            gas_traj = _trajectory_raster(abs_gas, img_size)
            maps9 = np.stack([rec, multi_rec, trend_gadf, lag, gasf, gadf, mtf, source_traj, gas_traj], axis=0)
            return np.clip(maps9, 0.0, 1.0).astype(np.float32)
        maps9 = np.stack([rec, multi_rec, trend_gadf, lag, gasf, gadf, mtf, cwt, stft], axis=0)
        return np.clip(maps9, 0.0, 1.0).astype(np.float32)
    maps9 = np.stack([rec, trend, energy, lag, abs_level, coupling, boundary, contrast, cross_source], axis=0)
    if map_type == 'precursor4':
        maps = maps9[:4]
    elif map_type == 'precursor6':
        maps = maps9[:6]
    else:
        maps = maps9
    return np.clip(maps, 0.0, 1.0).astype(np.float32)

def _safe_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or np.std(x) < 1e-08 or np.std(y) < 1e-08:
        return 0.0
    val = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(val) else val

def _cusum_np(x: np.ndarray, k: float=0.15) -> np.ndarray:
    pos = np.zeros_like(x, dtype=np.float32)
    neg = np.zeros_like(x, dtype=np.float32)
    for i in range(1, len(x)):
        pos[i] = max(0.0, pos[i - 1] + x[i] - k)
        neg[i] = min(0.0, neg[i - 1] + x[i] + k)
    return pos + neg

def _sigmoid_np(x: float, scale: float=1.0) -> float:
    return float(1.0 / (1.0 + np.exp(-x / max(scale, 1e-06))))

def make_temporal_concept_prior(x_ts_local: np.ndarray) -> np.ndarray:
    early = x_ts_local[:max(8, len(x_ts_local) // 4)]
    tail = x_ts_local[-max(8, len(x_ts_local) // 4):]
    fused = x_ts_local.mean(axis=1)
    diff = np.diff(x_ts_local, axis=0)
    tail_level = float(np.mean(np.abs(tail)))
    level_gap = float(np.mean(tail) - np.mean(early))
    vol_ratio = float(np.std(tail) / (np.std(x_ts_local) + 1e-06))
    burst_density = float((np.abs(x_ts_local) > 1.8).mean())
    cusum_area = float(np.mean(np.abs(_cusum_np(fused))))
    complexity = float(np.std(diff) + np.mean(np.abs(diff)))
    corr_vals = [abs(_safe_corr_np(x_ts_local[:, a], x_ts_local[:, b])) for a, b in ((0, 1), (0, 2), (1, 2))]
    coupling = float(np.mean(corr_vals))
    risk_acc = float(np.mean(np.maximum(fused - np.quantile(fused, 0.75), 0.0)))
    lag_scores = []
    for src in (0, 1):
        for lag in (4, 8, 12, 16):
            if len(x_ts_local) > lag + 8:
                lag_scores.append(_safe_corr_np(x_ts_local[:-lag, src], x_ts_local[lag:, 2]))
    lead_lag = float(max(lag_scores) if lag_scores else 0.0)
    concepts = np.array([_sigmoid_np(tail_level + abs(level_gap), 1.2), _sigmoid_np(vol_ratio - 1.0, 0.55), np.clip(burst_density * 3.0, 0.0, 1.0), _sigmoid_np(cusum_area - 2.5, 2.0), _sigmoid_np(complexity - 0.8, 0.6), np.clip(coupling, 0.0, 1.0), _sigmoid_np(risk_acc - 0.12, 0.25), _sigmoid_np(lead_lag, 0.45)], dtype=np.float32)
    return np.clip(concepts, 0.0, 1.0)

def _entropy01(x: np.ndarray, bins: int=16) -> float:
    hist, _ = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float32) / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))

def make_image_concept_prior(x_img: np.ndarray, map_type: str='precursor9') -> np.ndarray:
    map_type = str(map_type).lower()
    if map_type == 'sensor_trajectory9' and x_img.shape[0] >= 9:
        ae_level, ae_diff, ae_energy = (x_img[0], x_img[1], x_img[2])
        emr_level, emr_diff, emr_energy = (x_img[3], x_img[4], x_img[5])
        gas_level, gas_diff, gas_energy = (x_img[6], x_img[7], x_img[8])
        level_maps = [ae_level, emr_level, gas_level]
        diff_maps = [ae_diff, emr_diff, gas_diff]
        energy_maps = [ae_energy, emr_energy, gas_energy]
        source_level = np.maximum(ae_level, emr_level)
        source_energy = np.maximum(ae_energy, emr_energy)
        cross_source = source_level * gas_level
        concepts = np.array([np.clip(float(np.mean([m.max() for m in level_maps])), 0.0, 1.0), np.clip(float(np.mean([m.mean() for m in diff_maps]) * 1.6), 0.0, 1.0), np.clip(float(np.mean([m.mean() for m in energy_maps]) * 1.8), 0.0, 1.0), np.clip(float(cross_source.mean() * 2.0), 0.0, 1.0), np.clip(float((ae_level * emr_level).mean() * 2.0), 0.0, 1.0), np.clip(float(gas_level.mean() * 1.8), 0.0, 1.0), _sigmoid_np(float(np.mean([m.std() for m in diff_maps])) - 0.12, 0.08), _sigmoid_np(float(source_energy.mean() + gas_energy.mean()) - 0.3, 0.12)], dtype=np.float32)
        return np.clip(concepts, 0.0, 1.0)
    rec, trend, energy, lag = (x_img[0], x_img[1], x_img[2], x_img[3])
    abs_level = x_img[4] if x_img.shape[0] > 4 else rec
    coupling = x_img[5] if x_img.shape[0] > 5 else rec
    boundary_map = x_img[6] if x_img.shape[0] > 6 else trend
    contrast_map = x_img[7] if x_img.shape[0] > 7 else energy
    cross_source = x_img[8] if x_img.shape[0] > 8 else lag
    high = [(m > 0.75).mean() for m in (rec, trend, energy, lag)]
    boundary = float(np.mean(boundary_map) + np.mean(np.abs(np.diff(rec, axis=0))) + np.mean(np.abs(np.diff(trend, axis=1))))
    local_cluster = float(np.percentile(rec, 90) - np.percentile(rec, 50) + np.percentile(energy, 90) - np.percentile(energy, 50) + np.percentile(cross_source, 90) - np.percentile(cross_source, 50))
    all_maps = np.concatenate([m.ravel() for m in (rec, trend, energy, lag, abs_level, coupling, boundary_map, contrast_map, cross_source)])
    contrast = float(np.percentile(all_maps, 90) - np.percentile(all_maps, 10))
    concepts = np.array([np.clip(high[0] * 1.6, 0.0, 1.0), np.clip(high[1] * 1.6, 0.0, 1.0), np.clip(high[2] * 1.6, 0.0, 1.0), np.clip(max(high[3] * 2.5, (cross_source > 0.5).mean() * 2.0), 0.0, 1.0), np.clip(float(coupling.mean()), 0.0, 1.0), np.clip(float((abs_level > 0.75).mean() * 1.8), 0.0, 1.0), _sigmoid_np(boundary - 0.08, 0.05), _sigmoid_np(local_cluster - 0.35, 0.15)], dtype=np.float32)
    return np.clip(concepts, 0.0, 1.0)
