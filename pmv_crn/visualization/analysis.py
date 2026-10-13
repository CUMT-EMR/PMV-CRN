from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
try:
    import umap
except ImportError:
    umap = None
from pmv_crn.models.common import IMAGE_CONCEPT_NAMES, TEMPORAL_CONCEPT_NAMES
from pmv_crn.models.network import OptimizedDualBranchCMRNet
from pmv_crn.training.batch import forward_model_batch

def plot_prediction(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, constrained_layout=True)
    x = np.arange(len(df))
    axes[0].plot(x, df['prob'], linewidth=0.8, alpha=0.45, label='risk probability')
    axes[0].plot(x, df['prob_smooth'], linewidth=1.3, label='smoothed probability')
    axes[0].plot(x, df['true_label'], linewidth=1.0, alpha=0.75, label='target label')
    axes[0].plot(x, df['raw_label'], linewidth=0.9, alpha=0.65, label='raw label')
    axes[0].axhline(float(df['threshold'].iloc[0]), linestyle='--', linewidth=1.0, label='threshold')
    axes[0].set_ylabel('Probability / label')
    axes[0].set_title(f'{split} recursive prediction')
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(x, df['AE'], linewidth=0.8, label='AE')
    axes[1].plot(x, df['EMR'], linewidth=0.8, label='EMR')
    axes[1].plot(x, df['GAS'], linewidth=0.8, label='GAS')
    axes[1].set_xlabel('Point step')
    axes[1].set_ylabel('Observed value')
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_dir / f'{split}_recursive_prediction.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def plot_training_curves(history: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / 'training_history_origin_data.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    loss_cols = [c for c in ['train_loss_loss', 'val_loss', 'test_loss'] if c in history.columns]
    for col in loss_cols:
        axes[0, 0].plot(history['epoch'], history[col], label=col.replace('_loss', ''))
    axes[0, 0].set_title('Loss')
    for metric, ax in [('accuracy', axes[0, 1]), ('f1', axes[1, 0]), ('auc', axes[1, 1])]:
        for split in ['train', 'val', 'test']:
            col = f'{split}_{metric}'
            if col in history.columns:
                ax.plot(history['epoch'], history[col], label=split)
        ax.axhline(0.9, color='#B24C43', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.02)
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(out_dir / 'training_validation_test_curves.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    metric_cols = ['train_accuracy', 'val_accuracy', 'test_accuracy', 'train_precision', 'val_precision', 'test_precision', 'train_recall', 'val_recall', 'test_recall', 'train_f1', 'val_f1', 'test_f1', 'train_auc', 'val_auc', 'test_auc', 'train_pr_auc', 'val_pr_auc', 'test_pr_auc']
    present = [c for c in metric_cols if c in history.columns]
    history[['epoch', *present]].to_csv(out_dir / 'metric_curves_origin_data.csv', index=False, encoding='utf-8-sig')

def _safe_corr_series(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or float(a.std()) < 1e-08 or float(b.std()) < 1e-08:
        return 0.0
    val = float(a.corr(b))
    return 0.0 if np.isnan(val) else val

def _top_rule_concept_rows(df: pd.DataFrame, split: str, branch: str, concept_names: Sequence[str]) -> list[Dict[str, object]]:
    rule_cols = sorted([c for c in df.columns if c.startswith(f'{branch}_rule_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
    concept_cols = sorted([c for c in df.columns if c.startswith(f'{branch}_concept_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
    rows: list[Dict[str, object]] = []
    for rule_col in rule_cols:
        rule_id = int(rule_col.rsplit('_', 1)[-1])
        corr_items = []
        for concept_col in concept_cols:
            concept_id = int(concept_col.rsplit('_', 1)[-1])
            concept_name = concept_names[concept_id] if concept_id < len(concept_names) else concept_col
            corr_items.append((concept_id, concept_name, _safe_corr_series(df[rule_col], df[concept_col])))
        corr_items.sort(key=lambda item: abs(item[2]), reverse=True)
        top = corr_items[:3]
        rows.append({'split': split, 'branch': 'temporal' if branch == 'ts' else 'image', 'rule_id': rule_id, 'rule_column': rule_col, 'mean_rule_weight': float(df[rule_col].mean()) if len(df) else float('nan'), 'positive_mean_rule_weight': float(df.loc[df['true_label'] == 1, rule_col].mean()) if (df['true_label'] == 1).any() else float('nan'), 'negative_mean_rule_weight': float(df.loc[df['true_label'] == 0, rule_col].mean()) if (df['true_label'] == 0).any() else float('nan'), 'top_concept_1': top[0][1] if len(top) > 0 else '', 'top_concept_1_corr': top[0][2] if len(top) > 0 else float('nan'), 'top_concept_2': top[1][1] if len(top) > 1 else '', 'top_concept_2_corr': top[1][2] if len(top) > 1 else float('nan'), 'top_concept_3': top[2][1] if len(top) > 2 else '', 'top_concept_3_corr': top[2][2] if len(top) > 2 else float('nan')})
    return rows

def export_rule_concept_maps(pred_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for split, df in pred_dfs.items():
        rows.extend(_top_rule_concept_rows(df, split, 'ts', TEMPORAL_CONCEPT_NAMES))
        rows.extend(_top_rule_concept_rows(df, split, 'img', IMAGE_CONCEPT_NAMES))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / 'rule_concept_mapping.csv', index=False, encoding='utf-8-sig')
    return out

def export_point_rule_explanations(pred_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for split, df in pred_dfs.items():
        ts_rule_cols = sorted([c for c in df.columns if c.startswith('ts_rule_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
        img_rule_cols = sorted([c for c in df.columns if c.startswith('img_rule_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
        ts_concept_cols = sorted([c for c in df.columns if c.startswith('ts_concept_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
        img_concept_cols = sorted([c for c in df.columns if c.startswith('img_concept_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
        for _, row in df.iterrows():
            ts_rule_id = int(np.argmax(row[ts_rule_cols].to_numpy(dtype=float))) if ts_rule_cols else -1
            img_rule_id = int(np.argmax(row[img_rule_cols].to_numpy(dtype=float))) if img_rule_cols else -1
            ts_concept_id = int(np.argmax(row[ts_concept_cols].to_numpy(dtype=float))) if ts_concept_cols else -1
            img_concept_id = int(np.argmax(row[img_concept_cols].to_numpy(dtype=float))) if img_concept_cols else -1
            rows.append({'split': split, 'point_index': int(row['point_index']), 'date': row['date'], 'true_label': int(row['true_label']), 'pred_label': int(row['pred_label']), 'prob_smooth': float(row['prob_smooth']), 'ts_prob': float(row['ts_prob']), 'img_prob': float(row['img_prob']), 'temporal_gate': float(row['temporal_gate']), 'image_gate': float(row['image_gate']), 'dominant_ts_rule_id': ts_rule_id, 'dominant_ts_rule_weight': float(row[ts_rule_cols[ts_rule_id]]) if ts_rule_id >= 0 else float('nan'), 'dominant_ts_concept': TEMPORAL_CONCEPT_NAMES[ts_concept_id] if 0 <= ts_concept_id < len(TEMPORAL_CONCEPT_NAMES) else '', 'dominant_img_rule_id': img_rule_id, 'dominant_img_rule_weight': float(row[img_rule_cols[img_rule_id]]) if img_rule_id >= 0 else float('nan'), 'dominant_img_concept': IMAGE_CONCEPT_NAMES[img_concept_id] if 0 <= img_concept_id < len(IMAGE_CONCEPT_NAMES) else ''})
    pd.DataFrame(rows).to_csv(out_dir / 'point_rule_explanations.csv', index=False, encoding='utf-8-sig')

def plot_branch_probability_curves(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    x = np.arange(len(df))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(x, df['prob_smooth'], label='fused', linewidth=1.4)
    axes[0].plot(x, df['ts_prob'], label='temporal branch', linewidth=1.0, alpha=0.78)
    axes[0].plot(x, df['img_prob'], label='image branch', linewidth=1.0, alpha=0.78)
    axes[0].plot(x, df['true_label'], label='true label', linewidth=1.0, alpha=0.75)
    axes[0].axhline(float(df['threshold'].iloc[0]), linestyle='--', linewidth=1.0, color='#8A4F2A', label='threshold')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'{split} branch probability curves')
    axes[0].legend(frameon=False, ncol=5)
    axes[1].plot(x, df['temporal_gate'], label='temporal gate', linewidth=1.1)
    axes[1].plot(x, df['image_gate'], label='image gate', linewidth=1.1)
    axes[1].set_ylabel('Gate')
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(x, df['AE'], label='AE', linewidth=0.8)
    axes[2].plot(x, df['EMR'], label='EMR', linewidth=0.8)
    axes[2].plot(x, df['GAS'], label='GAS', linewidth=0.8)
    axes[2].set_ylabel('Value')
    axes[2].set_xlabel('Point step')
    axes[2].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f'{split}_branch_probability_curves.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def plot_concept_rule_heatmaps(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    ts_concepts = [f'ts_concept_{i}' for i in range(len(TEMPORAL_CONCEPT_NAMES)) if f'ts_concept_{i}' in df.columns]
    img_concepts = [f'img_concept_{i}' for i in range(len(IMAGE_CONCEPT_NAMES)) if f'img_concept_{i}' in df.columns]
    ts_rules = sorted([c for c in df.columns if c.startswith('ts_rule_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
    img_rules = sorted([c for c in df.columns if c.startswith('img_rule_')], key=lambda x: int(x.rsplit('_', 1)[-1]))
    blocks = [('Temporal concepts', ts_concepts, TEMPORAL_CONCEPT_NAMES), ('Image concepts', img_concepts, IMAGE_CONCEPT_NAMES), ('Temporal rules', ts_rules, [f'ts_rule_{i}' for i in range(len(ts_rules))]), ('Image rules', img_rules, [f'img_rule_{i}' for i in range(len(img_rules))])]
    fig, axes = plt.subplots(4, 1, figsize=(14, 9.5), sharex=True, constrained_layout=True)
    for ax, (title, cols, labels) in zip(axes, blocks):
        if cols:
            arr = df[cols].to_numpy(dtype=float).T
            im = ax.imshow(arr, aspect='auto', interpolation='nearest', cmap='viridis')
            ax.set_yticks(np.arange(len(cols)))
            ax.set_yticklabels(labels[:len(cols)], fontsize=7)
            fig.colorbar(im, ax=ax, pad=0.006, fraction=0.018)
        ax.set_title(title)
        ax.set_ylabel('Feature')
    axes[-1].set_xlabel('Point step')
    fig.savefig(out_dir / f'{split}_concept_rule_heatmaps.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def export_key_segments(df: pd.DataFrame, out_dir: Path, split: str, radius: int=45, max_segments: int=8) -> None:
    work = df.copy()
    work['error'] = (work['true_label'] != work['pred_label']).astype(int)
    candidates = pd.concat([work[work['true_label'] == 1].sort_values('prob_smooth', ascending=False).head(max_segments), work[work['error'] == 1].sort_values('prob_smooth', ascending=False).head(max_segments), work.sort_values('prob_smooth', ascending=False).head(max_segments)]).drop_duplicates('point_index').head(max_segments)
    rows = []
    for seg_id, (_, center) in enumerate(candidates.iterrows(), start=1):
        loc = int(work.index[work['point_index'] == center['point_index']][0])
        s = max(0, loc - radius)
        e = min(len(work) - 1, loc + radius)
        seg = work.iloc[s:e + 1].copy()
        seg['segment_id'] = seg_id
        seg['center_point_index'] = int(center['point_index'])
        rows.append(seg[['segment_id', 'center_point_index', 'point_index', 'date', 'AE', 'EMR', 'GAS', 'true_label', 'pred_label', 'prob_smooth', 'ts_prob', 'img_prob']])
        x = np.arange(len(seg))
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, constrained_layout=True)
        axes[0].plot(x, seg['prob_smooth'], label='fused', linewidth=1.4)
        axes[0].plot(x, seg['ts_prob'], label='temporal', linewidth=1.0)
        axes[0].plot(x, seg['img_prob'], label='image', linewidth=1.0)
        axes[0].plot(x, seg['true_label'], label='true label', linewidth=1.0)
        axes[0].axhline(float(seg['threshold'].iloc[0]), linestyle='--', linewidth=1.0, color='#8A4F2A')
        axes[0].axvline(loc - s, linestyle=':', linewidth=1.0, color='#B24C43')
        axes[0].set_title(f"{split} key segment {seg_id}: point {int(center['point_index'])}")
        axes[0].set_ylabel('Probability')
        axes[0].legend(frameon=False, ncol=4)
        axes[1].plot(x, seg['AE'], label='AE', linewidth=0.9)
        axes[1].plot(x, seg['EMR'], label='EMR', linewidth=0.9)
        axes[1].plot(x, seg['GAS'], label='GAS', linewidth=0.9)
        axes[1].axvline(loc - s, linestyle=':', linewidth=1.0, color='#B24C43')
        axes[1].set_ylabel('Value')
        axes[1].set_xlabel('Local step')
        axes[1].legend(frameon=False, ncol=3)
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.savefig(out_dir / f'{split}_key_segment_{seg_id:02d}.png', dpi=220, bbox_inches='tight')
        plt.close(fig)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_dir / f'{split}_key_segments.csv', index=False, encoding='utf-8-sig')

def plot_scatter_and_3d(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    color = df['true_label'].map({0: '#3C5488', 1: '#E64B35'}).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].scatter(df['ts_prob'], df['img_prob'], c=color, s=22, alpha=0.72, edgecolors='none')
    axes[0].set_xlabel('Temporal probability')
    axes[0].set_ylabel('Image probability')
    axes[0].set_title(f'{split} branch probability scatter')
    axes[1].scatter(df['prob_smooth'], df['temporal_gate'], c=color, s=22, alpha=0.72, edgecolors='none')
    axes[1].set_xlabel('Fused probability')
    axes[1].set_ylabel('Temporal gate')
    axes[1].set_title('Gate versus risk')
    axes[2].scatter(df['AE'], df['GAS'], c=df['prob_smooth'], cmap='magma', s=22, alpha=0.78, edgecolors='none')
    axes[2].set_xlabel('AE')
    axes[2].set_ylabel('GAS')
    axes[2].set_title('AE-GAS risk cloud')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / f'{split}_scatter_clouds.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['AE'], df['EMR'], df['GAS'], c=df['prob_smooth'], cmap='magma', s=24, alpha=0.78)
    ax.set_title(f'{split} AE-EMR-GAS 3D risk cloud')
    ax.set_xlabel('AE')
    ax.set_ylabel('EMR')
    ax.set_zlabel('GAS')
    fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72, label='risk probability')
    fig.savefig(out_dir / f'{split}_ae_emr_gas_3d_risk_cloud.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

def plot_explainability_suite(pred_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_rule_concept_maps(pred_dfs, out_dir)
    export_point_rule_explanations(pred_dfs, out_dir)
    for split, df in pred_dfs.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        plot_branch_probability_curves(df, split_dir, split)
        plot_concept_rule_heatmaps(df, split_dir, split)
        export_key_segments(df, split_dir, split)
        plot_scatter_and_3d(df, split_dir, split)

@torch.no_grad()
def plot_embedding_visualizations(model: OptimizedDualBranchCMRNet, loaders: Dict[str, DataLoader], device: torch.device, out_dir: Path, seed: int, max_points: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    embeddings: Dict[str, list[np.ndarray]] = {'temporal': [], 'image': [], 'fused': [], 'decision': []}
    for split, loader in loaders.items():
        for batch in loader:
            out = forward_model_batch(model, batch, device)
            y = batch['y'].detach().cpu().numpy()
            raw_y = batch['raw_y'].detach().cpu().numpy()
            logits = out['logits'].detach().cpu()
            ts_logits = out['ts_logits'].detach().cpu()
            img_logits = out['img_logits'].detach().cpu()
            prob = torch.softmax(logits, dim=-1)[:, 1].numpy()
            ts_prob = torch.softmax(ts_logits, dim=-1)[:, 1].numpy()
            img_prob = torch.softmax(img_logits, dim=-1)[:, 1].numpy()
            n = len(y)
            for i in range(n):
                rows.append({'split': split, 'point_index': int(batch['point_index'][i]), 'true_label': int(y[i]), 'raw_label': int(raw_y[i]), 'prob': float(prob[i]), 'ts_prob': float(ts_prob[i]), 'img_prob': float(img_prob[i])})
            embeddings['temporal'].append(out['ts_embedding'].detach().cpu().numpy())
            embeddings['image'].append(out['img_embedding'].detach().cpu().numpy())
            embeddings['fused'].append(out['fused_embedding'].detach().cpu().numpy())
            decision_parts = [out['fused_embedding'].detach().cpu(), out['ts_y_rule'].detach().cpu(), out['img_y_rule'].detach().cpu(), torch.softmax(ts_logits, dim=-1), torch.softmax(img_logits, dim=-1)]
            if 'stat_features' in batch:
                decision_parts.append(batch['stat_features'].detach().cpu())
            embeddings['decision'].append(torch.cat(decision_parts, dim=-1).numpy())
    if not rows:
        return
    meta = pd.DataFrame(rows)
    emb_arrays = {name: np.concatenate(parts, axis=0) for name, parts in embeddings.items() if parts}
    if max_points > 0 and len(meta) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(meta), size=max_points, replace=False))
        meta = meta.iloc[keep].reset_index(drop=True)
        emb_arrays = {name: emb[keep] for name, emb in emb_arrays.items()}
    marker_map = {'train': 'o', 'val': 's', 'test': '^'}
    color_map = {0: '#3C5488', 1: '#E64B35'}
    for emb_name, emb in emb_arrays.items():
        methods: Dict[str, np.ndarray] = {}
        if len(emb) < 3:
            continue
        train_mask = meta['split'].to_numpy() == 'train'
        fit_emb = emb[train_mask] if train_mask.any() else emb
        scaler = StandardScaler()
        scaler.fit(fit_emb)
        emb_scaled = scaler.transform(emb)
        fit_scaled = emb_scaled[train_mask] if train_mask.any() else emb_scaled
        fit_y = meta.loc[train_mask, 'true_label'].to_numpy() if train_mask.any() else meta['true_label'].to_numpy()
        methods['PCA'] = PCA(n_components=2, random_state=seed).fit_transform(emb_scaled)
        if len(np.unique(fit_y)) >= 2:
            try:
                lda = LinearDiscriminantAnalysis(n_components=1)
                x_lda = lda.fit(fit_scaled, fit_y).transform(emb_scaled).reshape(-1)
                y_pca = PCA(n_components=2, random_state=seed).fit_transform(emb_scaled)[:, 1]
                methods['LDA_PCA'] = np.stack([x_lda, y_pca], axis=1)
            except Exception:
                pass
        perplexity = max(5, min(30, (len(emb) - 1) // 3))
        methods['TSNE'] = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=seed).fit_transform(emb_scaled)
        n_neighbors = max(2, min(15, len(emb) - 1))
        methods['Isomap'] = Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(emb_scaled)
        if umap is not None:
            methods['UMAP'] = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=seed).fit_transform(emb_scaled)
            if len(np.unique(meta['true_label'].to_numpy())) >= 2:
                try:
                    methods['SupervisedUMAP'] = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.05, target_metric='categorical', random_state=seed).fit_transform(emb_scaled, y=meta['true_label'].to_numpy())
                except Exception:
                    pass
        else:
            (out_dir / 'umap_unavailable.txt').write_text('umap-learn is not installed, so UMAP visualization was skipped.\n', encoding='utf-8')
        for method_name, coords in methods.items():
            coord_df = meta.copy()
            coord_df['x'] = coords[:, 0]
            coord_df['y'] = coords[:, 1]
            stem = f'{emb_name}_{method_name.lower()}_embedding'
            coord_df.to_csv(out_dir / f'{stem}.csv', index=False, encoding='utf-8-sig')
            fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
            for split in ['train', 'val', 'test']:
                for label in [0, 1]:
                    sub = coord_df[(coord_df['split'] == split) & (coord_df['true_label'] == label)]
                    if len(sub) == 0:
                        continue
                    ax.scatter(sub['x'], sub['y'], s=18, alpha=0.72, c=color_map[label], marker=marker_map[split], edgecolors='none', label=f'{split}-label{label}')
            ax.set_title(f'{emb_name.title()} embedding: {method_name}')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.grid(True, alpha=0.22)
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False, ncol=2)
            fig.savefig(out_dir / f'{stem}.png', dpi=220, bbox_inches='tight')
            plt.close(fig)
            if 'prob' in coord_df.columns:
                fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
                for split in ['train', 'val', 'test']:
                    sub = coord_df[coord_df['split'] == split]
                    if len(sub) == 0:
                        continue
                    sc = ax.scatter(sub['x'], sub['y'], s=np.where(sub['true_label'].to_numpy() == 1, 34, 18), alpha=0.78, c=sub['prob'], cmap='magma', marker=marker_map[split], edgecolors=np.where(sub['true_label'].to_numpy() == 1, 'black', 'none'), linewidths=np.where(sub['true_label'].to_numpy() == 1, 0.25, 0.0), label=split)
                ax.set_title(f'{emb_name.title()} embedding: {method_name} risk score')
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.grid(True, alpha=0.22)
                ax.legend(fontsize=8, frameon=False)
                fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04, label='fused probability')
                fig.savefig(out_dir / f'{stem}_risk.png', dpi=220, bbox_inches='tight')
                plt.close(fig)
            for split in ['train', 'val', 'test']:
                split_df = coord_df[coord_df['split'] == split].copy()
                if len(split_df) == 0:
                    continue
                split_stem = f'{emb_name}_{method_name.lower()}_{split}_embedding'
                split_df.to_csv(out_dir / f'{split_stem}.csv', index=False, encoding='utf-8-sig')
                fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
                for label in [0, 1]:
                    sub = split_df[split_df['true_label'] == label]
                    if len(sub) == 0:
                        continue
                    ax.scatter(sub['x'], sub['y'], s=22 if label == 0 else 34, alpha=0.7 if label == 0 else 0.88, c=color_map[label], marker=marker_map[split], edgecolors='black' if label == 1 else 'none', linewidths=0.25 if label == 1 else 0.0, label=f'label{label}')
                ax.set_title(f'{emb_name.title()} embedding: {method_name} ({split})')
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.grid(True, alpha=0.22)
                ax.legend(fontsize=8, frameon=False)
                fig.savefig(out_dir / f'{split_stem}.png', dpi=220, bbox_inches='tight')
                plt.close(fig)
        if len(emb) >= 4 and emb.shape[1] >= 3:
            coords3 = PCA(n_components=3, random_state=seed).fit_transform(emb)
            coord3_df = meta.copy()
            coord3_df['x'] = coords3[:, 0]
            coord3_df['y'] = coords3[:, 1]
            coord3_df['z'] = coords3[:, 2]
            stem3 = f'{emb_name}_pca3d_embedding'
            coord3_df.to_csv(out_dir / f'{stem3}.csv', index=False, encoding='utf-8-sig')
            fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
            ax = fig.add_subplot(111, projection='3d')
            for split in ['train', 'val', 'test']:
                for label in [0, 1]:
                    sub = coord3_df[(coord3_df['split'] == split) & (coord3_df['true_label'] == label)]
                    if len(sub) == 0:
                        continue
                    ax.scatter(sub['x'], sub['y'], sub['z'], s=18, alpha=0.72, c=color_map[label], marker=marker_map[split], edgecolors='none', label=f'{split}-label{label}')
            ax.set_title(f'{emb_name.title()} embedding: PCA 3D')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False, ncol=2)
            fig.savefig(out_dir / f'{stem3}.png', dpi=220, bbox_inches='tight')
            plt.close(fig)
