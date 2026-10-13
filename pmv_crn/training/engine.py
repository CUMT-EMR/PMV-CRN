from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pmv_crn.data.dataset import RecursivePointDataset, collate_points
from pmv_crn.data.features import STAT_FEATURE_DIM, compute_stat_normalizer, make_future_any_labels
from pmv_crn.data.splits import split_indices, split_leakage_summary
from pmv_crn.evaluation.inference import predict
from pmv_crn.evaluation.metrics import best_threshold, metrics_from_prob
from pmv_crn.models.network import OptimizedDualBranchCMRNet
from pmv_crn.training.batch import forward_model_batch
from pmv_crn.training.losses import optimized_dual_branch_cmr_loss
from pmv_crn.utils.random import set_seed
from pmv_crn.visualization.analysis import plot_embedding_visualizations, plot_explainability_suite, plot_prediction, plot_training_curves
from pmv_crn.config.cli import build_argparser
from pmv_crn.data.features import window_stat_features
from pmv_crn.data.representations import make_image_concept_prior, make_precursor_image_maps, make_temporal_concept_prior, standardize_channels
from pmv_crn.data.splits import label_segments, allocate_segments_by_class, purge_near_splits, random_stratified_split, blocked_random_stratified_split
from pmv_crn.evaluation.metrics import fbeta_score_np, rolling_mean_prob
from pmv_crn.models.common import IMAGE_CONCEPT_NAMES, TEMPORAL_CONCEPT_NAMES
from pmv_crn.visualization.analysis import _safe_corr_series, _top_rule_concept_rows, export_rule_concept_maps, export_point_rule_explanations, plot_branch_probability_curves, plot_concept_rule_heatmaps, export_key_segments, plot_scatter_and_3d

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures_and_data'
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if args.max_points and args.max_points > 0:
        df = df.iloc[:args.max_points].reset_index(drop=True)
    for col in ['AE', 'EMR', 'GAS', 'LABEL']:
        if col not in df.columns:
            raise ValueError(f'Input CSV must contain column: {col}')
    raw_labels = df['LABEL'].to_numpy(dtype=np.int64)
    labels = make_future_any_labels(raw_labels, args.horizon) if args.label_mode == 'future_any' else raw_labels
    train_idx, val_idx, test_idx = split_indices(labels, args.window, args.train_ratio, args.val_ratio, args.seed, args.max_normal_segment_len, args.purge_windows, args.split_mode, args.block_size)
    if args.sample_stride > 1:

        def _apply_stride(idx: np.ndarray) -> np.ndarray:
            idx = np.asarray(idx, dtype=np.int64)
            keep = (idx - (args.window - 1)) % args.sample_stride == 0
            return idx[keep]
        train_idx = _apply_stride(train_idx)
        val_idx = _apply_stride(val_idx)
        test_idx = _apply_stride(test_idx)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError('Empty split detected. Reduce purge_windows or adjust split ratios.')
    if args.keep_all_positive_train:
        y_train_full = labels[train_idx]
        pos = train_idx[y_train_full == 1]
        neg = train_idx[y_train_full == 0][::max(1, args.negative_train_stride)]
        train_idx = np.sort(np.concatenate([pos, neg]).astype(np.int64))
    values = df[['AE', 'EMR', 'GAS']].to_numpy(dtype=np.float32)
    train_values = values[train_idx]
    global_mean = train_values.mean(axis=0)
    global_std = train_values.std(axis=0) + 1e-06
    stat_mean, stat_std = compute_stat_normalizer(values, train_idx, args.window, global_mean, global_std)
    train_ds = RecursivePointDataset(df, train_idx, labels, args.window, args.img_size, args.image_map_type, global_mean, global_std, stat_mean, stat_std, args.use_absolute_ts, args.cache_dataset)
    val_ds = RecursivePointDataset(df, val_idx, labels, args.window, args.img_size, args.image_map_type, global_mean, global_std, stat_mean, stat_std, args.use_absolute_ts, args.cache_dataset)
    test_ds = RecursivePointDataset(df, test_idx, labels, args.window, args.img_size, args.image_map_type, global_mean, global_std, stat_mean, stat_std, args.use_absolute_ts, args.cache_dataset)
    y_train = labels[train_idx]
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    loader_kwargs = {'collate_fn': collate_points, 'num_workers': args.num_workers, 'pin_memory': pin_memory}
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    if args.balance_mode in {'sampler', 'both'}:
        pos_ratio = float(np.clip(args.sampler_positive_ratio, 0.05, 0.95))
        class_sampling_mass = np.asarray([1.0 - pos_ratio, pos_ratio], dtype=np.float32)
        sample_weights = [float(class_sampling_mass[int(y)] / max(1.0, counts[int(y)])) for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
        class_weights = torch.tensor(len(y_train) / np.maximum(counts, 1.0), dtype=torch.float32).clamp(max=args.max_class_weight) if args.balance_mode == 'both' else torch.ones(2, dtype=torch.float32)
    elif args.balance_mode == 'weight':
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        class_weights = torch.tensor(len(y_train) / np.maximum(counts, 1.0), dtype=torch.float32).clamp(max=args.max_class_weight)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        class_weights = torch.ones(2, dtype=torch.float32)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)
    model = OptimizedDualBranchCMRNet(ts_channels=6 if args.use_absolute_ts else 3, img_channels=9, n_classes=2, d_model=args.d_model, n_heads=args.n_heads, ts_rules=args.ts_rules, img_rules=args.img_rules, dropout=args.dropout, concept_prior_blend=args.concept_prior_blend, stat_feature_dim=0 if args.model_mode == 'no_stat_features' else STAT_FEATURE_DIM, model_mode=args.model_mode, fixed_temporal_weight=args.fixed_temporal_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    split_summary = pd.DataFrame([{'split_mode': args.split_mode, 'split': 'train', 'points': len(train_idx), 'label_0': int((labels[train_idx] == 0).sum()), 'label_1': int((labels[train_idx] == 1).sum()), 'raw_label_0': int((raw_labels[train_idx] == 0).sum()), 'raw_label_1': int((raw_labels[train_idx] == 1).sum())}, {'split_mode': args.split_mode, 'split': 'val', 'points': len(val_idx), 'label_0': int((labels[val_idx] == 0).sum()), 'label_1': int((labels[val_idx] == 1).sum()), 'raw_label_0': int((raw_labels[val_idx] == 0).sum()), 'raw_label_1': int((raw_labels[val_idx] == 1).sum())}, {'split_mode': args.split_mode, 'split': 'test', 'points': len(test_idx), 'label_0': int((labels[test_idx] == 0).sum()), 'label_1': int((labels[test_idx] == 1).sum()), 'raw_label_0': int((raw_labels[test_idx] == 0).sum()), 'raw_label_1': int((raw_labels[test_idx] == 1).sum())}])
    split_summary.to_csv(out_dir / 'optimized_split_summary.csv', index=False, encoding='utf-8-sig')
    leakage_summary = split_leakage_summary(train_idx, val_idx, test_idx, args.window)
    leakage_summary.to_csv(out_dir / 'optimized_split_leakage_summary.csv', index=False, encoding='utf-8-sig')
    balance_summary = {'split_mode': args.split_mode, 'balance_mode': args.balance_mode, 'sampler_positive_ratio': float(args.sampler_positive_ratio) if args.balance_mode in {'sampler', 'both'} else float('nan'), 'original_train_points': float(len(y_train)), 'original_label_0': float(counts[0]), 'original_label_1': float(counts[1]), 'original_label_0_ratio': float(counts[0] / max(1.0, len(y_train))), 'original_label_1_ratio': float(counts[1] / max(1.0, len(y_train))), 'class_weight_0': float(class_weights.detach().cpu()[0]), 'class_weight_1': float(class_weights.detach().cpu()[1]), 'cache_dataset': bool(args.cache_dataset), 'pin_memory': bool(pin_memory), 'num_workers': int(args.num_workers), 'prefetch_factor': int(args.prefetch_factor) if args.num_workers > 0 else 0, 'batch_size': int(args.batch_size), 'sampled_pos_to_raw_pos_ratio': float(args.sampler_positive_ratio / max(1e-06, counts[1] / max(1.0, len(y_train))) if args.balance_mode in {'sampler', 'both'} else 1.0)}
    pd.DataFrame([balance_summary]).to_csv(out_dir / 'optimized_balance_summary.csv', index=False, encoding='utf-8-sig')
    print(f"Loaded {len(df)} points; split_mode={args.split_mode}; device={device}; class_weights={class_weights.detach().cpu().tolist()}; cache_dataset={args.cache_dataset}; pin_memory={pin_memory}; num_workers={args.num_workers}; batch_size={args.batch_size}; sampled_pos_to_raw_pos_ratio={balance_summary['sampled_pos_to_raw_pos_ratio']:.2f}")
    print(split_summary.to_string(index=False))
    print(leakage_summary.to_string(index=False))
    if args.split_mode == 'random_stratified':
        print('WARNING: random_stratified is a leaky upper-bound split for sliding windows; nearby windows can appear in train/val/test.')
    print(f'balance_summary={balance_summary}')
    history = []
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_path = out_dir / 'optimized_dual_branch_cmr_best.pt'
    for epoch in range(1, args.epochs + 1):
        model.train()
        lambda_contrast = 0.0 if epoch <= args.contrast_warmup_epochs else args.lambda_contrast
        loss_meter: Dict[str, list[float]] = {}
        sampled_label_0 = 0
        sampled_label_1 = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = forward_model_batch(model, batch, device)
            y_batch = batch['y'].to(device)
            sampled_label_0 += int((y_batch == 0).sum().item())
            sampled_label_1 += int((y_batch == 1).sum().item())
            losses = optimized_dual_branch_cmr_loss(out, y_batch, model=model, concept_targets_ts=batch['ts_concepts'].to(device) if args.lambda_concept > 0 else None, concept_targets_img=batch['img_concepts'].to(device) if args.lambda_concept > 0 else None, class_weights=class_weights if args.balance_mode in {'weight', 'both'} else None, loss_type=args.loss_type, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma, lambda_aux=args.lambda_aux, lambda_contrast=lambda_contrast, lambda_rule_consistency=args.lambda_rule_consistency, lambda_concept=args.lambda_concept, lambda_rule_sparse=args.lambda_rule_sparse, lambda_proto=args.lambda_proto, lambda_img_aux=args.lambda_img_aux, lambda_image_metric=args.lambda_image_metric, lambda_image_proto=args.lambda_image_proto, lambda_image_view_diversity=args.lambda_image_view_diversity, lambda_img_rate=args.lambda_img_rate, lambda_gate_balance=args.lambda_gate_balance, gate_target_image_share=args.gate_target_image_share, metric_temperature=args.metric_temperature)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in losses.items():
                loss_meter.setdefault(k, []).append(float(v.detach().cpu()))
        eval_loss_kwargs = {'class_weights': class_weights if args.balance_mode in {'weight', 'both'} else None, 'loss_type': args.loss_type, 'focal_alpha': args.focal_alpha, 'focal_gamma': args.focal_gamma, 'lambda_aux': args.lambda_aux, 'lambda_contrast': lambda_contrast, 'lambda_rule_consistency': args.lambda_rule_consistency, 'lambda_concept': args.lambda_concept, 'lambda_rule_sparse': args.lambda_rule_sparse, 'lambda_proto': args.lambda_proto, 'lambda_img_aux': args.lambda_img_aux, 'lambda_image_metric': args.lambda_image_metric, 'lambda_image_proto': args.lambda_image_proto, 'lambda_image_view_diversity': args.lambda_image_view_diversity, 'lambda_img_rate': args.lambda_img_rate, 'lambda_gate_balance': args.lambda_gate_balance, 'gate_target_image_share': args.gate_target_image_share, 'metric_temperature': args.metric_temperature}
        val_m05, val_df05 = predict(model, val_loader, device, threshold=0.5, smooth_window=args.smooth_window, loss_kwargs=eval_loss_kwargs)
        th = best_threshold(val_df05['true_label'].to_numpy(), val_df05['prob_smooth'].to_numpy(), threshold_min=args.threshold_min, threshold_max=args.threshold_max, min_precision=args.threshold_min_precision)
        train_m, _ = predict(model, train_eval_loader, device, threshold=th, smooth_window=args.smooth_window, loss_kwargs=eval_loss_kwargs)
        val_m, val_df = predict(model, val_loader, device, threshold=th, smooth_window=args.smooth_window, loss_kwargs=eval_loss_kwargs)
        test_m, test_df_probe = predict(model, test_loader, device, threshold=th, smooth_window=args.smooth_window, loss_kwargs=eval_loss_kwargs)
        test_oracle_th = best_threshold(test_df_probe['true_label'].to_numpy(), test_df_probe['prob_smooth'].to_numpy(), threshold_min=0.001, threshold_max=0.999, min_precision=0.0)
        test_oracle_m = metrics_from_prob(test_df_probe['true_label'].to_numpy(), test_df_probe['prob_smooth'].to_numpy(), test_oracle_th)
        scheduler.step(val_m['f1'])
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'threshold': th, 'lambda_contrast_active': lambda_contrast, 'sampled_train_label_0': sampled_label_0, 'sampled_train_label_1': sampled_label_1, 'sampled_train_label_0_ratio': sampled_label_0 / max(1, sampled_label_0 + sampled_label_1), 'sampled_train_label_1_ratio': sampled_label_1 / max(1, sampled_label_0 + sampled_label_1), **{f'train_loss_{k}': float(np.mean(v)) for k, v in loss_meter.items()}, **{f'train_{k}': v for k, v in train_m.items()}, **{f'val_{k}': v for k, v in val_m.items()}, **{f'test_{k}': v for k, v in test_m.items()}, **{f'test_oracle_{k}': v for k, v in test_oracle_m.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / 'optimized_training_history.csv', index=False, encoding='utf-8-sig')
        print(f"Epoch {epoch:03d} | sampled_0={sampled_label_0} sampled_1={sampled_label_1} val_f1={val_m['f1']:.4f} val_auc={val_m['auc']:.4f} val_prauc={val_m['pr_auc']:.4f} test_f1={test_m['f1']:.4f} test_auc={test_m['auc']:.4f} test_prauc={test_m['pr_auc']:.4f} test_best_f1={test_oracle_m['f1']:.4f} test_best_th={test_oracle_th:.3f} th={th:.3f} train_loss={row['train_loss_loss']:.4f} val_loss={val_m.get('loss', float('nan')):.4f} test_loss={test_m.get('loss', float('nan')):.4f}")
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            best_epoch = epoch
            bad_epochs = 0
            torch.save({'model_state': model.state_dict(), 'args': vars(args), 'threshold': th, 'best_epoch': best_epoch, 'global_mean': global_mean, 'global_std': global_std, 'stat_mean': stat_mean, 'stat_std': stat_std}, best_path)
            val_df.to_csv(out_dir / 'optimized_val_predictions_best.csv', index=False, encoding='utf-8-sig')
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_f1={best_f1:.4f}')
                break
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    best_th = float(ckpt['threshold'])
    final_loss_kwargs = {'class_weights': class_weights if args.balance_mode in {'weight', 'both'} else None, 'loss_type': args.loss_type, 'focal_alpha': args.focal_alpha, 'focal_gamma': args.focal_gamma, 'lambda_aux': args.lambda_aux, 'lambda_contrast': float(ckpt['args'].get('lambda_contrast', args.lambda_contrast)), 'lambda_rule_consistency': args.lambda_rule_consistency, 'lambda_concept': args.lambda_concept, 'lambda_rule_sparse': args.lambda_rule_sparse, 'lambda_proto': args.lambda_proto, 'lambda_img_aux': args.lambda_img_aux, 'lambda_image_metric': args.lambda_image_metric, 'lambda_image_proto': args.lambda_image_proto, 'lambda_image_view_diversity': args.lambda_image_view_diversity, 'lambda_img_rate': args.lambda_img_rate, 'lambda_gate_balance': args.lambda_gate_balance, 'gate_target_image_share': args.gate_target_image_share, 'metric_temperature': args.metric_temperature}
    final_rows = []
    final_pred_dfs: Dict[str, pd.DataFrame] = {}
    for split, loader in [('train', train_eval_loader), ('val', val_loader), ('test', test_loader)]:
        m, pred_df = predict(model, loader, device, threshold=best_th, smooth_window=args.smooth_window, loss_kwargs=final_loss_kwargs)
        pred_df.to_csv(out_dir / f'optimized_{split}_predictions.csv', index=False, encoding='utf-8-sig')
        plot_prediction(pred_df, fig_dir, split)
        final_pred_dfs[split] = pred_df
        final_rows.append({'split': split, **m})
    pd.DataFrame(final_rows).to_csv(out_dir / 'optimized_final_metrics.csv', index=False, encoding='utf-8-sig')
    hist_df = pd.DataFrame(history)
    plot_training_curves(hist_df, fig_dir)
    plot_embedding_visualizations(model, {'train': train_eval_loader, 'val': val_loader, 'test': test_loader}, device, fig_dir / 'embeddings', args.seed, args.embedding_max_points)
    plot_explainability_suite(final_pred_dfs, fig_dir / 'explainability')
    print('Final metrics:')
    print(pd.DataFrame(final_rows).to_string(index=False))
    return pd.DataFrame(final_rows)

def auto_search(args: argparse.Namespace) -> None:
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    configs = [{'name': 'blocked_sampler012_raw', 'split_mode': 'blocked_random_stratified', 'block_size': 1024, 'label_mode': 'raw', 'horizon': 0, 'balance_mode': 'sampler', 'sampler_positive_ratio': 0.12, 'loss_type': 'ce', 'lr': 0.0008, 'dropout': 0.15, 'lambda_aux': 0.1, 'lambda_contrast': 0.0, 'lambda_concept': 0.0, 'lambda_proto': 0.0, 'd_model': 96}, {'name': 'blocked_sampler008_raw', 'split_mode': 'blocked_random_stratified', 'block_size': 1024, 'label_mode': 'raw', 'horizon': 0, 'balance_mode': 'sampler', 'sampler_positive_ratio': 0.08, 'loss_type': 'ce', 'lr': 0.0006, 'dropout': 0.18, 'lambda_aux': 0.1, 'lambda_contrast': 0.0, 'lambda_concept': 0.0, 'lambda_proto': 0.0, 'd_model': 96}, {'name': 'event_sampler012_future120', 'split_mode': 'event', 'label_mode': 'future_any', 'horizon': 120, 'balance_mode': 'sampler', 'sampler_positive_ratio': 0.12, 'loss_type': 'ce', 'lr': 0.0006, 'dropout': 0.2, 'lambda_aux': 0.08, 'lambda_contrast': 0.0, 'lambda_concept': 0.0, 'lambda_proto': 0.0, 'd_model': 96}, {'name': 'random_leaky_upper_bound', 'split_mode': 'random_stratified', 'label_mode': 'raw', 'horizon': 0, 'balance_mode': 'sampler', 'sampler_positive_ratio': 0.12, 'loss_type': 'ce', 'lr': 0.0005, 'dropout': 0.15, 'lambda_aux': 0.1, 'lambda_contrast': 0.0, 'lambda_concept': 0.0, 'lambda_proto': 0.0, 'd_model': 96}]
    rows = []
    best_score = -1.0
    best_dir = ''
    for i, cfg in enumerate(configs, start=1):
        trial_args = argparse.Namespace(**vars(args))
        for key, value in cfg.items():
            if key != 'name':
                setattr(trial_args, key, value)
        trial_args.out_dir = str(base_out / f"trial_{i:02d}_{cfg['name']}")
        print(f"\n=== Trial {i}/{len(configs)}: {cfg['name']} ===")
        metrics_df = train(trial_args)
        flat = {'trial': i, 'name': cfg['name'], 'out_dir': trial_args.out_dir}
        for _, metric_row in metrics_df.iterrows():
            split = str(metric_row['split'])
            for col, value in metric_row.items():
                if col != 'split':
                    flat[f'{split}_{col}'] = value
        score = float(flat.get('val_f1', 0.0)) + float(flat.get('val_auc', 0.0)) + float(flat.get('val_pr_auc', 0.0))
        flat['selection_score'] = score
        rows.append(flat)
        if score > best_score:
            best_score = score
            best_dir = trial_args.out_dir
        target_ok = all((float(flat.get(f'{split}_{metric}', 0.0)) >= args.target_metric for split in ['val', 'test'] for metric in ['accuracy', 'f1', 'auc', 'pr_auc']))
        if target_ok:
            print(f'Reached target >= {args.target_metric:.2f}: {trial_args.out_dir}')
            break
    summary = pd.DataFrame(rows)
    summary.to_csv(base_out / 'auto_search_summary.csv', index=False, encoding='utf-8-sig')
    print(f"Auto-search summary saved: {base_out / 'auto_search_summary.csv'}")
    print(f'Best trial by validation F1+AUC+PR-AUC: {best_dir}')
