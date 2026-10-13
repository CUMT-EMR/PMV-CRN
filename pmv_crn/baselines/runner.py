from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pmv_crn.baselines.data import prepare_data
from pmv_crn.baselines.models import BPNN, CNN1D, RNNClassifier, TransformerClassifier
from pmv_crn.baselines.plots import plot_summary, plot_training_history
from pmv_crn.baselines.training import train_sklearn, train_torch_model
from pmv_crn.utils.random import set_seed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', default='classifier.csv')
    parser.add_argument('--out-dir', default=str(Path('results') / 'classifier_baselines_stride10'))
    parser.add_argument('--window', type=int, default=256)
    parser.add_argument('--max-points', type=int, default=0)
    parser.add_argument('--sample-stride', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--purge-windows', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--sampler-positive-ratio', type=float, default=0.3)
    parser.add_argument('--balance-eval-splits', action='store_true')
    parser.add_argument('--save-embeddings', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test, splits = prepare_data(args)
    pd.DataFrame([{'split': 'train', 'n': len(y_train), 'pos': int(y_train.sum()), 'neg': int((y_train == 0).sum())}, {'split': 'val', 'n': len(y_val), 'pos': int(y_val.sum()), 'neg': int((y_val == 0).sum())}, {'split': 'test', 'n': len(y_test), 'pos': int(y_test.sum()), 'neg': int((y_test == 0).sum())}]).to_csv(out_dir / 'split_summary.csv', index=False, encoding='utf-8-sig')
    rows: list[dict[str, object]] = []
    channels = x_train.shape[-1]
    in_dim = x_train.shape[1] * x_train.shape[2]
    deep_models = {'BPNN': BPNN(in_dim), 'CNN': CNN1D(channels), 'LSTM': RNNClassifier(channels, 'lstm'), 'GRU': RNNClassifier(channels, 'gru'), 'Transformer': TransformerClassifier(channels, medformer_like=False), 'Medformer_like': TransformerClassifier(channels, medformer_like=True)}
    for name, model in deep_models.items():
        print(f'=== {name} ===')
        final_rows, hist_rows = train_torch_model(name, model, x_train, y_train, x_val, y_val, x_test, y_test, args, out_dir)
        rows.extend(final_rows)
        pd.DataFrame(hist_rows).to_csv(out_dir / f'{name}_training_history.csv', index=False, encoding='utf-8-sig')
        if hist_rows:
            existing_history = pd.read_csv(out_dir / 'baseline_training_history_long_partial.csv') if (out_dir / 'baseline_training_history_long_partial.csv').exists() else pd.DataFrame()
            pd.concat([existing_history, pd.DataFrame(hist_rows)], ignore_index=True).to_csv(out_dir / 'baseline_training_history_long_partial.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(rows).to_csv(out_dir / 'baseline_metrics_long_partial.csv', index=False, encoding='utf-8-sig')
    print('=== XGBoost ===')
    pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    xgb = XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.035, subsample=0.9, colsample_bytree=0.85, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=pos_weight, random_state=args.seed, n_jobs=4)
    rows.extend(train_sklearn('XGBoost', xgb, x_train, y_train, x_val, y_val, x_test, y_test))
    print('=== SVM ===')
    svm = make_pipeline(StandardScaler(), SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=args.seed))
    rows.extend(train_sklearn('SVM', svm, x_train, y_train, x_val, y_val, x_test, y_test))
    long = pd.DataFrame(rows)
    long.to_csv(out_dir / 'baseline_metrics_long.csv', index=False, encoding='utf-8-sig')
    history_path = out_dir / 'baseline_training_history_long_partial.csv'
    if history_path.exists():
        history = pd.read_csv(history_path).drop_duplicates(subset=['model', 'epoch'], keep='last')
        history.to_csv(out_dir / 'baseline_training_history_long.csv', index=False, encoding='utf-8-sig')
        plot_training_history(history, out_dir)
    wide_rows = []
    for model, group in long.groupby('model'):
        val = group[group['split'] == 'val'].iloc[0]
        test = group[group['split'] == 'test'].iloc[0]
        wide_rows.append({'model': model, 'val_f1': val['f1'], 'val_auc': val['auc'], 'val_pr_auc': val['pr_auc'], 'test_f1': test['f1'], 'test_auc': test['auc'], 'test_pr_auc': test['pr_auc'], 'test_precision': test['precision'], 'test_recall': test['recall'], 'threshold': val['threshold']})
    wide = pd.DataFrame(wide_rows).sort_values(['test_pr_auc', 'test_auc'], ascending=False)
    wide.to_csv(out_dir / 'baseline_metrics_wide.csv', index=False, encoding='utf-8-sig')
    plot_summary(wide, out_dir)
    print(wide.to_string(index=False))

from pmv_crn.baselines.models import BPNN, CNN1D, RNNClassifier, TransformerClassifier

from pmv_crn.baselines.data import balanced_subset, make_windows, WindowDataset, prepare_data

from pmv_crn.baselines.evaluation import metrics, ce_loss_from_prob, best_threshold, predict_torch, collect_torch_embeddings

from pmv_crn.baselines.training import train_torch_model, train_sklearn

from pmv_crn.baselines.plots import plot_summary, plot_training_history
