from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_summary(wide: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_df = wide.sort_values('test_pr_auc', ascending=False)
    x = np.arange(len(plot_df))
    ax.bar(x - 0.25, plot_df['test_f1'], width=0.25, label='Test F1')
    ax.bar(x, plot_df['test_auc'], width=0.25, label='Test AUC')
    ax.bar(x + 0.25, plot_df['test_pr_auc'], width=0.25, label='Test PR-AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['model'], rotation=35, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'baseline_test_metric_comparison.png', dpi=220)
    plt.close(fig)

def plot_training_history(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    for metric in ['ce_loss', 'f1', 'auc', 'pr_auc']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=False)
        for model, group in history.groupby('model'):
            if metric == 'ce_loss':
                axes[0].plot(group['epoch'], group['train_ce_loss'], label=model, linewidth=1.4)
                axes[1].plot(group['epoch'], group['val_ce_loss'], label=model, linewidth=1.4)
                axes[0].set_title('Train CE loss')
                axes[1].set_title('Val CE loss')
                axes[0].set_ylabel('CE loss')
            else:
                axes[0].plot(group['epoch'], group[f'train_{metric}'], label=model, linewidth=1.4)
                axes[1].plot(group['epoch'], group[f'val_{metric}'], label=model, linewidth=1.4)
                axes[0].set_title(f'Train {metric.upper()}')
                axes[1].set_title(f'Val {metric.upper()}')
                axes[0].set_ylabel(metric.upper())
                axes[0].set_ylim(0, 1.05)
                axes[1].set_ylim(0, 1.05)
            axes[0].grid(alpha=0.25)
            axes[1].grid(alpha=0.25)
            axes[0].set_xlabel('Epoch')
            axes[1].set_xlabel('Epoch')
        axes[1].legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(out_dir / f'baseline_epoch_{metric}_curves.png', dpi=220)
        plt.close(fig)
