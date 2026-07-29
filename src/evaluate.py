"""
评估与可视化：R²/RMSE/MAPE，多模型对比表，散点图
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def plot_scatter(y_true, y_pred, title, filename):
    """真实 vs 预测散点图"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidth=0.5)

    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('True Yield (kg/ha)')
    ax.set_ylabel('Predicted Yield (kg/ha)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(mn - 200, mx + 200)
    ax.set_ylim(mn - 200, mx + 200)
    ax.set_aspect('equal')

    # R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)

def plot_multi_model_comparison(Y, preds_all, test_idx):
    """多模型预测对比图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(test_idx))
    ax.plot(x, Y[test_idx], 'ko-', linewidth=2.5, markersize=6, label='True')

    colors = {'Lasso': 'gray', 'SVR': 'orange', 'RF': 'green', 'XGBoost': 'brown',
              'LSTM': 'blue', '1D-CNN': 'purple', 'Detrending-LSTM-CNN-XGBoost': 'red'}

    for name, pred in preds_all.items():
        if name in colors:
            ax.plot(x, pred[test_idx], 's--', linewidth=1.2, markersize=4,
                    color=colors[name], label=name, alpha=0.7)

    ax.set_xlabel('Sample Index (2023-2024)')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title('Multi-Model Yield Prediction Comparison (Test Set)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'multi_model_comparison.png'), dpi=150)
    plt.close(fig)

def plot_metrics_bar(results):
    """指标柱状对比图"""
    models = list(results.keys())
    r2_vals = [v[0] for v in results.values()]
    rmse_vals = [v[1] for v in results.values()]
    mape_vals = [v[2] for v in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['#3498db'] * len(models)
    # 高亮最优
    best_idx = np.argmax(r2_vals)
    colors[best_idx] = '#e74c3c'

    axes[0].barh(models, r2_vals, color=colors)
    axes[0].set_xlabel('R²')
    axes[0].set_title('R² by Model')
    axes[0].axvline(x=0.8236, color='red', linestyle='--', linewidth=1, label='Paper target')
    axes[0].legend(fontsize=8)

    axes[1].barh(models, rmse_vals, color=colors)
    axes[1].set_xlabel('RMSE (kg/ha)')
    axes[1].set_title('RMSE by Model')

    axes[2].barh(models, mape_vals, color=colors)
    axes[2].set_xlabel('MAPE (%)')
    axes[2].set_title('MAPE by Model')

    for ax in axes:
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'metrics_comparison.png'), dpi=150)
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载结果
    data = np.load(os.path.join(OUT_DIR, 'results.npz'), allow_pickle=True)
    results = data['results'].item()
    preds_all = data['preds'].item()
    Y = data['Y']
    test_idx = data['test_idx']

    print('Model Performance on Test Set (2023-2024):')
    print(f'{"Model":35s} {"R²":>8s} {"RMSE":>10s} {"MAPE":>8s}')
    print('-' * 70)
    for name, (r2, rmse, mape) in results.items():
        print(f'{name:35s} {r2:>8.4f} {rmse:>10.2f} {mape:>7.2f}%')

    # 生成图表
    plot_multi_model_comparison(Y, preds_all, test_idx)
    print('\nSaved: multi_model_comparison.png')

    plot_metrics_bar(results)
    print('Saved: metrics_comparison.png')

    # 组合模型散点图
    name = 'Detrending-LSTM-CNN-XGBoost'
    if name in preds_all:
        plot_scatter(Y[test_idx], preds_all[name][test_idx],
                     f'{name} — Scatter Plot', 'combined_scatter.png')
        print('Saved: combined_scatter.png')

    print('\nDone!')

if __name__ == '__main__':
    main()
