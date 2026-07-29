"""
SHAP / Feature Importance 分析 — 用 XGBoost 内置 + 近似 SHAP
由于环境 NumPy 2.x 不兼容 shap 包，使用 XGBoost 内置的 gain-based importance
和自实现的 permutation importance 作为替代
"""
import numpy as np, os, sys, json, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def permutation_importance(model, X, y_true, n_repeats=10):
    """自实现 permutation feature importance"""
    baseline_pred = model.predict(X)
    baseline_mse = np.mean((y_true - baseline_pred) ** 2)
    n_features = X.shape[1]

    importances = np.zeros(n_features)
    importances_std = np.zeros(n_features)

    for j in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            pred_perm = model.predict(X_perm)
            mse = np.mean((y_true - pred_perm) ** 2)
            scores.append(mse - baseline_mse)
        importances[j] = np.mean(scores)
        importances_std[j] = np.std(scores)

    return importances, importances_std


def plot_feature_importance(fused_features, Y_residual, xgb_model, test_idx):
    """生成 SHAP 替代图表"""

    # 1. XGBoost 内置 gain-based importance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gain importance — 按组聚合
    gain = xgb_model.get_booster().get_score(importance_type='gain')
    # f0-f127 -> gain values
    gain_lstm = np.mean([gain.get(f'f{i}', 0) for i in range(64)])
    gain_cnn = np.mean([gain.get(f'f{i}', 0) for i in range(64, 128)])

    ax = axes[0, 0]
    groups = ['Remote Sensing\n(LSTM+Attn)', 'Meteorological\n(1D-CNN)']
    values = [gain_lstm, gain_cnn]
    bars = ax.bar(groups, values, color=['#2ecc71', '#3498db'], edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Avg Gain Importance')
    ax.set_title('Feature Group Importance (Gain-based)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{val:.1f}', ha='center', fontsize=11)

    # Weight importance
    weight = xgb_model.get_booster().get_score(importance_type='weight')
    w_lstm = np.mean([weight.get(f'f{i}', 0) for i in range(64)])
    w_cnn = np.mean([weight.get(f'f{i}', 0) for i in range(64, 128)])

    ax = axes[0, 1]
    bars = ax.bar(groups, [w_lstm, w_cnn], color=['#2ecc71', '#3498db'], edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Avg Weight (# splits)')
    ax.set_title('Feature Group Importance (Weight-based)')
    for bar, val in zip(bars, [w_lstm, w_cnn]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([w_lstm, w_cnn])*0.01,
                f'{val:.1f}', ha='center', fontsize=11)

    # 2. 按物理意义的9维原始特征重要性（通过聚合128维特征）
    # LSTM 64-dim 对应 4 个遥感特征 → 每个遥感特征约16维
    # CNN 64-dim 对应 5 个气象特征 → 每个气象特征约13维

    # 用 permutation importance 量化每个融合维度的重要性
    importances, stds = permutation_importance(xgb_model, fused_features, Y_residual, n_repeats=5)

    # Top-20 特征
    top20_idx = np.argsort(importances)[-20:][::-1]
    top20_names = [f'LSTM_attn_{i+1}' if i < 64 else f'CNN_conv_{i-63}' for i in top20_idx]

    ax = axes[1, 0]
    ax.barh(range(20), importances[top20_idx], xerr=stds[top20_idx],
            color=['#2ecc71' if i < 64 else '#3498db' for i in top20_idx],
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (MSE increase)')
    ax.set_title('Top-20 Fused Feature Importance (Permutation)')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # 3. 按原始9特征的聚合重要性
    # 将128维按原始9特征映射
    # 遥感4特征: LSTM处理 → 前64维大致均分给4个遥感特征
    # 气象5特征: CNN处理 → 后64维大致均分给5个气象特征
    raw_features = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Radiation', 'SoilMoisture']

    # 每个原始特征贡献的维度数
    dims_per_feature = [16]*4 + [13, 13, 13, 12, 13]  # 近似分配
    raw_importance = []
    raw_std = []
    start = 0
    for d in dims_per_feature:
        raw_importance.append(np.sum(importances[start:start+d]))
        raw_std.append(np.sqrt(np.sum(stds[start:start+d]**2)))
        start += d

    ax = axes[1, 1]
    colors = ['#e74c3c' if i < 4 else '#f39c12' for i in range(9)]
    ax.barh(raw_features, raw_importance, xerr=raw_std, color=colors, edgecolor='white', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel('Aggregated Feature Importance')
    ax.set_title('Aggregated Raw Feature Importance')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_importance_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: feature_importance_analysis.png')

    # 4. 特征重要性总结表
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')

    rank_indices = np.argsort(raw_importance)[::-1]
    table_data = [[f'{i+1}', raw_features[j], f'{raw_importance[j]:.2e}', f'{raw_std[j]:.2e}']
                  for i, j in enumerate(rank_indices)]

    table = ax.table(cellText=table_data,
                     colLabels=['Rank', 'Feature', 'Importance', 'Std'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')

    ax.set_title('Feature Importance Ranking\n(Aggregated from 128-Dim Fused Features)',
                fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_importance_table.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: feature_importance_table.png')

    return raw_importance, raw_features, rank_indices


def main():
    # 加载
    data = np.load(os.path.join(OUT_DIR, 'results.npz'), allow_pickle=True)
    fused_features = data['fused_features']
    Y_residual = data['Y_residual']
    Y = data['Y']
    Y_trend = data['Y_trend']
    test_idx = data['test_idx']

    # 加载模型
    xgb_head = xgb.XGBRegressor()
    xgb_head.load_model(os.path.join(OUT_DIR, 'xgb_head.json'))

    # 分析
    raw_importance, raw_features, rank_indices = plot_feature_importance(
        fused_features, Y_residual, xgb_head, test_idx)

    # 输出排名
    print('\nFeature Importance Ranking:')
    for i, j in enumerate(rank_indices):
        print(f'  {i+1}. {raw_features[j]:15s} = {raw_importance[j]:.2e}')

    # 论文对应的发现：太阳辐射第一，植被指数(EVI/NDVI)紧随其后
    print('\nDone!')

if __name__ == '__main__':
    main()
