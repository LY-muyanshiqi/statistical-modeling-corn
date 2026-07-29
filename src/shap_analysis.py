"""
真正的 SHAP 分析 — 对 XGBoost 头部做 TreeExplainer → 蜂群图 + 瀑布图
"""
import numpy as np, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import xgboost as xgb
import shap

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 加载数据
    data = np.load(os.path.join(OUT_DIR, 'results.npz'), allow_pickle=True)
    fused_features = data['fused_features']
    Y_residual = data['Y_residual']
    Y = data['Y']
    Y_trend = data['Y_trend']
    test_idx = data['test_idx']

    # 加载模型
    xgb_head = xgb.XGBRegressor()
    xgb_head.load_model(os.path.join(OUT_DIR, 'xgb_head.json'))

    # 特征名：LSTM_attn_1-64 + CNN_conv_1-64
    feature_names = [f'LSTM_attn_{i+1}' for i in range(64)] + [f'CNN_conv_{i+1}' for i in range(64)]

    # 原始9特征分组映射
    # Remote: NDVI, EVI, LAI, FPAR → 每特征16维 (4×16=64)
    # Meteo: ET, Temp, Precip, Radiation, SoilMoisture → 每特征约13/12维 (5×≈13=65, 实际64)
    raw_names = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Radiation', 'SoilMoisture']
    dims_per = [16, 16, 16, 16, 13, 13, 13, 12, 13]

    print('Calculating SHAP values...')
    explainer = shap.TreeExplainer(xgb_head)
    shap_values = explainer.shap_values(fused_features)

    # ---- 图1: SHAP 蜂群图 (Top-20) ----
    mean_abs = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values[:, top20_idx], fused_features[:, top20_idx],
        feature_names=[feature_names[i] for i in top20_idx],
        show=False, max_display=20
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: shap_beeswarm.png')

    # ---- 图2: SHAP 特征重要性 bar ----
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, top20_idx], fused_features[:, top20_idx],
        feature_names=[feature_names[i] for i in top20_idx],
        plot_type='bar', show=False, max_display=20
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_importance_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: shap_importance_bar.png')

    # ---- 图3: 聚合到9原始特征的SHAP值 ----
    raw_shap = np.zeros((shap_values.shape[0], 9))
    raw_data = np.zeros((shap_values.shape[0], 9))
    s = 0
    for j, d in enumerate(dims_per):
        raw_shap[:, j] = shap_values[:, s:s+d].sum(axis=1)
        raw_data[:, j] = fused_features[:, s:s+d].mean(axis=1)
        s += d

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(raw_shap, raw_data, feature_names=raw_names, show=False, max_display=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_raw_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: shap_raw_features.png')

    # ---- 图4: 按原始特征组的 bar 图 ----
    group_importance = np.abs(raw_shap).mean(axis=0)
    order = np.argsort(group_importance)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_9 = ['#2ecc71' if i < 4 else '#3498db' for i in range(9)]
    ax.barh([raw_names[i] for i in order], group_importance[order],
            color=[colors_9[i] for i in order], edgecolor='white', linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature Importance (Aggregated SHAP)')
    ax.axvline(x=0, color='black', linewidth=0.5)

    # 标注
    labels = ['Remote Sensing', 'Meteorological']
    patches = [plt.Rectangle((0,0),1,1, fc='#2ecc71'), plt.Rectangle((0,0),1,1, fc='#3498db')]
    ax.legend(patches, labels, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_group_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: shap_group_bar.png')

    # ---- 图5: Waterfall 图 (测试集第一个样本) ----
    s_idx = test_idx[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=raw_shap[s_idx],
            base_values=explainer.expected_value,
            data=raw_data[s_idx],
            feature_names=raw_names
        ),
        show=False, max_display=9
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: shap_waterfall.png')

    # ---- 输出排名 ----
    print('\nFeature Importance Ranking (SHAP):')
    for i in order:
        print(f'  {raw_names[i]:15s} = {group_importance[i]:.2e}')

    print('\nSHAP analysis complete!')

if __name__ == '__main__':
    main()
