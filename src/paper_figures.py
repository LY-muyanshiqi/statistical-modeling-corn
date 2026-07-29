"""
论文图表复现 — 关键图表 Matplotlib 精确重绘
"""
import numpy as np, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 图4: 核心指标三维空间相关性矩阵
# ============================================================
def fig4_correlation_matrix():
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(DATA_DIR, '九大指标.csv'))

    features = ['NDVI', 'EVI', 'LAI', 'FPAR', 'NDVI', 'NDVI', 'NDVI', 'NDVI', 'NDVI']  # dummy
    features_real = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Radiation', 'SoilMoisture']
    col_names = ['NDVI', 'EVI', 'LAI', 'FPAR', '蒸散发ET(mm)', '平均气温(℃)', '8天累计降水(mm)', '8天累计辐射(MJ)', '土壤水分(m3/m3)']

    # 计算相关系数矩阵
    X = df[col_names].values
    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    short_names = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Rad', 'SoilM']
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # 标注数值
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(corr[i,j]) > 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson r')
    ax.set_title('Core Indicators Correlation Matrix', fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig4_correlation_matrix.png')


# ============================================================
# 图6: Lasso 特征系数收缩路径
# ============================================================
def fig6_lasso_path():
    from sklearn.linear_model import lasso_path
    import pandas as pd

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(DATA_DIR, '九大指标.csv'))
    col_names = ['NDVI', 'EVI', 'LAI', 'FPAR', '蒸散发ET(mm)', '平均气温(℃)', '8天累计降水(mm)', '8天累计辐射(MJ)', '土壤水分(m3/m3)']

    X = df.groupby(['城市', '年份'])[col_names].mean().values
    Y = pd.read_csv(os.path.join(DATA_DIR, '河南省各市玉米产量(2002-2024).csv'))
    Y = Y.melt(id_vars=['地区'], var_name='年份', value_name='产量')
    area = pd.read_csv(os.path.join(DATA_DIR, '河南省各市玉米播种面积(2002-2024).csv'))
    area = area.melt(id_vars=['地区'], var_name='年份', value_name='播种面积')
    Y['年份'] = Y['年份'].astype(int)
    area['年份'] = area['年份'].astype(int)
    merged = pd.merge(Y, area, on=['地区', '年份'])
    merged['单产'] = (merged['产量'] / merged['播种面积']) * 10000

    y = merged['单产'].values[:X.shape[0]]

    alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-1, 3, 100), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    short_names = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Rad', 'SoilM']
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    for i in range(9):
        ax.plot(alphas, coefs[i], color=colors[i], label=short_names[i], linewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('Coefficient')
    ax.set_title('Lasso Coefficient Shrinkage Path', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_lasso_path.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig6_lasso_path.png')


# ============================================================
# 图10: RF 特征重要性
# ============================================================
def fig10_rf_importance():
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))

    X_full = np.concatenate([X_rem_s, X_met_s], axis=-1)
    X_ml = np.mean(X_full, axis=1)

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42)
    rf.fit(X_ml, Y)

    short_names = ['NDVI', 'EVI', 'LAI', 'FPAR', 'ET', 'Temp', 'Precip', 'Radiation', 'SoilMoisture']
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if i < 3 else '#f39c12' if i < 6 else '#3498db' for i in range(9)]
    ax.barh([short_names[i] for i in order], importances[order],
            color=[colors[i] for i in order], edgecolor='white', linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('RF Core Feature Importance Ranking', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig10_rf_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig10_rf_importance.png')


# ============================================================
# 图13: XGBoost 误差下降曲线
# ============================================================
def fig13_xgb_learning_curve():
    import xgboost as xgb

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))

    X_ml = np.mean(np.concatenate([X_rem_s, X_met_s], axis=-1), axis=1)

    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(DATA_DIR, 'val_idx.npy'))

    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             random_state=42, eval_metric='rmse', verbosity=0)
    model.fit(X_ml[train_idx], Y[train_idx],
              eval_set=[(X_ml[train_idx], Y[train_idx]), (X_ml[val_idx], Y[val_idx])],
              verbose=False)

    results = model.evals_result()
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_rmse)+1), train_rmse, 'b-', linewidth=1.5, label='Train')
    ax.plot(range(1, len(val_rmse)+1), val_rmse, 'r-', linewidth=1.5, label='Validation')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('RMSE (kg/ha)')
    ax.set_title('XGBoost Iterative Learning & Error Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(train_rmse))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig13_xgb_learning_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig13_xgb_learning_curve.png')


# ============================================================
# 图15: LSTM 训练损失下降
# ============================================================
def fig15_lstm_loss():
    import torch, torch.nn as nn, torch.optim as optim
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models import PureLSTM

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y_s.npy'))

    X_full = np.concatenate([X_rem_s, X_met_s], axis=-1)
    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(DATA_DIR, 'val_idx.npy'))

    X_tr = torch.FloatTensor(X_full[train_idx])
    Y_tr = torch.FloatTensor(Y[train_idx])
    X_val = torch.FloatTensor(X_full[val_idx])
    Y_val = torch.FloatTensor(Y[val_idx])

    model = PureLSTM()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tr).squeeze()
        loss = criterion(pred, Y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, Y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 201), train_losses, 'b-', linewidth=1.5, label='Train Loss')
    ax.plot(range(1, 201), val_losses, 'r-', linewidth=1.5, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM Training Loss Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig15_lstm_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig15_lstm_loss.png')


# ============================================================
# 图19: 组合模型构建流程 (流程图)
# ============================================================
def fig19_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 5个模块
    boxes = [
        (1, 3, 'Data Input\n数据输入', '#3498db'),
        (3.5, 3, 'Mechanism\nDecomposition\n机理分解\n(Linear Detrending)', '#2ecc71'),
        (6.5, 4.5, 'Remote Sensing\nChannel\n遥感通道\n(LSTM + Attention)', '#9b59b6'),
        (6.5, 1.5, 'Meteorological\nChannel\n气象通道\n(1D-CNN)', '#e67e22'),
        (9.5, 4.5, 'Feature Concatenation\n特征融合(Concat)', '#1abc9c'),
        (9.5, 1.5, 'Feature Concatenation\n特征融合(Concat)', '#1abc9c'),
        (11.5, 3, 'XGBoost\nResidual Fitting\n残差拟合', '#e74c3c'),
        (12.5, 0.5, 'Yield Reconstruction\n产量重构\n(Y_trend + residual)', '#34495e'),
    ]

    # 简化版：3行 x 4列
    # Row 1: Data → Detrending → Dual Channels → Fusion → XGBoost → Reconstruction
    positions = [
        (0.8, 2.8, 1.6, 1.0, '1. Data Input\n数据输入', '#3498db'),
        (2.8, 2.8, 1.6, 1.0, '2. Detrending\n去趋势化', '#2ecc71'),
        (4.8, 3.5, 1.6, 0.8, '3a. LSTM+Attn\n遥感通道', '#9b59b6'),
        (4.8, 2.2, 1.6, 0.8, '3b. 1D-CNN\n气象通道', '#e67e22'),
        (6.8, 2.8, 1.6, 1.0, '4. Fusion\n特征融合', '#1abc9c'),
        (8.8, 2.8, 1.6, 1.0, '5. XGBoost\n残差拟合', '#e74c3c'),
        (10.8, 2.8, 1.6, 1.0, '6. Reconstruction\n产量重构', '#34495e'),
    ]

    for x, y, w, h, label, color in positions:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                        facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')

    # 箭头
    arrow_positions = [
        (2.4, 3.3, 2.8, 3.3),   # 1→2
        (4.4, 3.3, 4.8, 3.3),   # 2→3
        (6.4, 3.3, 6.8, 3.3),   # 3→4
        (8.4, 3.3, 8.8, 3.3),   # 4→5
        (10.4, 3.3, 10.8, 3.3),  # 5→6
    ]
    for x1, y1, x2, y2 in arrow_positions:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_title('Detrending-LSTM-1D-CNN-XGBoost Pipeline\nCombined Model Architecture',
                fontsize=14, fontweight='bold', pad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig19_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig19_pipeline.png')

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    fig4_correlation_matrix()
    fig6_lasso_path()
    fig10_rf_importance()
    fig13_xgb_learning_curve()
    fig15_lstm_loss()
    fig19_pipeline()
    print('\nAll figures saved to output/figures/')
