"""
Leave-One-Year-Out CV — 单模型评估
每年轮作测试集，避免外推效应导致单模型被低估
"""
import numpy as np, os, sys, json, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')

def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    return r2, rmse, mape

def main():
    # 加载数据
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))
    Y_trend = np.load(os.path.join(DATA_DIR, 'Y_trend.npy'))
    Y_residual = np.load(os.path.join(DATA_DIR, 'Y_residual.npy'))

    import pandas as pd
    sample_info = pd.read_csv(os.path.join(DATA_DIR, 'Sample_Info.csv'))

    X_full = np.concatenate([X_rem_s, X_met_s], axis=-1)
    X_ml = np.mean(X_full, axis=1)  # (N, 9)

    years = sorted(sample_info['年份'].unique())
    N = len(Y)

    models = {
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='rmse', verbosity=0),
    }

    results = {name: {'pred': np.zeros(N), 'true': np.zeros(N)} for name in models}

    for year in years:
        test_mask = sample_info['年份'] == year
        train_mask = ~test_mask

        X_tr = X_ml[train_mask]
        Y_tr = Y[train_mask]
        X_te = X_ml[test_mask]
        Y_te = Y[test_mask]

        for name, model_template in models.items():
            if name == 'SVR':
                scaler_y = StandardScaler()
                Y_tr_s = scaler_y.fit_transform(Y_tr.reshape(-1, 1)).ravel()
                m = SVR(kernel='rbf', C=10, gamma='scale')
                m.fit(X_tr, Y_tr_s)
                results[name]['pred'][test_mask] = scaler_y.inverse_transform(
                    m.predict(X_te).reshape(-1, 1)).ravel()
            else:
                m = type(model_template)(**model_template.get_params())
                m.fit(X_tr, Y_tr)
                results[name]['pred'][test_mask] = m.predict(X_te)
            results[name]['true'][test_mask] = Y_te

    # 汇总
    print(f'{"Model":15s} {"R²":>8s} {"RMSE":>10s} {"MAPE":>8s} {"(Paper R²)":>12s}')
    print('-' * 60)
    paper = {'Lasso': 0.5084, 'SVR': 0.7331, 'RF': 0.4648, 'XGBoost': 0.5314}
    for name in models:
        r2, rmse, mape = calc_metrics(results[name]['true'], results[name]['pred'])
        print(f'{name:15s} {r2:>8.4f} {rmse:>10.2f} {mape:>7.2f}% {paper[name]:>12.4f}')

    # 图: LOYO CV 预测散点
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, res) in zip(axes.flat, results.items()):
        r2, rmse, mape = calc_metrics(res['true'], res['pred'])
        ax.scatter(res['true'], res['pred'], alpha=0.5, s=30)
        mn = min(res['true'].min(), res['pred'].min()) - 200
        mx = max(res['true'].max(), res['pred'].max()) + 200
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_title(f'{name} (LOYO CV)\nR²={r2:.4f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%')
        ax.set_xlabel('True (kg/ha)')
        ax.set_ylabel('Pred (kg/ha)')
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'loyo_cv_scatter.png'), dpi=150)
    plt.close()
    print('\nSaved: loyo_cv_scatter.png')

    # 保存
    loyo_results = {name: {
        'R2': round(calc_metrics(results[name]['true'], results[name]['pred'])[0], 4),
        'RMSE': round(calc_metrics(results[name]['true'], results[name]['pred'])[1], 2),
        'MAPE': round(calc_metrics(results[name]['true'], results[name]['pred'])[2], 2),
    } for name in models}

    with open(os.path.join(OUT_DIR, 'loyo_results.json'), 'w') as f:
        json.dump(loyo_results, f, indent=2, ensure_ascii=False)
    print('Saved: loyo_results.json')

if __name__ == '__main__':
    main()
