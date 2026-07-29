"""
训练入口：趋势分解 + 6单模型 + Detrending-LSTM-1D-CNN-XGBoost组合模型
"""
import numpy as np
import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from models import MD_STFF_Extractor, PureLSTM, PureCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ============================================================
# 加载数据
# ============================================================
def load_data():
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))
    Y_s = np.load(os.path.join(DATA_DIR, 'Y_s.npy'))
    Y_trend = np.load(os.path.join(DATA_DIR, 'Y_trend.npy'))
    Y_residual = np.load(os.path.join(DATA_DIR, 'Y_residual.npy'))
    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(DATA_DIR, 'val_idx.npy'))
    test_idx = np.load(os.path.join(DATA_DIR, 'test_idx.npy'))

    sp = np.load(os.path.join(DATA_DIR, 'scaler_params.npz'))
    y_mean, y_std = float(sp['y_mean']), float(sp['y_std'])

    print(f'Data: X_rem_s {X_rem_s.shape}, X_met_s {X_met_s.shape}, Y {Y.shape}')
    print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
    return X_rem_s, X_met_s, Y, Y_s, Y_trend, Y_residual, train_idx, val_idx, test_idx, y_mean, y_std

# ============================================================
# 评估函数
# ============================================================
def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    return r2, rmse, mape

# ============================================================
# 深度学习训练
# ============================================================
def train_dl_model(model, X_tr_tensor, Y_tr_tensor, X_val_tensor, Y_val_tensor,
                   epochs=500, lr=0.005, weight_decay=1e-4, loss_fn='mse'):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_fn == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_wts = None
    no_improve = 0
    patience = 80

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_tr = model(X_tr_tensor).squeeze()
        if pred_tr.dim() == 0:
            pred_tr = pred_tr.unsqueeze(0)
        loss = criterion(pred_tr, Y_tr_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_tensor).squeeze()
            if pred_val.dim() == 0:
                pred_val = pred_val.unsqueeze(0)
            val_loss = criterion(pred_val, Y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_wts)
    model.eval()
    return model

# ============================================================
# Extractor + Prediction head training (论文中的双通道网络训练)
# ============================================================
class ExtractorWithHead(nn.Module):
    """MD_STFF_Extractor + FC 预测头，通过预测残差来预训练特征"""
    def __init__(self):
        super().__init__()
        self.extractor = MD_STFF_Extractor()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_rem, x_met):
        feat = self.extractor(x_rem, x_met)  # (N, 128)
        return self.head(feat).squeeze(-1)

    def extract_features(self, x_rem, x_met):
        return self.extractor(x_rem, x_met)


def train_extractor(X_rem_s, X_met_s, Y_residual, train_idx, val_idx,
                    epochs=500, lr=0.003, weight_decay=1e-4):
    """训练 Extractor+Head，用 Huber Loss 拟合残差"""
    model = ExtractorWithHead().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

    X_rem_t = torch.FloatTensor(X_rem_s).to(DEVICE)
    X_met_t = torch.FloatTensor(X_met_s).to(DEVICE)

    # 标准化残差以稳定训练
    res_mean = Y_residual[train_idx].mean()
    res_std = Y_residual[train_idx].std()
    Y_res_norm = (Y_residual - res_mean) / (res_std + 1e-8)
    Y_tr = torch.FloatTensor(Y_res_norm[train_idx]).to(DEVICE)
    Y_val = torch.FloatTensor(Y_res_norm[val_idx]).to(DEVICE)

    best_val_loss = float('inf')
    best_wts = None
    no_improve = 0
    patience = 80

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_rem_t[train_idx], X_met_t[train_idx]).squeeze()
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        loss = criterion(pred, Y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_rem_t[val_idx], X_met_t[val_idx]).squeeze()
            if val_pred.dim() == 0:
                val_pred = val_pred.unsqueeze(0)
            val_loss = criterion(val_pred, Y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_wts)
    model.eval()

    # 提取全量融合特征
    with torch.no_grad():
        fused_features = model.extract_features(X_rem_t, X_met_t).cpu().numpy()  # (N, 128)

    return model, fused_features, res_mean, res_std

# ============================================================
# 主训练流程
# ============================================================
def train_all():
    X_rem_s, X_met_s, Y, Y_s, Y_trend, Y_residual, train_idx, val_idx, test_idx, y_mean, y_std = load_data()

    full_train_idx = np.concatenate([train_idx, val_idx])
    test_idx_orig = test_idx.copy()

    results = {}
    preds_all = {}

    # 输入准备
    X_full = np.concatenate([X_rem_s, X_met_s], axis=-1)  # (N, 11, 9)
    X_ml = np.mean(X_full, axis=1)  # (N, 9) for traditional ML

    X_tr_dl = torch.FloatTensor(X_full[train_idx]).to(DEVICE)
    Y_tr_dl = torch.FloatTensor(Y_s[train_idx]).to(DEVICE)
    X_val_dl = torch.FloatTensor(X_full[val_idx]).to(DEVICE)
    Y_val_dl = torch.FloatTensor(Y_s[val_idx]).to(DEVICE)
    X_all_dl = torch.FloatTensor(X_full).to(DEVICE)

    # --------------------------------------------------------
    # 1. Lasso
    # --------------------------------------------------------
    print('\n=== Lasso ===')
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_ml[full_train_idx], Y[full_train_idx])
    lasso_pred = lasso.predict(X_ml)
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], lasso_pred[test_idx_orig])
    results['Lasso'] = (r2, rmse, mape)
    preds_all['Lasso'] = lasso_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # 2. SVR
    print('\n=== SVR ===')
    scaler_y_ml = StandardScaler()
    Y_train_scaled = scaler_y_ml.fit_transform(Y[full_train_idx].reshape(-1, 1)).ravel()
    svr = SVR(kernel='rbf', C=10, gamma='scale')
    svr.fit(X_ml[full_train_idx], Y_train_scaled)
    svr_pred = scaler_y_ml.inverse_transform(svr.predict(X_ml).reshape(-1, 1)).ravel()
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], svr_pred[test_idx_orig])
    results['SVR'] = (r2, rmse, mape)
    preds_all['SVR'] = svr_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # 3. Random Forest
    print('\n=== Random Forest ===')
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=4,
                               random_state=42, n_jobs=-1)
    rf.fit(X_ml[full_train_idx], Y[full_train_idx])
    rf_pred = rf.predict(X_ml)
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], rf_pred[test_idx_orig])
    results['RF'] = (r2, rmse, mape)
    preds_all['RF'] = rf_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # 4. XGBoost
    print('\n=== XGBoost ===')
    xgb_base = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                random_state=42, eval_metric='rmse', verbosity=0)
    xgb_base.fit(X_ml[full_train_idx], Y[full_train_idx])
    xgb_pred = xgb_base.predict(X_ml)
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], xgb_pred[test_idx_orig])
    results['XGBoost'] = (r2, rmse, mape)
    preds_all['XGBoost'] = xgb_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # 5. Pure LSTM (训练标准化后的 Y_s)
    print('\n=== Pure LSTM ===')
    lstm_model = train_dl_model(PureLSTM(), X_tr_dl, Y_tr_dl, X_val_dl, Y_val_dl, epochs=500)
    with torch.no_grad():
        lstm_pred_s = lstm_model(X_all_dl).cpu().numpy().ravel()
    lstm_pred = lstm_pred_s * y_std + y_mean
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], lstm_pred[test_idx_orig])
    results['LSTM'] = (r2, rmse, mape)
    preds_all['LSTM'] = lstm_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # 6. Pure 1D-CNN
    print('\n=== Pure 1D-CNN ===')
    cnn_model = train_dl_model(PureCNN(), X_tr_dl, Y_tr_dl, X_val_dl, Y_val_dl, epochs=500)
    with torch.no_grad():
        cnn_pred_s = cnn_model(X_all_dl).cpu().numpy().ravel()
    cnn_pred = cnn_pred_s * y_std + y_mean
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], cnn_pred[test_idx_orig])
    results['1D-CNN'] = (r2, rmse, mape)
    preds_all['1D-CNN'] = cnn_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 7. Detrending-LSTM-1D-CNN-XGBoost
    # ============================================================
    print('\n=== Detrending-LSTM-1D-CNN-XGBoost ===')

    # 7a. 预训练 Extractor+Head (Huber Loss, 拟合标准化残差)
    extractor_model, fused_features, res_mean, res_std = train_extractor(
        X_rem_s, X_met_s, Y_residual, train_idx, val_idx, epochs=800, lr=0.003
    )

    # 7b. XGBoost 拟合残差 (网格搜索最优参数)
    xgb_head = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.01,
        subsample=0.8, random_state=42, eval_metric='rmse', verbosity=0
    )
    xgb_head.fit(fused_features[full_train_idx], Y_residual[full_train_idx])
    pred_residuals = xgb_head.predict(fused_features)

    # 7c. 产量重构
    final_pred = Y_trend + pred_residuals
    r2, rmse, mape = calc_metrics(Y[test_idx_orig], final_pred[test_idx_orig])
    results['Detrending-LSTM-CNN-XGBoost'] = (r2, rmse, mape)
    preds_all['Detrending-LSTM-CNN-XGBoost'] = final_pred
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 保存结果
    # ============================================================
    print('\n' + '=' * 70)
    print(f'{"Model":35s} {"R²":>8s} {"RMSE":>10s} {"MAPE":>8s}')
    print('-' * 70)
    for name, (r2, rmse, mape) in results.items():
        print(f'{name:35s} {r2:>8.4f} {rmse:>10.2f} {mape:>7.2f}%')

    np.savez(os.path.join(OUT_DIR, 'results.npz'),
             results=results, preds=preds_all,
             Y=Y, test_idx=test_idx_orig,
             fused_features=fused_features,
             Y_trend=Y_trend, Y_residual=Y_residual)

    torch.save(extractor_model.state_dict(), os.path.join(OUT_DIR, 'extractor_with_head.pth'))
    xgb_head.save_model(os.path.join(OUT_DIR, 'xgb_head.json'))

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump({k: {'R2': round(v[0], 4), 'RMSE': round(v[1], 2), 'MAPE': round(v[2], 2)}
                   for k, v in results.items()}, f, indent=2, ensure_ascii=False)

    print(f'\nResults saved to {OUT_DIR}/')
    return results, preds_all, Y, test_idx_orig

if __name__ == '__main__':
    train_all()
