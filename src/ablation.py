"""
消融实验：对照 Table 11
策略：每个消融变体 = Extractor(变体) + PredictionHead + 可选XGBoost
均使用与主训练相同的 Extractor+Head 预训练范式
"""
import numpy as np
import os, sys, json, warnings
warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
    return r2, rmse, mape

def load_data():
    X_rem_s = np.load(os.path.join(DATA_DIR, 'X_remote.npy'))
    X_met_s = np.load(os.path.join(DATA_DIR, 'X_meteo.npy'))
    Y = np.load(os.path.join(DATA_DIR, 'Y.npy'))
    Y_trend = np.load(os.path.join(DATA_DIR, 'Y_trend.npy'))
    Y_residual = np.load(os.path.join(DATA_DIR, 'Y_residual.npy'))
    train_idx = np.load(os.path.join(DATA_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(DATA_DIR, 'val_idx.npy'))
    test_idx = np.load(os.path.join(DATA_DIR, 'test_idx.npy'))
    return X_rem_s, X_met_s, Y, Y_trend, Y_residual, train_idx, val_idx, test_idx

def train_extractor_head(extractor, head, X_rem_tr, X_met_tr, Y_tr, X_rem_val, X_met_val, Y_val,
                         epochs=500, lr=0.003, use_huber=True):
    """训练 extractor+head 组合"""
    class Model(nn.Module):
        def __init__(self, ext, hd):
            super().__init__()
            self.ext = ext
            self.head = hd
        def forward(self, x_rem, x_met):
            feat = self.ext(x_rem, x_met)
            return self.head(feat).squeeze(-1)

    model = Model(extractor, head).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if use_huber:
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()

    # 标准化残差
    res_mean, res_std = Y_tr.mean().item(), Y_tr.std().item()
    Y_tr_n = (Y_tr - res_mean) / (res_std + 1e-8)
    Y_val_n = (Y_val - res_mean) / (res_std + 1e-8)

    best_val_loss = float('inf')
    best_wts = None
    no_improve = 0
    patience = 80

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_rem_tr, X_met_tr)
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        loss = criterion(pred, Y_tr_n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_rem_val, X_met_val)
            if val_pred.dim() == 0:
                val_pred = val_pred.unsqueeze(0)
            val_loss = criterion(val_pred, Y_val_n).item()

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
    return model, res_mean, res_std


# ============================================================
# 消融变体定义
# ============================================================

class FullLSTMAttn(nn.Module):
    """只有 LSTM+Attention (9维输入)"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 64, batch_first=True)
        self.attn = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        return torch.sum(w * out, dim=1)  # (N, 64)

class FullCNN(nn.Module):
    """只有 1D-CNN (9维输入)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(9, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).squeeze(-1)  # (N, 64)

class DualExtractor(nn.Module):
    """完整 LSTM+Attention + CNN 双通道"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 64, batch_first=True)
        self.attn = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.conv = nn.Sequential(nn.Conv1d(5, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
    def forward(self, x_rem, x_met):
        out, _ = self.lstm(x_rem)
        w = torch.softmax(self.attn(out), dim=1)
        f1 = torch.sum(w * out, dim=1)
        f2 = self.conv(x_met.transpose(1, 2)).squeeze(-1)
        return torch.cat([f1, f2], dim=1)  # (N, 128)

class DualExtractorNoLSTM(nn.Module):
    """只用 CNN 双通道替代"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(9, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
    def forward(self, x_rem, x_met):
        x = torch.cat([x_rem, x_met], dim=-1)  # concat features
        return self.conv(x.transpose(1, 2)).squeeze(-1)  # (N, 64)

class DualExtractorNoCNN(nn.Module):
    """只用 LSTM+Attention"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 64, batch_first=True)
        self.attn = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x_rem, x_met):
        x = torch.cat([x_rem, x_met], dim=-1)
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        return torch.sum(w * out, dim=1)  # (N, 64)


def run_ablation():
    X_rem_s, X_met_s, Y, Y_trend, Y_residual, train_idx, val_idx, test_idx = load_data()
    full_train_idx = np.concatenate([train_idx, val_idx])

    X_full = np.concatenate([X_rem_s, X_met_s], axis=-1)
    X_rem_t = torch.FloatTensor(X_rem_s).to(DEVICE)
    X_met_t = torch.FloatTensor(X_met_s).to(DEVICE)
    X_full_t = torch.FloatTensor(X_full).to(DEVICE)

    Y_tr = torch.FloatTensor(Y_residual[train_idx]).to(DEVICE)
    Y_val = torch.FloatTensor(Y_residual[val_idx]).to(DEVICE)

    results = {}

    # ============================================================
    # 1. LSTM-1D-CNN-XGBoost (无 Detrending → 直接用原始Y)
    # ============================================================
    print('=== 1. LSTM-1D-CNN-XGBoost (无 Detrending) ===')
    ext1 = DualExtractor()
    head1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    Y_tr_raw = torch.FloatTensor(Y[train_idx]).to(DEVICE)
    Y_val_raw = torch.FloatTensor(Y[val_idx]).to(DEVICE)
    model1, mu1, std1 = train_extractor_head(ext1, head1, X_rem_t[train_idx], X_met_t[train_idx], Y_tr_raw,
                                             X_rem_t[val_idx], X_met_t[val_idx], Y_val_raw)

    # 提取特征 → XGBoost 拟合原始Y
    model1.eval()
    with torch.no_grad():
        feat1 = model1.ext(X_rem_t, X_met_t).cpu().numpy()

    xgb1 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.005,
                            subsample=0.8, random_state=42, eval_metric='rmse', verbosity=0)
    xgb1.fit(feat1[full_train_idx], Y[full_train_idx])
    pred1 = xgb1.predict(feat1)
    r2, rmse, mape = calc_metrics(Y[test_idx], pred1[test_idx])
    results['LSTM-1D-CNN-XGBoost'] = (r2, rmse, mape)
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 2. Detrending-1D-CNN-XGBoost (无 LSTM)
    # ============================================================
    print('\n=== 2. Detrending-1D-CNN-XGBoost (无 LSTM) ===')
    ext2 = DualExtractorNoLSTM()
    head2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    model2, mu2, std2 = train_extractor_head(ext2, head2, X_rem_t[train_idx], X_met_t[train_idx], Y_tr,
                                             X_rem_t[val_idx], X_met_t[val_idx], Y_val)

    model2.eval()
    with torch.no_grad():
        feat2 = model2.ext(X_rem_t, X_met_t).cpu().numpy()

    xgb2 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.005,
                            subsample=0.8, random_state=42, eval_metric='rmse', verbosity=0)
    xgb2.fit(feat2[full_train_idx], Y_residual[full_train_idx])
    pred2 = Y_trend + xgb2.predict(feat2)
    r2, rmse, mape = calc_metrics(Y[test_idx], pred2[test_idx])
    results['Detrending-1D-CNN-XGBoost'] = (r2, rmse, mape)
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 3. Detrending-LSTM-XGBoost (无 CNN)
    # ============================================================
    print('\n=== 3. Detrending-LSTM-XGBoost (无 CNN) ===')
    ext3 = DualExtractorNoCNN()
    head3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    model3, mu3, std3 = train_extractor_head(ext3, head3, X_rem_t[train_idx], X_met_t[train_idx], Y_tr,
                                             X_rem_t[val_idx], X_met_t[val_idx], Y_val)

    model3.eval()
    with torch.no_grad():
        feat3 = model3.ext(X_rem_t, X_met_t).cpu().numpy()

    xgb3 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.005,
                            subsample=0.8, random_state=42, eval_metric='rmse', verbosity=0)
    xgb3.fit(feat3[full_train_idx], Y_residual[full_train_idx])
    pred3 = Y_trend + xgb3.predict(feat3)
    r2, rmse, mape = calc_metrics(Y[test_idx], pred3[test_idx])
    results['Detrending-LSTM-XGBoost'] = (r2, rmse, mape)
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 4. Detrending-LSTM-1D-CNN (无 XGBoost → 直接用DL head预测)
    # ============================================================
    print('\n=== 4. Detrending-LSTM-1D-CNN (无 XGBoost) ===')
    # 复用之前训练的模型4 (或用同一extractor+更大的head)
    ext4 = DualExtractor()
    head4 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(64, 1))
    model4, mu4, std4 = train_extractor_head(ext4, head4, X_rem_t[train_idx], X_met_t[train_idx], Y_tr,
                                             X_rem_t[val_idx], X_met_t[val_idx], Y_val,
                                             epochs=800, use_huber=True)

    model4.eval()
    with torch.no_grad():
        pred4_all = model4(X_rem_t, X_met_t).cpu().numpy().ravel()
    # 反标准化
    pred4_all = pred4_all * std4 + mu4
    pred4 = Y_trend + pred4_all
    r2, rmse, mape = calc_metrics(Y[test_idx], pred4[test_idx])
    results['Detrending-LSTM-1D-CNN'] = (r2, rmse, mape)
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 5. LSTM-1D-CNN (裸DL，无Detrending，无XGBoost)
    # ============================================================
    print('\n=== 5. LSTM-1D-CNN (裸DL) ===')
    ext5 = DualExtractor()
    head5 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(64, 1))
    model5, mu5, std5 = train_extractor_head(ext5, head5, X_rem_t[train_idx], X_met_t[train_idx], Y_tr_raw,
                                             X_rem_t[val_idx], X_met_t[val_idx], Y_val_raw,
                                             epochs=800, use_huber=True)

    model5.eval()
    with torch.no_grad():
        pred5 = model5(X_rem_t, X_met_t).cpu().numpy().ravel()
    pred5 = pred5 * std5 + mu5
    r2, rmse, mape = calc_metrics(Y[test_idx], pred5[test_idx])
    results['LSTM-1D-CNN'] = (r2, rmse, mape)
    print(f'  R²={r2:.4f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%')

    # ============================================================
    # 汇总
    # ============================================================
    print('\n' + '=' * 70)
    print(f'{"Ablation Model":40s} {"R²":>8s} {"RMSE":>10s} {"MAPE":>8s}')
    print('-' * 70)
    for name, (r2, rmse, mape) in results.items():
        print(f'{name:40s} {r2:>8.4f} {rmse:>10.2f} {mape:>7.2f}%')

    print('\nPaper targets (Table 11):')
    print(f'{"""LSTM-1D-CNN-XGBoost""":40s} {"0.6859":>8s} {"508.14":>10s} {"6.24":>7s}%')
    print(f'{"""Detrending-1D-CNN-XGBoost""":40s} {"0.7168":>8s} {"482.54":>10s} {"7.09":>7s}%')
    print(f'{"""Detrending-LSTM-XGBoost""":40s} {"0.4709":>8s} {"659.52":>10s} {"8.89":>7s}%')
    print(f'{"""Detrending-LSTM-1D-CNN""":40s} {"0.7883":>8s} {"417.20":>10s} {"5.27":>7s}%')
    print(f'{"""LSTM-1D-CNN""":40s} {"0.0257":>8s} {"894.99":>10s} {"12.41":>7s}%')

    np.savez(os.path.join(OUT_DIR, 'ablation_results.npz'), results=results)
    with open(os.path.join(OUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump({k: {'R2': round(v[0], 4), 'RMSE': round(v[1], 2), 'MAPE': round(v[2], 2)}
                   for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {OUT_DIR}/')

if __name__ == '__main__':
    run_ablation()
