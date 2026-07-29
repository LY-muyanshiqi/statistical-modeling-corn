"""
模型定义：MD_STFF_Extractor (LSTM+Attention + 1D-CNN 双通道), PureLSTM, PureCNN
"""
import torch
import torch.nn as nn


class MD_STFF_Extractor(nn.Module):
    """多源时空特征提取器 — 论文核心架构
    左通道 LSTM+Attention: 输入 (N, 11, 4) 遥感特征 → 输出 64-dim
    右通道 1D-CNN:         输入 (N, 5, 11) 气象特征 → 输出 64-dim
    融合: concat → 128-dim 特征向量
    """
    def __init__(self, remote_dim=4, meteo_dim=5, hidden_dim=64):
        super().__init__()
        # 左通道 — LSTM + 时间注意力
        self.lstm = nn.LSTM(input_size=remote_dim, hidden_size=hidden_dim,
                           num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 右通道 — 1D-CNN
        self.conv = nn.Sequential(
            nn.Conv1d(meteo_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x_rem, x_met):
        # 遥感通道: LSTM + 时间注意力
        lstm_out, _ = self.lstm(x_rem)                    # (N, 11, 64)
        attn_scores = self.attention(lstm_out)              # (N, 11, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (N, 11, 1)
        feat_rem = torch.sum(attn_weights * lstm_out, dim=1)  # (N, 64)

        # 气象通道: 1D-CNN
        feat_met = self.conv(x_met.transpose(1, 2)).squeeze(-1)  # (N, 64)

        # 特征融合
        return torch.cat([feat_rem, feat_met], dim=1)  # (N, 128)


class PureLSTM(nn.Module):
    """纯LSTM基准模型 — 9维全特征输入，无注意力"""
    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)         # x: (N, 11, 9)
        return self.fc(out[:, -1, :])  # 取最后时间步


class PureCNN(nn.Module):
    """纯1D-CNN基准模型 — 9通道输入"""
    def __init__(self, in_channels=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.conv(x.transpose(1, 2)).squeeze(-1)  # x: (N, 9, 11) → (N, 64)
        return self.fc(feat)
