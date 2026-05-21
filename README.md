# 全国大学生统计建模大赛 · 遥感玉米产量预测

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-16a34a?style=flat-square)](https://xgboost.readthedocs.io/)

> 把脉中原粮仓：多源遥感驱动深度学习的玉米产量预测研究 — 51 页论文 · R² = 0.8236 · MAPE = 5.11%

---

## 研究成果

| 指标 | 数值 | 较基线提升 |
|------|------|-----------|
| R² | **0.8236** | ↑ 62.0% |
| RMSE | **380.83 kg/ha** | ↓ 40.1% |
| MAPE | **5.11%** | ↓ 45.9% |

## 模型架构

**Detrending + LSTM-Attention + 1D-CNN + XGBoost 四层融合：**

| 层 | 技术 | 作用 |
|-----|------|------|
| ① 去趋势化 | 线性 Detrending | 分离技术进步趋势产量与气象波动残差 |
| ② 时序建模 | LSTM + 时间注意力 | 提取全生育期长程生长特征（MODIS 11 时间步） |
| ③ 卷积 | 1D-CNN | 捕捉短期极端气象突变（高温/干旱/涝渍） |
| ④ 集成 | XGBoost | 残差重构产量（Huber 损失防梯度爆炸） |

## 6 模型系统对比 + 消融实验 + SHAP 分析

| 模型 | R² | RMSE |
|------|-----|------|
| Lasso（基线） | — | — |
| SVR | — | — |
| Random Forest | — | — |
| XGBoost | — | — |
| LSTM | — | — |
| 1D-CNN | — | — |
| **四层融合（最优）** | **0.8236** | **380.83** |

- 消融实验验证各模块不可替代协同作用
- SHAP：太阳辐射为首要因子（双向效应），EVI/NDVI 为核心正向驱动

## 数据体系

- **时间跨度**：2002-2024 年（23 年长时序）
- **数据源**：GEE 云平台 → MODIS NDVI/EVI/LAI/FPAR/ET + ERA5-Land 气象再分析
- **范围**：河南省 18 市 + China Crop Dataset 玉米分布掩膜
- **特征**：9 核心指标 · 训练 87% / 测试 13%

## 竞赛信息

- **竞赛**：全国大学生统计建模大赛（2026）
- **参赛编号**：TJJM20260416350473
- **团队**：郭正阳、舒艺林、李垚 | 指导教师：郝孟丽
- **产出**：51 页完整论文

## 相关项目

| 项目 | 描述 |
|------|------|
| [PCCP](https://github.com/LY-muyanshiqi/PCCP) | Inception-ResNet-LSTM · R²=0.986 |
| [坝道微医](https://github.com/LY-muyanshiqi/badao-weiyi) | 水利部认定 · 92%诊断精度 |
| [华中杯-VRP](https://github.com/LY-muyanshiqi/huazhong-cup-vrp) | Hybrid-ILS · 278页论文 |
| [蓄能智调](https://github.com/LY-muyanshiqi/pumped-storage-carbon) | LSTM来水预测 · 碳减排优化 |
| [智慧水利](https://github.com/LY-muyanshiqi/smart-water-demo) | LSTM洪水预测 · Flask+Streamlit |
| [个人主页](https://github.com/LY-muyanshiqi/LY-muyanshiqi) | GitHub Profile · 项目总览 |
