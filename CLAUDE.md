# CLAUDE.md — statistical-modeling-corn：统计建模·玉米产量预测

## 项目简介
统计建模竞赛项目：多源遥感数据驱动的玉米产量预测。LSTM+CNN+XGBoost 融合模型，R²=0.82。

## 技术栈 (README 描述)
- Python · PyTorch · XGBoost · scikit-learn
- 遥感数据处理 · 时序建模

## 项目结构
```
01-粮食/                    # 20+ 篇参考文献 PDF
把脉中原粮仓：多源遥感驱动深度学习.pdf  # 51 页完整论文
```

## 关键术语
- **多源遥感**: 卫星遥感 + 气象站 + 土壤等多源数据融合
- **LSTM+CNN+XGBoost**: 时序 + 空间 + 树模型三路融合架构
- **产量预测**: 基于生长季遥感时序数据预测玉米产量

## 注意事项
- ⚠️ 仓库当前无源码，仅含论文和参考文献 PDF
- 如需复现，需从竞赛提交材料中找回源代码
- requirements.txt 已补充 (基于 README 描述的技术栈)
