"""
数据预处理：插值 → 张量构建 → 标准化 → 趋势分解 → 数据集划分
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# 9核心指标
FEATURE_COLS = ['NDVI', 'EVI', 'LAI', 'FPAR', '蒸散发ET(mm)',
                '平均气温(℃)', '8天累计降水(mm)', '8天累计辐射(MJ)', '土壤水分(m3/m3)']
REMOTE_COLS = ['NDVI', 'EVI', 'LAI', 'FPAR']
METEO_COLS = ['蒸散发ET(mm)', '平均气温(℃)', '8天累计降水(mm)', '8天累计辐射(MJ)', '土壤水分(m3/m3)']

def load_and_fill_features():
    """读取九大指标.csv，按城市-年份分组线性插值填充缺失值"""
    csv_path = os.path.join(DATA_DIR, '九大指标.csv')
    df = pd.read_csv(csv_path)

    # 论文使用的是 interpolate 而非邻近值均值填充（附录代码确认）
    df[FEATURE_COLS] = df.groupby(['城市', '年份'])[FEATURE_COLS].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

    # 对首尾仍为NaN的填充为列均值
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

    filled_path = os.path.join(DATA_DIR, 'Henan_Corn_HighPrecision_2002_2024_filled.csv')
    df.to_csv(filled_path, index=False, encoding='utf-8-sig')
    print(f'[1/5] 缺失值填充完成 → {filled_path}')
    return df

def load_yield_data():
    """读取产量和面积CSV，计算实际单产 kg/ha"""
    yield_csv = os.path.join(DATA_DIR, '河南省各市玉米产量(2002-2024).csv')
    area_csv = os.path.join(DATA_DIR, '河南省各市玉米播种面积(2002-2024).csv')

    df_yield = pd.read_csv(yield_csv).melt(id_vars=['地区'], var_name='年份', value_name='产量')
    df_area = pd.read_csv(area_csv).melt(id_vars=['地区'], var_name='年份', value_name='播种面积')

    df_yield['年份'] = df_yield['年份'].astype(int)
    df_area['年份'] = df_area['年份'].astype(int)

    df_target = pd.merge(df_yield, df_area, on=['地区', '年份'])
    df_target['实际单产(Y)'] = (df_target['产量'] / df_target['播种面积']) * 10000  # kg/ha
    df_target.rename(columns={'地区': '城市'}, inplace=True)

    print(f'[2/5] 产量数据加载完成 → {len(df_target)} 条 (18市 × 23年)')
    return df_target

def build_tensors(df_features, df_target):
    """构建3D张量: X_remote (N,11,4), X_meteo (N,11,5), Y (N,)"""
    df_features = df_features.sort_values(by=['城市', '年份', '时间步(Step)'])
    unique_samples = df_features[['城市', '年份']].drop_duplicates().reset_index(drop=True)

    N, T = len(unique_samples), 11
    X_remote = np.zeros((N, T, len(REMOTE_COLS)))
    X_meteo = np.zeros((N, T, len(METEO_COLS)))
    Y = np.zeros((N,))
    sample_info = []

    for i, row in unique_samples.iterrows():
        city, year = row['城市'], row['年份']
        sd = df_features[(df_features['城市'] == city) & (df_features['年份'] == year)]
        if len(sd) < T:
            continue  # 跳过数据不完整的样本

        X_remote[i, :, :] = sd[REMOTE_COLS].values
        X_meteo[i, :, :] = sd[METEO_COLS].values
        tv = df_target[(df_target['城市'] == city) & (df_target['年份'] == year)]['实际单产(Y)'].values
        Y[i] = tv[0] if len(tv) > 0 else np.nan
        sample_info.append({'城市': city, '年份': year})

    # 移除NaN (产量数据缺失的年份)
    valid_idx = ~np.isnan(Y)
    X_remote, X_meteo, Y = X_remote[valid_idx], X_meteo[valid_idx], Y[valid_idx]

    print(f'[3/5] 张量构建完成 → X_remote: {X_remote.shape}, X_meteo: {X_meteo.shape}, Y: {Y.shape}')
    return X_remote, X_meteo, Y, unique_samples[valid_idx].reset_index(drop=True)

def standardize(X_remote, X_meteo, Y, train_idx):
    """Z-score标准化，基于训练集统计量"""
    # 遥感特征 — 按特征维度标准化
    N, T, D_r = X_remote.shape
    X_rem_s = X_remote.copy()
    scalers_rem = []
    for d in range(D_r):
        vals = X_remote[:, :, d]
        mu, std = vals[train_idx].mean(), vals[train_idx].std()
        X_rem_s[:, :, d] = (vals - mu) / (std + 1e-8)
        scalers_rem.append((mu, std))

    # 气象特征 — 按特征维度标准化
    N, T, D_m = X_meteo.shape
    X_met_s = X_meteo.copy()
    scalers_met = []
    for d in range(D_m):
        vals = X_meteo[:, :, d]
        mu, std = vals[train_idx].mean(), vals[train_idx].std()
        X_met_s[:, :, d] = (vals - mu) / (std + 1e-8)
        scalers_met.append((mu, std))

    # 目标标准化
    y_mean, y_std = Y[train_idx].mean(), Y[train_idx].std()
    Y_s = (Y - y_mean) / (y_std + 1e-8)

    print(f'[4/5] 标准化完成')
    return X_rem_s, X_met_s, Y_s, y_mean, y_std

def detrend(Y, sample_info, train_idx):
    """按城市线性去趋势化，仅用训练集拟合"""
    Y_trend = np.zeros_like(Y)
    Y_residual = np.zeros_like(Y)
    city_models = {}

    for city in sample_info['城市'].unique():
        c_mask = sample_info['城市'] == city
        c_train_mask = c_mask & (sample_info['年份'] <= 2021)  # 训练集: 2002-2021 (论文实际用≤2021)

        yrs_all = sample_info.loc[c_mask, '年份'].values.reshape(-1, 1)
        yrs_train = sample_info.loc[c_train_mask, '年份'].values.reshape(-1, 1)
        y_train_city = Y[c_train_mask]

        if len(yrs_train) > 1:
            lr = LinearRegression().fit(yrs_train, y_train_city)
            Y_trend[c_mask] = lr.predict(yrs_all)
            city_models[city] = (lr.coef_[0], lr.intercept_)

    Y_residual = Y - Y_trend
    print(f'[5/5] 趋势分解完成 → 18市趋势模型已拟合')
    return Y_trend, Y_residual, city_models

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 缺失值填充
    df_features = load_and_fill_features()

    # 2. 产量数据
    df_target = load_yield_data()

    # 3. 张量构建
    X_remote, X_meteo, Y, sample_info = build_tensors(df_features, df_target)

    # 数据集划分: 2002-2021 训练, 2022 验证, 2023-2024 测试
    train_idx = np.where(sample_info['年份'] <= 2021)[0]
    val_idx = np.where(sample_info['年份'] == 2022)[0]
    test_idx = np.where(sample_info['年份'] >= 2023)[0]

    # 4. 标准化
    X_rem_s, X_met_s, Y_s, y_mean, y_std = standardize(X_remote, X_meteo, Y, train_idx)

    # 5. 趋势分解 (在原始尺度上)
    Y_trend, Y_residual, city_models = detrend(Y, sample_info, train_idx)

    # 保存
    np.save(os.path.join(OUT_DIR, 'X_remote.npy'), X_rem_s)
    np.save(os.path.join(OUT_DIR, 'X_meteo.npy'), X_met_s)
    np.save(os.path.join(OUT_DIR, 'Y.npy'), Y)
    np.save(os.path.join(OUT_DIR, 'Y_s.npy'), Y_s)
    np.save(os.path.join(OUT_DIR, 'Y_trend.npy'), Y_trend)
    np.save(os.path.join(OUT_DIR, 'Y_residual.npy'), Y_residual)
    np.save(os.path.join(OUT_DIR, 'train_idx.npy'), train_idx)
    np.save(os.path.join(OUT_DIR, 'val_idx.npy'), val_idx)
    np.save(os.path.join(OUT_DIR, 'test_idx.npy'), test_idx)
    sample_info.to_csv(os.path.join(OUT_DIR, 'Sample_Info.csv'), index=False, encoding='utf-8-sig')

    # 保存标准化参数
    np.savez(os.path.join(OUT_DIR, 'scaler_params.npz'), y_mean=y_mean, y_std=y_std)

    print(f'\n数据预处理完成!')
    print(f'  训练集: {len(train_idx)} 样本 (2002-2021)')
    print(f'  验证集: {len(val_idx)} 样本 (2022)')
    print(f'  测试集: {len(test_idx)} 样本 (2023-2024)')

    # 输出 city_models 供验证
    for city, (slope, intercept) in city_models.items():
        print(f'  {city}: trend = {slope:.2f}*year + {intercept:.1f}')

if __name__ == '__main__':
    main()
