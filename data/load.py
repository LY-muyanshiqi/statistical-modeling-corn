```python
import pandas as pd
import numpy as np

df_yield_wide = pd.read_csv('河南省各市玉米产量(2002-2024).csv')
df_area_wide = pd.read_csv('河南省各市玉米播种面积(2002-2024).csv')
df_features = pd.read_csv('Henan_Corn_HighPrecision_2002_2024_filled.csv')

df_yield = df_yield_wide.melt(id_vars=['地区'], var_name='年份', value_name='产量')
df_area = df_area_wide.melt(id_vars=['地区'], var_name='年份', value_name='播种面积')

df_yield['年份'] = df_yield['年份'].astype(int)
df_area['年份'] = df_area['年份'].astype(int)

df_target = pd.merge(df_yield, df_area, on=['地区', '年份'])

df_target['实际单产(Y)'] = (df_target['产量'] / df_target['播种面积']) * 10000

df_target.rename(columns={'地区': '城市'}, inplace=True)

remote_cols = ['NDVI', 'EVI', 'LAI', 'FPAR']
meteo_cols = ['蒸散发ET(mm)', '平均气温(℃)', '8天累计降水(mm)', '8天累计辐射(MJ)', '土壤水分(m3/m3)']

df_features = df_features.sort_values(by=['城市', '年份', '时间步(Step)'])
unique_samples = df_features[['城市', '年份']].drop_duplicates().reset_index(drop=True)

num_samples = len(unique_samples)
num_timesteps = 11

X_remote = np.zeros((num_samples, num_timesteps, len(remote_cols)))
X_meteo = np.zeros((num_samples, num_timesteps, len(meteo_cols)))
Y = np.zeros((num_samples,))

for i, row in unique_samples.iterrows():
    city, year = row['城市'], row['年份']
    sample_data = df_features[(df_features['城市'] == city) & (df_features['年份'] == year)]

    X_remote[i, :, :] = sample_data[remote_cols].values
    X_meteo[i, :, :] = sample_data[meteo_cols].values

    target_val = df_target[(df_target['城市'] == city) & (df_target['年份'] == year)]['实际单产(Y)'].values
    Y[i] = target_val[0] if len(target_val) > 0 else np.nan

valid_idx = ~np.isnan(Y)
X_remote, X_meteo, Y = X_remote[valid_idx], X_meteo[valid_idx], Y[valid_idx]

np.save('X_remote.npy', X_remote)
np.save('X_meteo.npy', X_meteo)
np.save('Y_target.npy', Y)

unique_samples_valid = unique_samples[valid_idx]
unique_samples_valid.to_csv('Sample_Info_414.csv', index=False, encoding='utf-8-sig')
```