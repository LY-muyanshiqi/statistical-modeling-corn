import ee
import json
import pandas as pd
import time

PROJECT_ID = 'celtic-fact-493310-a3'
ee.Initialize(project=PROJECT_ID)

try:
    with open('henan.geojson', 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
except FileNotFoundError:
    exit()

ee_features = []
for feat in geojson_data['features']:
    geom = ee.Geometry(feat['geometry'])
    city_name = feat['properties'].get('name', '未知城市')
    ee_feat = ee.Feature(geom, {'city_name': city_name})
    ee_features.append(ee_feat)

henan_cities = ee.FeatureCollection(ee_features)

def get_safe_mean(collection, band_name, factor, rename_to):
    scaled_coll = collection.select(band_name).map(lambda img: img.multiply(factor).rename(rename_to))
    mean_img = scaled_coll.mean()
    dummy = ee.Image.constant(0).mask(0).rename(rename_to)
    return mean_img.addBands(dummy).select([0]).rename(rename_to)

def get_safe_sum(collection, band_name, factor, rename_to):
    scaled_coll = collection.select(band_name).map(lambda img: img.multiply(factor).rename(rename_to))
    sum_img = scaled_coll.sum()
    dummy = ee.Image.constant(0).mask(0).rename(rename_to)
    return sum_img.addBands(dummy).select([0]).rename(rename_to)

def get_safe_temp(collection):
    scaled_coll = collection.select('temperature_2m').map(lambda img: img.subtract(273.15).rename('Mean_Temp_C'))
    return scaled_coll.mean().addBands(ee.Image.constant(0).mask(0).rename('Mean_Temp_C')).select([0])

def get_safe_rad(collection):
    scaled_coll = collection.select('surface_solar_radiation_downwards').map(
        lambda img: img.divide(1000000).rename('Total_Solar_Rad_MJ_m2'))
    return scaled_coll.sum().addBands(ee.Image.constant(0).mask(0).rename('Total_Solar_Rad_MJ_m2')).select([0])

all_features_data = []

for year in range(2002, 2025):
    mask_asset_id = f'projects/{PROJECT_ID}/assets/henan-crops-{year}'
    try:
        ccd_img = ee.Image(mask_asset_id)
        corn_mask = ccd_img.eq(6)
    except Exception as e:
        continue

    start_season = ee.Date.fromYMD(year, 7, 1)

    for step in range(11):
        start_date = start_season.advance(step * 8, 'day')
        end_date = start_date.advance(8, 'day')

        veg_coll = ee.ImageCollection('MODIS/061/MYD13Q1').filterDate(start_date.advance(-16, 'day'), end_date)
        ndvi = get_safe_mean(veg_coll, 'NDVI', 0.0001, 'NDVI_Mean')
        evi = get_safe_mean(veg_coll, 'EVI', 0.0001, 'EVI_Mean')

        lai_fpar_coll = ee.ImageCollection('MODIS/061/MYD15A2H').filterDate(start_date, end_date)
        lai = get_safe_mean(lai_fpar_coll, 'Lai_500m', 0.1, 'LAI_Mean')
        fpar = get_safe_mean(lai_fpar_coll, 'Fpar_500m', 0.01, 'FPAR_Mean')

        et_coll = ee.ImageCollection('MODIS/061/MOD16A2GF').filterDate(start_date, end_date)
        et = get_safe_sum(et_coll, 'ET', 0.1, 'Total_ET_mm')

        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start_date, end_date)
        temp = get_safe_temp(era5)
        precip = get_safe_sum(era5, 'total_precipitation', 1000, 'Total_Precip_mm')
        rad = get_safe_rad(era5)
        soil_moisture = get_safe_mean(era5, 'volumetric_soil_water_layer_1', 1.0, 'Soil_Moisture_L1')

        combined_img = ee.Image([ndvi, evi, lai, fpar, et, temp, precip, rad, soil_moisture])
        final_img = combined_img.updateMask(corn_mask)

        stats = final_img.reduceRegions(
            collection=henan_cities,
            reducer=ee.Reducer.mean(),
            scale=250
        )

        try:
            features = stats.getInfo()['features']
            for feat in features:
                props = feat['properties']
                all_features_data.append({
                    '年份': year,
                    '时间步(Step)': step + 1,
                    '城市': props.get('city_name', '未知'),
                    'NDVI': props.get('NDVI_Mean'),
                    'EVI': props.get('EVI_Mean'),
                    'LAI': props.get('LAI_Mean'),
                    'FPAR': props.get('FPAR_Mean'),
                    '蒸散发ET(mm)': props.get('Total_ET_mm'),
                    '平均气温(℃)': props.get('Mean_Temp_C'),
                    '8天累计降水(mm)': props.get('Total_Precip_mm'),
                    '8天累计辐射(MJ)': props.get('Total_Solar_Rad_MJ_m2'),
                    '土壤水分(m3/m3)': props.get('Soil_Moisture_L1')
                })
        except Exception as e:
            continue

df = pd.DataFrame(all_features_data)
df = df.sort_values(by=['城市', '年份', '时间步(Step)']).reset_index(drop=True)
output_name = 'Henan_Corn_HighPrecision_2002_2024.csv'
df.to_csv(output_name, index=False, encoding='utf-8-sig')