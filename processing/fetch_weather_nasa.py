#!/usr/bin/env python3
"""
fetch_and_process_power.py

下载、处理并插值 NASA POWER API 的每小时天气数据至 5 分钟分辨率。
支持全球城市，支持多年份范围。

步骤：
  1. 将城市名称地理编码为经纬度。
  2. 获取每年的逐小时数据。
  3. 识别 CSV 中的 "YEAR,MO,DY,HR" 表头行。
  4. 解析日期/时间并构建时间序列索引。
  5. 将全球辐射单位从 MJ/h/m² 转换为 W/m²。
  6. 根据气温和云量计算天空温度。
  7. 计算自年初以来的秒数和天数。
  8. 填充恒定的 CO₂ 浓度。
  9. 使用 PCHIP 算法重采样至 5 分钟间隔。
 10. 保存处理后的 CSV 至 weather/<城市>/<年份>.csv。
"""

import os
import argparse
import requests
import pandas as pd
import calendar
from geopy.geocoders import Nominatim
from io import StringIO
from time import sleep
from requests.exceptions import HTTPError

# 用于 PCHIP 插值
from scipy.interpolate import PchipInterpolator
from timezonefinder import TimezoneFinder
import numpy as np
import pytz


def compute_sky_temp(air_temp_c, cloud_frac):
    """
    根据空气温度和云量计算天空温度。
    """
    sigma = 5.67e-8
    C2K = 273.15
    # 计算晴天长波辐射下行
    ld_clear = 213 + 5.5 * air_temp_c
    eps_clear = ld_clear / (sigma * (air_temp_c + C2K) ** 4)
    # 考虑云量的修正系数
    eps_cloud = (1 - 0.84 * cloud_frac) * eps_clear + 0.84 * cloud_frac
    ld_cloud = eps_cloud * sigma * (air_temp_c + C2K) ** 4
    sky_temp_k = (ld_cloud / sigma) ** 0.25
    return sky_temp_k - C2K


def get_coordinates(city_name):
    """
    将城市名称转换为经纬度坐标。
    """
    geolocator = Nominatim(user_agent="weather_fetcher")
    loc = geolocator.geocode(city_name)
    if loc is None:
        raise ValueError(f"无法获取 '{city_name}' 的地理坐标")
    return loc.latitude, loc.longitude


def _parse_power_csv(text):
    """
    定位 CSV 表头行，读取数据并构建日期时间索引。
    """
    lines = text.splitlines()
    header_idx = next((i for i, L in enumerate(lines) if L.startswith("YEAR,MO,DY,HR")), None)
    if header_idx is None:
        raise ValueError("未能找到 'YEAR,MO,DY,HR' 数据表头")
    data = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data))
    # 构建时间列
    df['time'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MO'],
        'day': df['DY'],
        'hour': df['HR']
    })
    df.set_index('time', inplace=True)
    return df


def fetch_process_year(lat, lon, year, city_dir, parameters, community="AG"):
    """
    获取、处理、插值并保存单年份的数据。
    """
    # 根据经纬度确定当地时区
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    if timezone_str is None:
        raise ValueError("无法确定给定坐标的时区")
    tz = pytz.timezone(timezone_str)

    # 确定当地时间的年初和年末
    local_start = tz.localize(pd.Timestamp(f"{year}-01-01 00:00:00"))
    local_end = tz.localize(pd.Timestamp(f"{year}-12-31 23:00:00"))
    # 转换为 UTC 以匹配 API 的查询窗口
    utc_start = local_start.astimezone(pytz.UTC)
    utc_end = local_end.astimezone(pytz.UTC)

    # 获取 UTC 偏移量（小时）
    local_dt = tz.localize(pd.Timestamp(2001, 1, 1, 0, 0, 0))
    offset_td = local_dt.utcoffset()  # 时间差对象
    offset_hours = offset_td.total_seconds() // 3600.0  # 转换为小时

    # 为 NASA POWER 格式化 YYYYMMDD 字符串
    if offset_hours > 0 and year == 2001:
        start_date = (utc_start.date() + pd.Timedelta(days=1)).strftime('%Y%m%d')
    else:
        start_date = utc_start.date().strftime('%Y%m%d')
    end_date = utc_end.date().strftime('%Y%m%d')

    def _build_url(start, end):
        """构建 API 请求 URL"""
        return (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?start={start}&end={end}"
            f"&latitude={lat}&longitude={lon}"
            f"&community={community}"
            f"&parameters={','.join(parameters)}"
            "&format=CSV&header=true&time-standard=utc"
        )

    print(f"正在获取 {year} 全年数据...")
    url_year = _build_url(start_date, end_date)
    try:
        resp = requests.get(url_year)
        resp.raise_for_status()
        df = _parse_power_csv(resp.text)
    except HTTPError as e:
        # 如果全年获取失败（服务器错误），尝试按月份逐个获取
        if e.response is not None and 500 <= e.response.status_code < 600:
            print(f"  全年获取失败 (状态码 {e.response.status_code})，正在尝试按月获取...")
            monthly = []
            for m in range(1, 13):
                last = calendar.monthrange(year, m)[1]
                start = f"{year}{m:02d}01"
                end = f"{year}{m:02d}{last:02d}"
                try:
                    sleep(1)
                    r = requests.get(_build_url(start, end))
                    r.raise_for_status()
                    monthly.append(_parse_power_csv(r.text))
                except Exception as me:
                    print(f"    月份 {m:02d} 获取失败: {me}")
            if not monthly:
                raise RuntimeError(f"无法获取 {year} 的任何数据")
            df = pd.concat(monthly).sort_index()
        else:
            raise

    # 单位转换与特征计算
    # 将 MJ/m2/h 转换为 W/m2
    df['ALLSKY_SFC_SW_DWN'] *= (1e6 / 3600)
    # 计算云量比例 (0-1)
    df['cloud_frac'] = df['CLOUD_AMT'] / 100.0
    # 计算天空温度
    df['sky_temperature'] = df.apply(lambda r: compute_sky_temp(r['T2M'], r['cloud_frac']), axis=1)

    n = len(df)
    # 处理由于时区偏移导致的数据切片
    if offset_hours > 0:
        # 跳过前面的偏移量，截取中间的 365/366 天
        start_idx = 24 - offset_hours
        end_idx = n - (offset_hours)
    elif offset_hours < 0:
        start_idx = -offset_hours
        end_idx = n - (24 + offset_hours)
    else:
        start_idx, end_idx = 0, n

    df = df.iloc[int(start_idx):int(end_idx)]

    # 使用 PCHIP 插值到 5 分钟频率
    df5 = df.resample('5min').asfreq()
    cols_interp = ['ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M', 'sky_temperature', 'cloud_frac', 'RH2M']
    for col in cols_interp:
        series = df[col]
        # 使用时间戳作为 X 轴进行三次插值
        pchip = PchipInterpolator(series.index.astype(int), series.values)
        df5[col] = pchip(df5.index.astype(int))

    # 填充恒定 CO2 浓度
    df5['CO2_ppm'] = 400.0
    # 重新计算自开始以来的秒数（5分钟=300秒）
    df5['seconds_since_start'] = np.arange(len(df5)) * 300
    df5['day_number'] = df5.index.dayofyear

    # 准备输出列名，确保与 GreenLight 模型完全匹配
    out = df5[[
        'seconds_since_start', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M',
        'sky_temperature', 'cloud_frac', 'CO2_ppm', 'day_number', 'RH2M'
    ]].copy()
    out.columns = [
        'time', 'global radiation', 'wind speed', 'air temperature',
        'sky temperature', 'cloud cover', 'CO2 concentration',
        'day number', 'RH'
    ]

    # 保存文件
    filepath = os.path.join(city_dir, f"{year}.csv")
    out.to_csv(filepath, index=False)
    print(f"已保存处理后的插值数据至 {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("city", help="城市名称，例如 'Beijing, China'")
    parser.add_argument("--output-dir", default="weather")
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2020)
    args = parser.parse_args()

    # 处理文件名，移除空格
    key = args.city.split(',')[0].capitalize().replace(' ', '_')
    base = os.path.join(args.output_dir, key)
    os.makedirs(base, exist_ok=True)

    print(f"正在对 {args.city} 进行地理编码...")
    lat, lon = get_coordinates(args.city)
    print(f"经纬度: {lat:.4f}, {lon:.4f}\n")

    # NASA POWER 观测参数列表
    params = ["ALLSKY_SFC_SW_DWN", "T2M", "WS2M", "RH2M", "CLOUD_AMT"]
    for yr in range(args.start_year, args.end_year + 1):
        try:
            fetch_process_year(lat, lon, yr, base, params)
            sleep(1)  # 避免请求频率过快
        except Exception as e:
            print(f"处理 {yr} 年时出错: {e}")


if __name__ == "__main__":
    main()