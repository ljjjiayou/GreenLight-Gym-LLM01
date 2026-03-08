import pandas as pd
import os
import numpy as np
from datetime import datetime


def add_headers_to_csv():
    """
    为原始数据文件添加标准化的表头（列名）
    """
    # 定义数据文件路径
    data_dir = 'data/bleiswijk'
    hps_file = 'dataHPS.csv'
    led_file = 'dataLED.csv'

    # 定义数据集的列名
    column_names = [
        'Time',  # 时间
        'global radiation',  # 全球辐射
        'air temperature',  # 空气温度
        'RH',  # 相对湿度
        'wind speed',  # 风速
        'IndoorTemp',  # 室内温度
        'IndoorVPD',  # 室内水汽压亏缺 (VPD)
        'ThScrPos',  # 热遮阳网位置 (0-100)
        'BlScrPos',  # 黑体遮阳网位置 (0-100)
        'LeeSideVent',  # 背风侧通风口开度
        'WindSideVent',  # 迎风侧通风口开度
        'PipeTemp',  # 加热管道温度
        'GrowPipeTemp',  # 生长管道温度
        'TopLight',  # 顶部补光强度
        'Interlight',  # 植株间补光强度
        'Unused1', 'Unused2', 'Unused3', 'Unused4', 'Unused5', 'Unused6', 'Unused7',  # 未使用列
        'Co2Injection',  # 二氧化碳注入
        'CO2 concentration',  # 二氧化碳浓度
    ]

    # 处理 HPS (高压钠灯) 文件
    hps_path = os.path.join(data_dir, hps_file)
    if os.path.exists(hps_path):
        df_hps = pd.read_csv(hps_path, header=None)
        df_hps.columns = column_names
        print(f"已处理 {hps_file}")

    # 处理 LED 文件
    led_path = os.path.join(data_dir, led_file)
    if os.path.exists(led_path):
        df_led = pd.read_csv(led_path, header=None)
        df_led.columns = column_names
        print(f"已处理 {led_file}")
    return df_hps, df_led


def split_weather_controls(df):
    """
    将数据框拆分为天气变量和控制变量

    参数:
        df: 包含所有变量的输入数据框

    返回:
        weather_df: 仅包含天气变量的数据框
        controls_df: 仅包含控制变量的数据框
    """
    # 根据管道温度是否高于空气温度创建锅炉控制逻辑 (uBoil)
    df['uBoil'] = (df['PipeTemp'] > df['air temperature']).astype(float)

    # 天气变量列表
    weather_vars = [
        'Time',
        'global radiation',
        'air temperature',
        'RH',
        'wind speed',
    ]

    # 控制变量列表
    control_vars = [
        'uBoil',
        'ThScrPos',
        'BlScrPos',
        'LeeSideVent',
        'WindSideVent',
        'PipeTemp',
        'GrowPipeTemp',
        'TopLight',
        'Interlight',
        'Co2Injection'
    ]

    weather_df = df[weather_vars].copy()
    controls_df = df[control_vars].copy()

    return weather_df, controls_df


def compute_sky_temp(air_temp, cloud):
    """
    根据空气温度和云量计算天空温度。
    参数:
        air_temp: 空气温度 (°C)
        cloud: 云量 (0-1)
    返回:
        sky_temp: 天空温度 (°C)
    """
    sigma = 5.67e-8  # 斯特藩-玻尔兹曼常数
    C2K = 273.15  # 摄氏度转开尔文的常数

    ld_clear = 213 + 5.5 * air_temp  # 方程 5.26
    eps_clear = ld_clear / (sigma * (air_temp + C2K) ** 4)  # 方程 5.22
    eps_cloud = (1 - 0.84 * cloud) * eps_clear + 0.84 * cloud  # 方程 5.32
    ld_cloud = eps_cloud * sigma * (air_temp + C2K) ** 4  # 方程 5.22
    sky_temp = (ld_cloud / sigma) ** (0.25) - C2K  # 方程 5.22，此处假设发射率为 1
    return sky_temp


def format_weather_df(weather_df, seconds, interpolated_cloud):
    """
    格式化天气数据框以符合模型要求的列名和单位

    参数:
        weather_df: 包含原始数据的天气数据框
        seconds: 起始时刻的秒数
        interpolated_cloud: 插值后的云量数据
    """
    formatted_df = pd.DataFrame()

    # 将时间从天转换为秒，采样间隔为 300 秒 (5分钟)
    formatted_df['time'] = np.arange(seconds, seconds + len(weather_df) * 300, 300)

    # 复制匹配的列
    formatted_df['global radiation'] = weather_df['global radiation']
    formatted_df['wind speed'] = weather_df['wind speed']
    formatted_df['air temperature'] = weather_df['air temperature']

    # 计算天空温度
    formatted_df['sky temperature'] = compute_sky_temp(weather_df['air temperature'], interpolated_cloud)

    # 为未知变量添加空列
    formatted_df['??'] = 0.0

    # 添加常数级别的室外 CO2 浓度 (400 ppm)
    formatted_df['CO2 concentration'] = 400.0

    # 从秒数提取天数
    formatted_df['day number'] = formatted_df['time'] / 24 / 3600

    # 复制相对湿度
    formatted_df['RH'] = weather_df['RH']

    return formatted_df


def compute_seconds_in_2009():
    """计算 2009 年 1 月 1 日到实验开始日期之间的总秒数"""
    start = datetime(2009, 1, 1, 0, 0)  # 2009年元旦
    end = datetime(2009, 10, 19, 15, 15)  # 实验起始时间

    time_diff = end - start
    seconds = time_diff.total_seconds()
    print(f"{start} 到 {end} 之间的秒数: {seconds}")
    return seconds


def extract_cloud_cover(time_column):
    """
    从鹿特丹的 .mat 文件加载云量数据，并提取与给定时间戳对应的值。
    """
    try:
        from scipy.io import loadmat
        import numpy as np

        # 加载包含 2009-2012 年鹿特丹云量数据的 .mat 文件
        cloud_data = loadmat('data/bleiswijk/cloudRotterdam2009_2012.mat')["cloudRotterdam2009_2012"]

        # 从文件中提取时间轴和云量数值
        cloud_time = cloud_data[:, 0]
        cloud_cover = cloud_data[:, 1]

        # 将云量数据插值到温室数据的时间戳上
        from scipy.interpolate import interp1d
        f = interp1d(cloud_time, cloud_cover, bounds_error=False, fill_value='extrapolate')
        interpolated_cloud = f(time_column)

        return interpolated_cloud

    except Exception as e:
        print(f"加载云量数据时出错: {str(e)}")
        # 如果加载失败，返回全零数组
        return np.zeros(len(time_column))


def format_controls_df(controls_df):
    """
    格式化控制变量数据框，将百分比转换为 0-1 范围，并对齐列名
    """
    np_controls = np.zeros((len(controls_df), 7))
    np_controls[:, 0] = controls_df['uBoil']
    np_controls[:, 1] = controls_df['Co2Injection']
    np_controls[:, 2] = controls_df['ThScrPos'] / 100  # 转换百分比为 0-1
    np_controls[:, 3] = 0.5 * (controls_df['WindSideVent'] + controls_df['LeeSideVent']) / 100  # 平均风速侧通风
    np_controls[:, 4] = controls_df['TopLight'] / 100
    np_controls[:, 5] = controls_df['BlScrPos'] / 100
    np_controls[:, 6] = controls_df['PipeTemp']

    # 检查控制参数中是否存在缺失值 (NaN)
    for i, control in enumerate(
            ['Boiler', 'CO2', 'Thermal Screen', 'Ventilation', 'Top Light', 'Blackout Screen', 'PipeTemp']):
        nan_count = np.isnan(np_controls[:, i]).sum()
        if nan_count > 0:
            print(f"警告: 在 {control} 控制中发现 {nan_count} 个缺失值")
            # 将缺失值填充为 0 避免计算出错
            np_controls[:, i] = np.nan_to_num(np_controls[:, i], 0)

    df_controls = pd.DataFrame(np_controls, columns=['Boiler', 'CO2', 'Thermal Screen', 'Ventilation', 'Top Light',
                                                     'Blackout Screen', 'PipeTemp'])
    return df_controls


def process_time_data(df):
    """
    处理时间数据：
    1. 为 2010 年的数据（第 365 天以后）创建副本
    2. 调整 2010 年数据的时间戳起点
    3. 过滤并保留 2009 年的数据
    """
    # 为 2010 年数据创建副本 (天数 >= 365)
    df_2010 = df[df['day number'] >= 365].copy()
    # 调整 2010 年的天数和秒数（减去 365 天对应的数值）
    df_2010['day number'] = df_2010['day number'] - 365
    df_2010['time'] = df_2010['time'] - (365 * 86400)  # 一天 86400 秒

    # 过滤 2009 年数据
    df_2009 = df[(df['day number'] < 365)].copy()

    return df_2009, df_2010