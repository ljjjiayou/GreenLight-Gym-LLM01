# 导入必要的类型注解模块
from typing import Dict, Tuple, Any
# 导入路径拼接函数
from os.path import join
# 导入日期时间处理模块
from datetime import datetime, timedelta
# 导入深拷贝函数
from copy import deepcopy

# 导入yaml解析库
import yaml
# 导入数值计算库
import numpy as np
# 导入数据分析库
import pandas as pd
# 导入PCHIP插值器（保形分段三次Hermite插值多项式）
from scipy.interpolate import PchipInterpolator

def load_model_hyperparams(algorithm: str, env_id: str) -> Dict[str, Any]:
    """
    加载指定算法和环境的模型超参数
    
    参数:
        algorithm: 算法名称（用于拼接配置文件路径）
        env_id: 环境ID（用于从配置文件中提取对应环境的超参数）
    
    返回:
        model_hyperparams: 指定环境的模型超参数字典
    """
    # 拼接超参数配置文件路径（gl_gym/configs/agents/下的algorithm.yml）
    with open(join("gl_gym/configs/agents/", algorithm + ".yml"), "r") as f:
        # 加载yaml配置文件
        params = yaml.load(f, Loader=yaml.FullLoader)
    # 提取指定环境的超参数
    model_hyperparams = params[env_id]
    return model_hyperparams


def load_env_params(env_id: str, path: str) -> Tuple[Dict, Dict]:
    '''
    加载环境参数，返回通用参数和特定环境参数
    
    参数:
        env_id (str): 环境ID（如GreenLightEnv或其子环境）
        path (str): yaml配置文件所在路径
    
    返回:
        Tuple[Dict, Dict]: 
            - env_base_params: GreenLightEnv基类的通用参数
            - env_specific_params: 特定环境的参数（若env_id为GreenLightEnv则返回空字典）
    '''
    # 拼接环境配置文件路径并读取
    with open(join(path, env_id + ".yml"), "r", encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # 区分通用参数和特定环境参数
    if env_id != "GreenLightEnv":
        env_specific_params = params[env_id]
    else:
        env_specific_params = {}

    env_base_params = params["GreenLightEnv"]

    return env_base_params, env_specific_params


def get_starting_date(df):
    """
    从数据框中提取生长阶段的起始日期相关信息
    
    参数:
        df: 包含DateTime列的DataFrame
    
    返回:    
        start_day: 模拟起始日（一年中的第几天，从0开始）
        days_sim: 模拟总天数（从起始日到数据最后一天的天数+1）
        year: 模拟起始年份
    """
    # 获取起始日期在一年中的第几天（tm_yday从1开始，减1转为0开始）
    start_day = df['DateTime'].iloc[0].timetuple().tm_yday - 1
    # 计算模拟总天数
    days_sim = df['DateTime'].iloc[-1].timetuple().tm_yday - start_day + 1
    # 获取起始年份
    year = df['DateTime'].iloc[0].year
    return start_day, days_sim, year

def get_end_date_veg(df):
    """
    获取营养生长阶段的结束日期（一年中的第几天）
    
    参数:
        df: 包含DateTime列的DataFrame
    
    返回:
        end_date: 营养生长阶段最后一天在一年中的第几天（tm_yday，从1开始）
    """
    return df['DateTime'].iloc[-1].timetuple().tm_yday

def start_dmc(x):
    """
    计算初始DMC（Drought Moisture Code，干旱湿度码）值
    线性计算公式：0.0000691*x + 0.0564
    
    参数:
        x: 输入变量（具体物理意义需结合业务场景）
    
    返回:
        计算得到的初始DMC值
    """
    return 0.0000691*x + 0.0564

def read_json_file(fname: str) -> str:
    '''	
    读取JSON文件内容并以字符串形式返回
    
    参数:
        fname (str): JSON文件的路径
    
    返回:
        param_str (str): JSON文件的文本内容
    '''
    with open(fname, 'r') as file:
        param_str = file.read()
    return param_str

@np.vectorize
def excel_to_datetime(excel_serial_date: np.ndarray | float) -> datetime:
    '''	
    将Excel序列号日期转换为Python datetime对象
    （使用np.vectorize装饰器支持数组输入）
    
    参数:
        excel_serial_date (np.ndarray | float): Excel序列号日期（整数+小数，小数表示时分秒）
    
    返回:
        datetime: 转换后的datetime对象（数组输入时返回datetime数组）
    '''
    # Excel的基准日期是1899-12-30
    excel_base_date = datetime(1899, 12, 30)
    # 按天数偏移计算实际日期
    return excel_base_date + timedelta(days=excel_serial_date)

def format_time_date(gh_data):
    '''
    标准化温室数据的时间格式为YYYY-MM-DD HH:MM，并按小时取整
    
    参数:
        gh_data: 包含DateTime列的DataFrame（DateTime为Excel序列号格式）
    
    返回:
        gh_data: 时间格式标准化后的DataFrame
    '''
    # 提取Excel时间列并转换为datetime对象
    excel_times = gh_data['DateTime']
    date_times = excel_to_datetime(excel_times)
    
    # 格式化为YYYY-MM-DD HH:MM字符串
    formatted_times = [dt.strftime('%Y-%m-%d %H:%M') for dt in date_times]
    gh_data['DateTime'] = formatted_times
    
    # 转换为pandas datetime类型并按小时取整（消除分钟级误差）
    gh_data['DateTime'] = pd.to_datetime(gh_data['DateTime']).dt.round('h')
    return gh_data

def compute_potential_growth(nett_fruit_growth):
    """
    计算潜在生长量（果实、叶片、茎干）
    
    参数:
        nett_fruit_growth: 果实净生长量
    
    返回:
        rgFruit: 果实总生长量（净生长量+果实维持呼吸消耗）
        rgLeaves: 叶片生长量
        rgStems: 茎干生长量
    """
    # 果实维持呼吸消耗系数
    fruit_maintenance_respiration = 0.027
    # 果实总生长量 = 净生长量 + 呼吸消耗
    rgFruit = nett_fruit_growth + fruit_maintenance_respiration
    # 叶片生长量计算公式（经验系数）
    rgLeaves = nett_fruit_growth/74*15.8 + 0.031
    # 茎干生长量计算公式（经验系数）
    rgStems = nett_fruit_growth/74*10.2 + 0.032
    return rgFruit, rgLeaves, rgStems

def days2date(timeInDays: float, referenceDate: str):
    """
    将参考日期以来的天数（含小数）转换为具体日期时间（DD-MM-YYYY HH:MM:SS）
    
    参数:
        timeInDays: 参考日期以来的天数（小数，小数部分表示小时、分钟）
        referenceDate: 参考日期，格式为DD-MM-YYYY
    
    返回:
        list[str]: 转换后的日期时间字符串列表，格式为YYYY-MM-DD HH:MM:SS
    """
    # 解析参考日期为datetime对象
    referenceDatetime = datetime.strptime(referenceDate, '%d-%m-%Y')
    
    # 拆分天数的整数和小数部分，转换为小时、分钟
    int_days = np.floor(timeInDays).astype(int)  # 整数天数
    time_component = (timeInDays - int_days) * 24  # 小数部分转小时
    hours = time_component.astype(int)  # 整数小时
    time_component = (time_component - hours) * 60  # 剩余小数转分钟
    minutes = time_component.astype(int)  # 整数分钟
    
    # 计算目标日期时间
    target_datetimes = [
        referenceDatetime + timedelta(days=int(int_day), hours=int(hour), minutes=int(minute)) 
        for int_day, hour, minute in zip(int_days, hours, minutes)
    ]

    # 格式化为指定字符串格式并返回
    return [target_datetime.strftime('%Y-%m-%d %H:%M:%S') for target_datetime in target_datetimes]

def process_weather_data(raw_weather, h: int, nd: int) -> np.ndarray:
    """
    处理原始气象数据：转换为GreenLight模型所需格式，并根据求解器采样频率插值重采样
    
    参数:
        raw_weather: 原始气象数据DataFrame（包含time、global radiation等列）
        h: 求解器采样时间（秒）
        nd: 气象变量数量（最终输出10列）
    
    返回:
        np.ndarray: 插值重采样后的气象数据矩阵，列定义：
            d[0]: iGlob - 全球辐射 [W m^{-2}]
            d[1]: tOut - 室外温度 [°C]    
            d[2]: vpOut - 室外蒸气压 [Pa]
            d[3]: co2Out - 室外CO2浓度 [mg m^{-3}]
            d[4]: wind - 室外风速 [m s^{-1}]
            d[5]: tSky - 天空温度 [°C]
            d[6]: tSoOut - 室外土壤温度 [°C]
            d[7]: dli - 日辐射总量 [MJ m^{-2} day^{-1}]
            d[8]: isDay - 是否为白天 [0,1]（硬阈值）
            d[9]: isDaySmooth - 是否为白天 [0,1]（平滑过渡）
    """
    # 一天的秒数
    c = 86400  

    # 提取时间列（一年中的秒数）并计算数据采样周期
    time = raw_weather["time"].values    # 距年初的秒数
    dt = np.mean(np.diff(time - time[0])) # 数据平均采样周期（秒）
    Ns = raw_weather.shape[0]            # 数据总行数

    # 预分配气象数据矩阵
    weatherData = np.zeros((Ns, nd))                                         
    time = raw_weather["time"].values[:]                               # 重新提取时间列
    weatherData[:, 0] = raw_weather["global radiation"][:]             # 全球辐射
    weatherData[:, 1] = raw_weather["air temperature"][:]              # 室外温度
    vpDensity = rh2vaporDens(weatherData[:, 1], raw_weather["RH"][:])  # 相对湿度转水汽密度
    weatherData[:, 2] = vaporDens2pres(weatherData[:, 1], vpDensity)   # 水汽密度转蒸气压
    # CO2浓度：ppm转mg/m³（乘以1e6转换单位）
    weatherData[:, 3] = co2ppm2dens(weatherData[:, 1], raw_weather["CO2 concentration"]) * 1e6
    weatherData[:, 4] = raw_weather["wind speed"][:]                    # 风速
    weatherData[:, 5] = raw_weather["sky temperature"][:]               # 天空温度
    weatherData[:, 6] = soilTempNl(raw_weather["time"][:])              # 土壤温度（荷兰地区经验公式）
    weatherData[:, 7] = dailLightSum(time, weatherData[:, 0], c)       # 日辐射总量（DLI）
    weatherData[:, 8], weatherData[:, 9] = computeisDay(weatherData[:, 0], dt)  # 白天/黑夜标记

    # 计算求解器所需的采样数（按求解器采样频率h重采样）
    ns = int((dt / h) * (Ns))

    # PCHIP插值（保形插值，避免过冲）
    interpolation = PchipInterpolator(time, weatherData)
    timeRes = np.linspace(time[0], time[-1], ns)  # 生成新的时间轴
    weatherDataResampled = interpolation(timeRes) # 插值得到重采样数据
 
    # 将极小的辐射值置0（消除数值噪声）
    weatherDataResampled[:, 0][weatherDataResampled[:, 0] < 1e-10] = 0

    return weatherDataResampled


def loadWeatherData(weatherDataDir: str, location: str, source: str, growthYear: int,
                    startDay: int, nDays: int, predHorizon: int, h: int, nd: int) -> np.ndarray:
    """
    加载气象数据：读取CSV文件，裁剪时间范围，处理跨年数据，插值重采样为模型所需格式
    
    参数:
        weatherDataDir: 气象数据根目录
        location: 温室位置（用于拼接文件路径）
        source: 数据来源（如KNMI，用于拼接文件路径）
        growthYear: 生长年份（用于拼接文件路径）
        startDay: 模拟起始日（一年中的第几天，0开始）
        nDays: 模拟天数
        predHorizon: 预测时域（天）
        h: 求解器采样时间（秒）
        nd: 气象变量数量（最终输出10列）
    
    返回:
        np.ndarray: 插值重采样后的气象数据矩阵（列定义同process_weather_data）
    """
    # 拼接气象数据文件路径
    weatherDataPath = weatherDataDir + location + "/" + source + str(growthYear) + ".csv"

    # 常量定义
    c = 86400      # 一天的秒数
    CO2_PPM = 400  # 室外CO2浓度默认值（ppm）
    rawWeather = pd.read_csv(weatherDataPath, sep=",")  # 读取CSV数据

    # 计算时间相关索引
    time = rawWeather["time"].values    # 距年初的秒数
    dt = np.mean(np.diff(time - time[0])) # 数据采样周期（秒）
    N0 = int(np.ceil(startDay * c / dt))    # 模拟起始索引
    Ns = int(np.ceil(nDays * c / dt))       # 模拟所需数据量
    Np = int(np.ceil(predHorizon * c / dt)) + 1 # 预测时域所需数据量

    # 检查是否超出当前年份数据长度，若超出则加载下一年数据补充
    if N0 + Ns + Np > len(time):
        rawWeather = expandWeatherData(weatherDataDir, rawWeather, location, source, growthYear, time, dt)

    # 预分配气象数据矩阵（模拟+预测时域）
    weatherData = np.zeros((Ns + Np, nd))                                         
    # 裁剪时间范围：起始索引到（模拟+预测）结束索引
    time = rawWeather["time"].values[N0:N0+Ns+Np]                               
    weatherData[:, 0] = rawWeather["global radiation"][N0:N0+Ns+Np]             # 全球辐射
    weatherData[:, 1] = rawWeather["air temperature"][N0:N0+Ns+Np]              # 室外温度
    vpDensity = rh2vaporDens(weatherData[:, 1], rawWeather["RH"][N0:N0+Ns+Np])  # 相对湿度转水汽密度
    weatherData[:, 2] = vaporDens2pres(weatherData[:, 1], vpDensity)             # 水汽密度转蒸气压
    # CO2浓度：默认400ppm转mg/m³
    weatherData[:, 3] = co2ppm2dens(weatherData[:, 1], CO2_PPM) * 1e6              
    weatherData[:, 4] = rawWeather["wind speed"][N0:N0+Ns+Np]                    # 风速
    weatherData[:, 5] = rawWeather["sky temperature"][N0:N0+Ns+Np]               # 天空温度
    weatherData[:, 6] = soilTempNl(rawWeather["time"][N0:N0+Ns+Np])              # 土壤温度
    weatherData[:, 7] = dailLightSum(time, weatherData[:, 0], c)                 # 日辐射总量
    weatherData[:, 8], weatherData[:, 9] = computeisDay(weatherData[:, 0], dt)   # 白天/黑夜标记

    # 计算求解器所需采样数并插值重采样
    ns = int((dt / h) * (Ns + Np))
    interpolation = PchipInterpolator(time, weatherData)
    timeRes = np.linspace(time[0], time[-1], ns)
    weatherDataResampled = interpolation(timeRes)
 
    # 极小辐射值置0
    weatherDataResampled[:, 0][weatherDataResampled[:, 0] < 1e-10] = 0

    return weatherDataResampled


def expandWeatherData(weatherDataDir: str, rawWeather: pd.DataFrame, location: str, 
                      source: str, growthYear: int, time: np.ndarray, dt: int) -> pd.DataFrame:
    """
    补充跨年气象数据：加载下一年的气象数据并拼接到当前数据后
    
    参数:
        weatherDataDir: 气象数据根目录
        rawWeather: 当前年份的气象数据DataFrame
        location: 温室位置
        source: 数据来源
        growthYear: 当前生长年份
        time: 当前数据的时间列（距年初的秒数）
        dt: 数据采样周期（秒）
    
    返回:
        pd.DataFrame: 拼接后的气象数据（当前年+下一年）
    """
    # 拼接下一年气象数据路径
    weatherDataPath = weatherDataDir + location + "/" + source + str(growthYear + 1) + ".csv"
    # 读取下一年数据
    newRawWeather = pd.read_csv(weatherDataPath, sep=",")
    # 调整下一年数据的时间轴（接续当前数据最后一秒）
    newRawWeather["time"] += time[-1] + dt
    # 拼接数据
    rawWeather = pd.concat([rawWeather, newRawWeather.iloc[:, :]])
    return rawWeather

def computeisDay(rad: np.ndarray, dt: int) -> tuple[np.ndarray, np.ndarray]:
    """
    根据辐射值判断白天/黑夜，生成硬阈值标记和平滑过渡标记
    
    参数:
        rad: 辐射值数组 [W m^{-2}]
        dt: 数据采样周期（秒）
    
    返回:
        tuple[np.ndarray, np.ndarray]:
            - isDay: 硬阈值标记（辐射>0为1，否则0）
            - isDaySmooth: 平滑过渡标记（基于Sigmoid函数，消除昼夜突变）
    """
    # 硬阈值标记：辐射>0为白天（1），否则黑夜（0）
    isDay = (rad > 0) * 1.0
    isDaySmooth = deepcopy(isDay)
    
    # 过渡周期长度（秒转采样数，默认1小时过渡）
    transSize = int(3600 / dt)    # 需为偶数，确保过渡对称
    
    # 生成0-1的线性过渡数组和Sigmoid平滑过渡数组
    trans = np.linspace(0, 1, transSize)
    transSmooth = 1 / (1 + np.exp(-10 * (trans - 0.5)))  # Sigmoid函数，陡峭度10
    sunset = False  # 标记是否处于日落阶段

    # 遍历数据，替换昼夜转换处的标记为过渡值
    for k in range(transSize, len(isDay) - transSize):
        if isDay[k] == 0:
            sunset = False  # 黑夜阶段，重置日落标记
        
        # 日出：从黑夜转白天，替换为上升过渡
        if isDay[k] == 0 and isDay[k + 1] == 1:
            isDay[k - transSize // 2 : k + transSize // 2] = trans
            isDaySmooth[k - transSize // 2 : k + transSize // 2] = transSmooth
        # 日落：从白天转黑夜，替换为下降过渡（仅首次触发）
        elif isDay[k] == 1 and isDay[k + 1] == 0 and not sunset:
            isDay[k - transSize // 2: k + transSize // 2] = 1 - trans
            isDaySmooth[k - transSize // 2: k + transSize // 2] = 1 - transSmooth
            sunset = True  # 标记日落阶段，避免重复处理
    return isDay, isDaySmooth

def dailLightSum(time: np.ndarray, rad: np.ndarray, c: int):
    """
    计算DLI（Daily Light Integral，日辐射总量）
    
    参数:
        time: 时间数组（距年初的秒数）
        rad: 辐射值数组 [W m^{-2}]
        c: 一天的秒数（通常86400）
    
    返回:
        lightSum: 逐时刻的日辐射总量 [MJ m^{-2} day^{-1}]
    """
    # 计算数据采样间隔（秒）
    interval = time[1] - time[0]
    # 时间转换为天数（距年初）
    time = time / c              

    # 初始化为当天0点前的索引
    mnBefore = 0

    # 找到当天0点后的第一个索引（日切换点）
    mnAfter = np.where(np.diff(np.floor(time)) == 1)[0] + 1
    if mnAfter.size == 0:
        mnAfter = len(time)
    else:
        mnAfter = mnAfter[0]
    
    # 预分配日辐射总量数组
    lightSum = np.zeros(len(time))

    # 遍历每个时刻，累加当天的辐射总量
    for i in range(len(time)):
        lightSum[i] = np.sum(rad[mnBefore:mnAfter + 1])

        # 到达当天结束点时，更新次日的0点索引
        if i == mnAfter - 1:
            mnBefore = mnAfter
            # 寻找次日0点后的索引
            mnAfter = np.where(np.diff(np.floor(time[mnBefore + 2:])) == 1)[0] + mnBefore + 2
            if mnAfter.size == 0:
                mnAfter = len(time)
            else:
                mnAfter = mnAfter[0]
    
    # 单位转换：W·s/m² = J/m² → 除以1e6转为MJ/m²
    return lightSum * interval * 1e-6

def soilTempNl(time):
    """
    估算荷兰地区1米深度的土壤温度（基于经验正弦函数）
    参考文献：Jacobs et al. (2011) Agric. For. Meteorol. 151, 774-780
    
    参数:
        time: 距年初的秒数数组
    
    返回:
        soilT: 土壤温度 [°C]
    """
    SECS_IN_YEAR = 3600 * 24 * 365  # 一年的秒数
    # 正弦函数拟合：平均10°C，振幅5°C，相位偏移0.625年
    soilT = 10 + 5 * np.sin((2 * np.pi * (time + 0.625 * SECS_IN_YEAR) / SECS_IN_YEAR))
    return soilT

def vaporDens2pres(temp, vaporDens):
    """
    水汽密度转蒸气压 [Pa]
    参考文献：http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    
    参数:
        temp: 温度 [°C]（数组）
        vaporDens: 水汽密度 [kg{H2O} m^{-3}]（数组）
    
    返回:
        vaporPres: 蒸气压 [Pa]（数组）
    """
    # 转换参数（经验值）
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # 相对湿度 = 实际水汽密度 / 饱和水汽密度
    rh = vaporDens / rh2vaporDens(temp, 100) 
    # 饱和蒸气压
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1]))
    # 实际蒸气压 = 饱和蒸气压 × 相对湿度
    return satP * rh

def satVp(temp):
    """
    计算给定温度下的饱和蒸气压 [Pa]
    参考文献：http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    
    参数:
        temp: 温度 [°C]（数组）
    
    返回:
        饱和蒸气压 [Pa]（数组）
    """
    return 610.78 * np.exp(17.2694 * temp / (temp + 237.3))  # 修正分母常数以匹配标准 Tetens 公式 (237.3 vs 238.3 差异不大，统一标准)


def calculate_vpd_kpa(temp, rh):
    """
    计算饱和水汽压差 (VPD) [kPa]
    
    参数:
        temp: 温度 [°C]
        rh: 相对湿度 [%] (0-100)
        
    返回:
        VPD [kPa]
    """
    sat_vp_pa = satVp(temp)
    vpd_pa = sat_vp_pa * (1 - rh / 100.0)
    return max(0.0, vpd_pa / 1000.0)



def co2ppm2dens(temp, ppm):
    """
    CO2浓度从ppm（摩尔浓度）转换为密度 [kg m^{-3}]
    基于理想气体定律 pV=nRT（假设标准大气压1atm）
    
    参数:
        temp: 温度 [°C]（数组）
        ppm: CO2浓度 [ppm]（数组）
    
    返回:
        co2Dens: CO2密度 [kg m^{-3}]（数组）
    """
    R = 8.3144598          # 气体常数 [J mol^{-1} K^{-1}]
    C2K = 273.15           # 摄氏转开尔文偏移量
    M_CO2 = 44.01e-3       # CO2摩尔质量 [kg mol^{-1}]
    P = 101325             # 标准大气压 [Pa]
    
    # 理想气体定律推导：密度 = (P × ppm × 1e-6 × M_CO2) / (R × (T))
    return P * 10**-6 * ppm * M_CO2 / (R * (temp + C2K))

def vaporDens2rh(temp, vaporDens):
    """
    水汽密度转相对湿度 [%]
    参考文献：http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    
    参数:
        temp: 温度 [°C]（数组）
        vaporDens: 水汽密度 [kg{H2O} m^{-3}]（数组）
    
    返回:
        rh: 相对湿度 [%]（数组，0-100）
    """
    # 常量定义
    R = 8.3144598       # 气体常数 [J mol^{-1} K^{-1}]
    C2K = 273.15        # 摄氏转开尔文偏移量
    Mw = 18.01528e-3    # 水的摩尔质量 [kg mol^{-1}]
    
    # 转换参数
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # 饱和蒸气压
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1])) 
    # 相对湿度 = (实际水汽密度 × R × T) / (Mw × 饱和蒸气压) × 100
    relhumid = 100 * R * (temp + C2K) / (Mw * satP) * vaporDens
    # 限制范围0-100（消除数值误差）
    return np.clip(relhumid, a_min=0, a_max=100)

def rh2vaporDens(temp, rh):
    """
    相对湿度转水汽密度 [kg{H2O} m^{-3}]
    参考文献：http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    
    参数:
        temp: 温度 [°C]（数组）
        rh: 相对湿度 [%]（数组，0-100）
    
    返回:
        vaporDens: 水汽密度 [kg{H2O} m^{-3}]（数组）
    """
    # 常量定义
    R = 8.3144598       # 气体常数 [J mol^{-1} K^{-1}]
    C2K = 273.15        # 摄氏转开尔文偏移量
    Mw = 18.01528e-3    # 水的摩尔质量 [kg mol^{-1}]
    
    # 转换参数
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # 饱和蒸气压
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1]))
    # 实际蒸气压 = 饱和蒸气压 × 相对湿度/100
    pascals = (rh / 100) * satP 
    # 水汽密度 = (实际蒸气压 × Mw) / (R × T)
    return pascals * Mw / (R * (temp + C2K))

def compute_sky_temp(air_temp, cloud):
    """
    从气温和云量计算天空温度 [°C]
    基于经验公式（温室模型常用）
    
    参数:
        air_temp: 气温 [°C]（数组）
        cloud: 云量 [0-1]（数组）
    
    返回:
        sky_temp: 天空温度 [°C]（数组）
    """
    sigma = 5.67e-8  # 斯特藩-玻尔兹曼常数 [W m^{-2} K^{-4}]
    C2K = 273.15      # 摄氏转开尔文偏移量

    # 晴天长波辐射 [W m^{-2}]
    ld_clear = 213 + 5.5 * air_temp                     
    # 晴天发射率
    eps_clear = ld_clear / (sigma * (air_temp + C2K)**4)    
    # 多云天发射率
    eps_cloud = (1 - 0.84 * cloud) * eps_clear + 0.84 * cloud   
    # 多云天长波辐射
    ld_cloud = eps_cloud * sigma * (air_temp + C2K)**4      
    # 天空温度（反推黑体温度）
    sky_temp = (ld_cloud / sigma)**(0.25) - C2K           
    return sky_temp