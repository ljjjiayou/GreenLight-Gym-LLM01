from typing import Tuple, SupportsFloat
from os.path import join

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from pandas._typing import ArrayLike
import pandas as pd
from scipy.interpolate import PchipInterpolator


def init_state(d0, rhMax=90, time_in_days=0):
    """
    初始化状态变量数组（共 28 个元素）。
    d0: 初始时刻的天气数据向量
    """
    state = np.zeros(28)

    state[0] = d0[3]  # 空气 CO2 浓度 (co2Air)
    state[1] = state[0]  # 顶部隔间 CO2 浓度 (co2Top)
    state[2] = 16.5  # 空气温度 (tAir)
    state[3] = state[2]  # 顶部温度 (tTop)
    state[4] = state[2] + 4  # 冠层温度 (tCan)
    state[5] = state[2]  # 内覆盖层温度 (tCovIn)
    state[6] = state[2]  # 外覆盖层温度 (tCovE)
    state[7] = state[2]  # 热遮阳网温度 (tThScr)
    state[8] = state[2]  # 地面温度 (tFlr)
    state[9] = state[2]  # 加热管道温度 (tPipe)
    state[10] = state[2]  # 第 1 层土壤温度 (tSoil1)
    state[11] = 0.25 * (3. * state[2] + d0[6])  # 第 2 层土壤温度 (tSoil2)
    state[12] = 0.25 * (2. * state[2] + 2 * d0[6])  # 第 3 层土壤温度 (tSoil3)
    state[13] = 0.25 * (state[2] + 3 * d0[6])  # 第 4 层土壤温度 (tSoil4)
    state[14] = d0[6]  # 第 5 层土壤温度 (tSoil5)
    state[15] = rhMax / 100. * satVp(state[2])  # 空气水蒸气压 (vpAir)
    state[16] = state[15]  # 顶部水蒸气压 (vpTop)
    state[17] = state[2]  # 灯具温度 (tLamp)
    state[18] = state[2]  # 内部灯具温度 (tIntLamp)
    state[19] = state[2]  # 生长管道温度 (tGroPipe)
    state[20] = state[2]  # 黑体遮阳网温度 (tBlScr)
    state[21] = state[4]  # 24小时平均冠层温度 (tCan24)
    state[22] = 0.  # 碳水化合物缓冲库 (cBuf)
    state[23] = 9.5283e4  # 叶片总碳量 (cLeaf)
    state[24] = 2.5107e5  # 茎干总碳量 (cStem)
    state[25] = 5.5338e4  # 果实总碳量 (cFruit)
    state[26] = 3.0978e3  # 累计积温 (tCanSum)
    state[27] = time_in_days  # 时间（天）

    return state


def load_weather_data(
        weatherDataDir: str,
        location: str,
        growthYear: int,
        startDay: int,
        nDays: int,
        predHorizon: int,
        h: float,
        nd: int
) -> np.ndarray:
    """
    加载原始天气数据并转换为 GreenLight 模型使用的格式。
    如果求解器需要更高频率的数据，则在现有数据间进行插值。
    原始数据间隔通常为 5 分钟，包含 9 列。

    参数:
        weatherDataDir  - 原始数据路径
        location        - 地点名称
        growthYear      - 生长季起始年份
        startDay        - 模拟起始天
        nDays           - 模拟总天数
        predHorizon     - 预测展望期 [天]
        h               - 求解器采样时间（秒）
        nd              - 天气变量数量

    返回:
        包含以下插值后天气变量的矩阵：
        d[0]: iGlob         全球辐射 [W m^{-2}]
        d[1]: tOut          室外温度 [deg C]
        d[2]: vpOut         室外水蒸气压 [Pa]
        d[3]: co2Out        室外 CO2 浓度 [mg m^{-3}]
        d[4]: wind          风速 [m s^{-1}]
        d[5]: tSky          天空温度 [deg C]
        d[6]: tSoOut        室外土壤温度 [deg C]
        d[7]: dli           每日辐射总量 [MJ m^{-2} day^{-1}]
        d[8]: isDay         是否为白天 [0,1]
        d[9]: isDaySmooth   是否为白天（带平滑过渡）[0,1]
    """
    weatherDataPath = join(join(weatherDataDir, location), str(growthYear)) + ".csv"

    c = 86400  # 一天的秒数
    CO2_PPM = 400  # 假设室外 CO2 浓度恒定 [ppm]
    rawWeather = pd.read_csv(weatherDataPath, sep=",")

    time = rawWeather["time"].values  # 自年初以来的时间 [s]
    dt = np.mean(np.diff(time - time[0]))  # 原始数据的采样周期 [s]
    N0 = int(np.ceil(startDay * c / dt))  # 起始索引
    Ns = int(np.ceil(nDays * c / dt))  # 模拟所需的样本数
    Np = int(np.ceil(predHorizon * c / dt)) + 1  # 预测展望期所需的样本数

    # 检查是否超出当前年份数据长度，若超出则加载下一年
    if N0 + Ns + Np > len(time):
        rawWeather = expandWeatherData(weatherDataDir, rawWeather, location, growthYear, time, dt)

    weatherData = np.zeros((Ns + Np, nd))  # 预分配矩阵
    time = rawWeather["time"].values[N0:N0 + Ns + Np]  # 截取对应时间段
    weatherData[:, 0] = rawWeather["global radiation"][N0:N0 + Ns + Np]  # 辐射
    weatherData[:, 1] = rawWeather["air temperature"][N0:N0 + Ns + Np]  # 室外温度
    vpDensity = rh2vaporDens(weatherData[:, 1], rawWeather["RH"][N0:N0 + Ns + Np])  # 计算水蒸气密度
    weatherData[:, 2] = vaporDens2pres(weatherData[:, 1], vpDensity)  # 计算水蒸气压
    weatherData[:, 3] = co2ppm2dens(weatherData[:, 1], CO2_PPM) * 1e6  # 转换 CO2 单位为 mg/m^3
    weatherData[:, 4] = rawWeather["wind speed"][N0:N0 + Ns + Np]  # 风速
    weatherData[:, 5] = rawWeather["sky temperature"][N0:N0 + Ns + Np]  # 天空温度
    weatherData[:, 6] = soilTempNl(rawWeather["time"][N0:N0 + Ns + Np])  # 土壤温度（模型估算）
    weatherData[:, 7] = dailLightSum(time, weatherData[:, 0], c)  # 每日辐射积分 (DLI)
    weatherData[:, 8], weatherData[:, 9] = computeisDay(weatherData[:, 0], dt)  # 计算昼夜标识

    # 计算求解器所需的总样本数
    ns = int((dt / h) * (Ns + Np))

    # 执行插值与重采样
    interpolation = PchipInterpolator(time, weatherData)
    timeRes = np.linspace(time[0], time[-1], ns)
    weatherDataResampled = interpolation(timeRes)

    # 将微小的负值辐射（插值误差）设为 0
    weatherDataResampled[:, 0][weatherDataResampled[:, 0] < 1e-10] = 0

    return weatherDataResampled


def expandWeatherData(
        weatherDataDir: str,
        rawWeather: pd.DataFrame,
        location: str,
        growthYear: int,
        time: ArrayLike,
        dt: SupportsFloat,
) -> pd.DataFrame:
    """
    加载下一年的天气数据并拼接到当前数据后。用于模拟时长跨越年份的情况。
    """
    weatherDataPath = join(join(weatherDataDir, location), str(growthYear + 1)) + ".csv"
    newRawWeather = pd.read_csv(weatherDataPath, sep=",")
    newRawWeather["time"] += time[-1] + dt
    rawWeather = pd.concat([rawWeather, newRawWeather.iloc[:, :]])
    return rawWeather


def days2date(timeInDays: float, referenceDate: str):
    """
    将自参考日期以来的天数转换为 DD-MM-YYYY 格式的日期。
    """
    referenceDatetime = datetime.strptime(referenceDate, '%d-%m-%Y')
    int_days = np.floor(timeInDays).astype(int)
    time_component = (timeInDays - int_days) * 24  # 转换为小时
    hours = time_component.astype(int)
    time_component = (time_component - hours) * 60  # 转换为分钟
    minutes = time_component.astype(int)

    target_datetimes = [referenceDatetime + timedelta(days=int(int_day), hours=int(hour), minutes=int(minute)) for
                        int_day, hour, minute in zip(int_days, hours, minutes)]

    return [target_datetime.strftime('%Y-%m-%d %H:%M:%S') for target_datetime in target_datetimes]


def computeisDay(rad: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于辐射强度计算当前是白天还是黑夜。
    通过 Sigmoid 函数在日出日落时提供平滑过渡，模拟黎明和黄昏。
    """
    # 基础昼夜判定
    isDay = (rad > 0) * 1.0
    isDaySmooth = deepcopy(isDay)
    transSize = int(3600 / dt)  # 过渡期长度设为一小时对应的样本数

    trans = np.linspace(0, 1, transSize)
    # 使用 sigmoid 增强过渡平滑度
    transSmooth = 1 / (1 + np.exp(-10 * (trans - 0.5)))
    sunset = False  # 标记是否处于日落期间

    for k in range(transSize, len(isDay) - transSize):
        if isDay[k] == 0:
            sunset = False
        # 日出检测与平滑处理
        if isDay[k] == 0 and isDay[k + 1] == 1:
            isDay[k - transSize // 2: k + transSize // 2] = trans
            isDaySmooth[k - transSize // 2: k + transSize // 2] = transSmooth
        # 日落检测与平滑处理
        elif isDay[k] == 1 and isDay[k + 1] == 0 and not sunset:
            isDay[k - transSize // 2: k + transSize // 2] = 1 - trans
            isDaySmooth[k - transSize // 2: k + transSize // 2] = 1 - transSmooth
            sunset = True
    return isDay, isDaySmooth


def dailLightSum(time: np.ndarray, rad: np.ndarray, c: int):
    """
    计算给定辐射时间序列的每日光照积分 (DLI)。
    返回单位: [MJ m^{-2} day^{-1}]
    """
    interval = time[1] - time[0]  # 采样间隔 [s]
    time = time / c  # 转换为天

    # 记录当前点之前的午夜索引
    mnBefore = 0

    # 查找当前点之后的第一个午夜索引
    mnAfter = np.where(np.diff(np.floor(time)) == 1)[0] + 1
    if mnAfter.size == 0:
        mnAfter = len(time)
    else:
        mnAfter = mnAfter[0]
    lightSum = np.zeros(len(time))

    for i in range(len(time)):
        # 累加当天从午夜到午夜的辐射
        lightSum[i] = np.sum(rad[mnBefore:mnAfter + 1])

        # 如果跨越了午夜，更新搜索窗口
        if i == mnAfter - 1:
            mnBefore = mnAfter
            mnAfter = np.where(np.diff(np.floor(time[mnBefore + 2:])) == 1)[0] + mnBefore + 2
            if mnAfter.size == 0:
                mnAfter = len(time)
            else:
                mnAfter = mnAfter[0]
    return lightSum * interval * 1e-6


def soilTempNl(time):
    """
    估算荷兰给定时间的土壤温度（1米深）。
    基于 Jacobs 等人 (2011) 在《农业与森林气象学》上发表的研究，使用正弦函数逼近。
    """
    SECS_IN_YEAR = 3600 * 24 * 365
    soilT = 10 + 5 * np.sin((2 * np.pi * (time + 0.625 * SECS_IN_YEAR) / SECS_IN_YEAR))
    return soilT


def vaporDens2pres(temp, vaporDens):
    """
    将水蒸气密度 [kg{H2O} m^{-3}] 转换为水蒸气压 [Pa]。
    基于 ideal gas law 和 saturation vapor pressure 计算。
    """
    # 转换参数
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]

    rh = vaporDens / rh2vaporDens(temp, 100)  # 计算相对湿度 [0-1]
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1]))  # 饱和水蒸气压

    return satP * rh


def satVp(temp):
    """
    计算给定温度 (°C) 下的饱和水蒸气压 (Pa)。
    """
    return 610.78 * np.exp(17.2694 * temp / (temp + 238.3))


def co2ppm2dens(temp, ppm):
    """
    将 CO2 摩尔浓度 [ppm] 转换为密度 [kg m^{-3}]。
    基于理想气体状态方程 pV=nRT，假设压力为 1 个标准大气压。
    """
    R = 8.3144598  # 摩尔气体常数 [J mol^{-1} K^{-1}]
    C2K = 273.15  # 摄氏度转开尔文
    M_CO2 = 44.01e-3  # CO2 摩尔质量 [kg mol^-{1}]
    P = 101325  # 标准大气压 [Pa]

    return P * 10 ** -6 * ppm * M_CO2 / (R * (temp + C2K))


def co2dens2ppm(temp, dens):
    """
    将 CO2 密度 [kg m^{-3}] 转换为摩尔浓度 [ppm]。
    """
    R = 8.3144598
    C2K = 273.15
    M_CO2 = 44.01e-3
    P = 101325

    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)


def vaporPres2rh(temp, vaporPres):
    """水蒸气压转相对湿度，限制在 0-100 之间"""
    return np.clip(100 * vaporPres / satVp(temp), a_min=0., a_max=100.)


def vaporDens2rh(temp, vaporDens):
    """
    将水蒸气密度 [kg{H2O} m^{-3}] 转换为相对湿度 [%]。
    基于物理常数（气体常数、水的摩尔质量等）计算。
    """
    R = 8.3144598
    C2K = 273.15
    Mw = 18.01528e-3  # 水的摩尔质量

    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1]))

    # 使用理想气体定律计算相对湿度
    relhumid = 100 * R * (temp + C2K) / (Mw * satP) * vaporDens
    return np.clip(relhumid, a_min=0, a_max=100)


def rh2vaporDens(temp, rh):
    """
    将相对湿度 [%] 转换为水蒸气密度 [kg{H2O} m^{-3}]。
    """
    R = 8.3144598
    C2K = 273.15
    Mw = 18.01528e-3

    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    satP = p[0] * np.exp(p[2] * temp / (temp + p[1]))

    pascals = (rh / 100) * satP  # 计算部分水蒸气分压 [Pa]

    return pascals * Mw / (R * (temp + C2K))


def compute_sky_temp(air_temp, cloud):
    """
    基于空气温度和云量计算天空温度。
    用于计算温室覆盖层与天空之间的长波辐射热交换。
    """
    sigma = 5.67e-8  # 斯特藩-玻尔兹曼常数
    C2K = 273.15

    ld_clear = 213 + 5.5 * air_temp
    eps_clear = ld_clear / (sigma * (air_temp + C2K) ** 4)
    eps_cloud = (1 - 0.84 * cloud) * eps_clear + 0.84 * cloud
    ld_cloud = eps_cloud * sigma * (air_temp + C2K) ** 4
    sky_temp = (ld_cloud / sigma) ** (0.25) - C2K  # 假设发射率为 1 时的计算
    return sky_temp