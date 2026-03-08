import numpy as np
from gl_gym.environments.utils import co2dens2ppm, satVp


class RuleBasedController:
    def __init__(
            self,
            lamps_on,  # 补光灯开启时间（小时）
            lamps_off,  # 补光灯关闭时间（小时）
            lamps_day_start,  # 一年中开始补光的日期
            lamps_day_stop,  # 一年中停止补光的日期
            lamps_off_sun,  # 阳光辐射超过此值时关闭补光灯
            lamp_rad_sum_limit,  # 每日累积辐射限值
            temp_setpoint_day,  # 白天温度目标值
            temp_setpoint_night,  # 夜间温度目标值
            heat_correction,  # 补光时的温度补偿
            heat_deadzone,  # 温度控制死区
            co2_day,  # 白天 CO2 目标浓度
            vent_heat_Pband,  # 降温通风的比例带
            rh_max,  # 最大相对湿度目标
            mech_dehumid_Pband,  # 机械除湿比例带
            vent_rh_Pband,  # 除湿通风比例带
            t_vent_off,  # 停止通风的低温阈值
            vent_cold_Pband,  # 防冷风通风比例带
            thScrSpDay,  # 白天遮阳网触发温度
            thScrSpNight,  # 夜间遮阳网触发温度
            thScrPband,  # 遮阳网比例带
            thScrDeadZone,  # 遮阳网控制死区
            thScrRh,  # 湿度导致的遮阳网开启阈值
            thScrRhPband,  # 湿度遮阳网比例带
            lampExtraHeat,  # 补光额外产热补偿
            blScrExtraRh,  # 黑幕开启时的额外湿度补偿
            rhMax,  # 相对湿度上限
            tHeatBand,  # 加热比例带
            co2Band,  # CO2 比例带
            useBlScr  # 是否使用黑幕
    ):
        # 初始化所有控制参数
        self.lamps_on = lamps_on
        self.lamps_off = lamps_off
        self.lamps_day_start = lamps_day_start
        self.lamps_day_stop = lamps_day_stop
        self.lamps_off_sun = lamps_off_sun
        self.lamp_rad_sum_limit = lamp_rad_sum_limit
        self.temp_setpoint_day = temp_setpoint_day
        self.temp_setpoint_night = temp_setpoint_night
        self.heat_correction = heat_correction
        self.heat_deadzone = heat_deadzone
        self.co2_day = co2_day
        self.vent_heat_Pband = vent_heat_Pband
        self.rh_max = rh_max
        self.mech_dehumid_Pband = mech_dehumid_Pband
        self.vent_rh_Pband = vent_rh_Pband
        self.t_vent_off = t_vent_off
        self.vent_cold_Pband = vent_cold_Pband
        self.thScrSpDay = thScrSpDay
        self.thScrSpNight = thScrSpNight
        self.thScrPband = thScrPband
        self.thScrDeadZone = thScrDeadZone
        self.thScrRh = thScrRh
        self.thScrRhPband = thScrRhPband
        self.lampExtraHeat = lampExtraHeat
        self.blScrExtraRh = blScrExtraRh
        self.rhMax = rhMax
        self.tHeatBand = tHeatBand
        self.co2Band = co2Band
        self.useBlScr = useBlScr

    def predict(self, x, d, env):
        """
        根据当前状态 x 和环境数据 d 预测控制输出 u
        """
        u = np.zeros(env.nu)

        # 1. 根据时间判断补光灯状态 [0/1]
        # 如果开启时间 < 关闭时间，则在两者之间开启
        # 如果开启时间 > 关闭时间，则跨天开启
        lampTimeOfDay = ((self.lamps_on <= self.lamps_off) * (
                    self.lamps_on < env.hour_of_day and env.hour_of_day < self.lamps_off) + \
                         (1 - (self.lamps_on <= self.lamps_off)) * (
                                     self.lamps_on < env.hour_of_day or env.hour_of_day < self.lamps_off))

        # 2. 根据一年中的日期判断是否处于补光季
        lampDayOfYear = ((self.lamps_day_start <= self.lamps_day_stop) * (
                    self.lamps_day_start < env.day_of_year and env.day_of_year < self.lamps_day_stop) + \
                         (1 - (self.lamps_day_start <= self.lamps_day_stop)) * (
                                     self.lamps_day_start < env.day_of_year or env.day_of_year < self.lamps_day_stop))

        # 判断是否处于“照明期”（不考虑温湿度约束）
        # 即使因为过热关了灯，逻辑上仍处于“照明期”以维持气候设定值
        lampNoCons = (d[0] < self.lamps_off_sun) * (d[7] < self.lamp_rad_sum_limit) * lampTimeOfDay * lampDayOfYear

        # 3. 补光灯开启/关闭的平滑处理（模拟线性过渡）
        linearLampSwitchOn = max(0, min(1, env.hour_of_day - self.lamps_on + 1))
        linearLampSwitchOff = max(0, min(1, self.lamps_off - env.hour_of_day + 1))

        linearLampBothSwitches = (self.lamps_on != self.lamps_off) * (
                    (self.lamps_on < self.lamps_off) * min(linearLampSwitchOn, linearLampSwitchOff)
                    + (1 - (self.lamps_on < self.lamps_off)) * max(linearLampSwitchOn, linearLampSwitchOff))

        # 平滑后的灯光信号，用于设定值切换
        smoothLamp = linearLampBothSwitches * (d[7] < self.lamp_rad_sum_limit) * lampDayOfYear

        # 4. 判断室内是否为“白天”（太阳出来或灯开着都算白天）
        isDayInside = max(smoothLamp, d[8])

        # 5. 计算加热目标温度 [°C]
        # 基础值 + 补光时的热修正
        heatSetPoint = isDayInside * self.temp_setpoint_day + (
                    1 - isDayInside) * self.temp_setpoint_night + self.heat_correction * lampNoCons

        # 通风降温的起始温度
        heatMax = heatSetPoint + self.heat_deadzone

        # CO2 目标浓度（仅在白天施肥）
        co2SetPoint = isDayInside * self.co2_day

        # 将 CO2 密度转换为 ppm
        co2InPpm = co2dens2ppm(x[2], 1e-6 * x[0])

        # 6. 计算各项控制逻辑
        # a) 因超温导致的通风
        ventHeat = self.proportional_control(x[2], heatMax, self.vent_heat_Pband, 0, 1)

        # 计算室内相对湿度 [%]
        rhIn = 100 * x[15] / satVp(x[2])

        # b) 因湿度过大导致的通风
        ventRh = self.proportional_control(rhIn, self.rh_max + 0 * self.mech_dehumid_Pband, self.vent_rh_Pband, 0, 1)

        # c) 因温度过低导致的通风关闭（保护逻辑）
        ventCold = self.proportional_control(x[2], heatSetPoint - self.t_vent_off, self.vent_cold_Pband, 1, 0)

        # d) 遮阳网目标设定（根据室外辐射判断）
        thScrSp = (d[8]) * self.thScrSpDay + (1 - (d[8])) * self.thScrSpNight

        # e) 根据室外温度决定遮阳网关闭（保温）
        thScrCold = self.proportional_control(d[1], thScrSp, self.thScrPband, 0, 1)

        # f) 如果室内过热，强制打开遮阳网
        thScrHeat = self.proportional_control(x[2], heatSetPoint + self.thScrDeadZone, -self.thScrPband, 1, 0)

        # g) 如果湿度过大，打开遮阳网缝隙排湿
        thScrRh = max(self.proportional_control(rhIn, self.rhMax + self.thScrRh, self.thScrRhPband, 1, 0), 1 - ventCold)

        # 7. 计算灯具实际开启状态（考虑热和湿度惩罚）
        lampOn = lampNoCons * self.proportional_control(x[2], heatMax + self.lampExtraHeat, -0.5, 0, 1) * \
                 (d[9] + (1 - d[9])) * \
                 max(self.proportional_control(rhIn, self.rhMax + self.blScrExtraRh, -0.5, 0, 1), 1 - ventCold)

        # 8. 映射到最终控制向量 u
        # u 索引依次为：锅炉(加热), CO2, 遮阳网, 屋顶通风, 补光灯, 黑幕
        u[0] = self.proportional_control(x[2], heatSetPoint, self.tHeatBand, 0, 1)
        u[1] = self.proportional_control(co2InPpm, co2SetPoint, self.co2Band, 0, 1)
        u[2] = min(thScrCold, max(thScrHeat, thScrRh))  # 遮阳网综合逻辑
        u[3] = min(ventCold, max(ventHeat, ventRh))  # 通风综合逻辑
        u[4] = lampOn
        u[5] = self.useBlScr * (1 - d[9]) * lampOn  # 黑幕逻辑

        return u

    def proportional_control(self, processVar, setPt, pBand, minVal, maxVal):
        """
        比例控制函数：使用 Sigmoid 曲线将输入变量映射到 [minVal, maxVal]
        processVar: 当前测量值
        setPt: 设定目标值
        pBand: 比例带宽度
        """
        return minVal + (maxVal - minVal) * (
                    1 / (1 + np.exp(-2 / pBand * np.log(100) * (processVar - setPt - pBand / 2))))