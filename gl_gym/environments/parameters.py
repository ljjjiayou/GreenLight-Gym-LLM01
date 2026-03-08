import numpy as np
from gl_gym.common.utils import satVp


def init_default_params(nparams):
    """
    初始化默认的物理、气候、温室、作物及设备参数数组。
    """
    params = np.zeros(nparams, dtype=np.float32)

    # 基础物理常数与气候参数
    params[0] = 5.  # alfaLeafAir:     叶片与空气间的对流换热系数 [W m-2 K-1]
    params[1] = 2.45e6  # L:               水的汽化潜热 [J kg-1]
    params[2] = 5.67e-8  # sigma:           斯特藩-玻尔兹曼常数 [W m-2 K-4]
    params[3] = 1.  # epsCan:          作物冠层的远红外发射率 []
    params[4] = 1.  # epsSky:          天空的远红外发射率 []
    params[5] = 0.5  # etaGlobNir:      全球辐射中近红外线 (NIR) 的比例 []
    params[6] = 0.5  # etaGlobPar:      全球辐射中光合有效辐射 (PAR) 的比例 []

    params[7] = 0.554  # etaMgPpm:        CO2 转换因子，从 mg/m^3 转换为 ppm []
    params[8] = 0.9  # etaRoofThr:      屋顶通风面积与总通风面积之比（假设无烟囱效应）
    params[9] = 1.2  # rhoAir0:         0 摄氏度时的空气密度 [kg m-3]
    params[10] = 0.07  # rhoCanPar:       作物冠层顶部的 PAR 反射系数
    params[11] = 0.35  # rhoCanNir:       作物冠层顶部的 NIR 反射系数
    params[12] = 7850.  # rhoSteel:        钢材密度 [kg m-3]
    params[13] = 1000.  # rhoWater:        水的密度 [kg m-3]
    params[14] = 65.8  # gamma:           湿度计常数 [Pa K-1]
    params[15] = 1.99e-7  # omega:           土壤温度计算的年频率 [s-1]

    params[16] = 1200.  # capLeaf:         叶片热容量 [J m-2 K-1]
    params[17] = 4.3  # cEvap1:          辐射对气孔阻力影响的系数
    params[18] = 0.54  # cEvap2:          水汽压差 (VPD) 对气孔阻力影响的系数
    params[19] = 6.1e-7  # cEvap3Day:       白天 CO2 对气孔阻力影响的系数
    params[20] = 1.1e-11  # cEvap3Night:     夜间 CO2 对气孔阻力影响的系数
    params[21] = 4.3e-6  # cEvap4Day:       白天 VPD 对气孔阻力影响的系数
    params[22] = 5.2e-6  # cEvap4Night:     夜间 VPD 对气孔阻力影响的系数
    params[23] = 1000.  # cPAir:           空气比热容 [J kg-1 K-1]
    params[24] = 640.  # cPSteel:         钢材比热容 [J kg-1 K-1]
    params[25] = 4180.  # cPWater:         水的比热容 [J kg-1 K-1]
    params[26] = 9.81  # g:               重力加速度 [m s-2]

    # 土壤参数
    params[27] = 0.04  # hSo1:            第 1 层土壤厚度 [m]
    params[28] = 0.08  # hSo2:            第 2 层土壤厚度 [m]
    params[29] = 0.16  # hSo3:            第 3 层土壤厚度 [m]
    params[30] = 0.32  # hSo4:            第 4 层土壤厚度 [m]
    params[31] = 0.64  # hSo5:            第 5 层土壤厚度 [m]

    params[32] = 0.7  # k1Par:           作物冠层 PAR 消光系数 [m2 m-2]
    params[33] = 0.7  # k2Par:           作物冠层 PAR 消光系数 [m2 m-2]
    params[34] = 0.27  # kNir:            作物冠层 NIR 消光系数 [m2 m-2]
    params[35] = 0.94  # kFir:            作物冠层远红外 (FIR) 消光系数 [m2 m-2]
    params[36] = 28.96  # mAir:            空气摩尔质量 [g mol-1]
    params[37] = 1.28  # hSoOut:          外部土壤层厚度 [m]
    params[38] = 18.  # mWater:          水的摩尔质量 [g mol-1]
    params[39] = 8314.  # R:               通用气体常数 [J mol-1 K-1]

    params[40] = 5.  # rCanSp:          黑夜转白天时冠层上方的辐射阈值
    params[41] = 275.  # rB:              Ball-Berry 模型参数 [s m-1]
    params[42] = 82.  # rSMin:           最小气孔阻力 [s m-1]
    params[43] = -1.  # sRs:             气孔阻力参数 []

    # 温室建筑结构参数
    params[44] = 0.1  # etaGlobAir:      温室结构吸收的全球辐射比例 []
    params[45] = 23.  # psi:             温室覆盖层平均坡度 []
    params[46] = 144.  # aFlr:            温室地面面积 [m2]
    params[47] = 216.6  # aCov:            覆盖层总表面积（含侧墙）[m2]
    params[48] = 5.7  # hAir:            温室主体隔间高度 [m]
    params[49] = 6.2  # hGh:             温室平均高度 [m]
    params[50] = 3.5  # cHecIn:          覆盖层与室内空气间的对流换热参数 [W m-2 K-1]
    params[51] = 2.8  # cHecOut1:        覆盖层与室外空气间的对流换热参数 1 [W m-2 K-1]
    params[52] = 1.2  # cHecOut2:        覆盖层与室外空气间的对流换热参数 2 [W m-2 K-1]
    params[53] = 1.  # cHecOut3:        覆盖层与室外空气间的对流换热参数 3 [W m-2 K-1]
    params[54] = 0.  # hElevation:      温室海拔高度 [m]
    params[55] = 52.2  # aRoof:           屋顶面积 [m2]
    params[56] = 0.87  # hVent:           通风口高度 [m]
    params[57] = 1.  # etaInsScr:       遮阳网绝缘因子 []
    params[58] = 0.  # aSide:           温室侧墙面积 [m2]
    params[59] = 0.35  # cDgh:            温室流量系数 [W m-2 K-1]
    params[60] = 0.3e-4  # cLeakage:        通风泄漏系数 [m3 s-1 m-2 Pa-1]
    params[61] = 0.02  # cWgh:            温室防风因子 []
    params[62] = 0.  # hSideRoof:       侧屋顶高度 [m]

    # 屋顶材料参数
    params[63] = 0.85  # epsRfFir:        屋顶远红外发射率 []
    params[64] = 2600.  # rhoRf:           屋顶密度 [kg m-3]
    params[65] = 0.13  # rhoRfNir:        屋顶 NIR 反射系数 []
    params[66] = 0.13  # rhoRfPar:        屋顶 PAR 反射系数 []
    params[67] = 0.15  # rhoRfFir:        屋顶远红外反射系数 []
    params[68] = 0.57  # tauRfNir:        屋顶 NIR 透射系数 []
    params[69] = 0.57  # tauRfPar:        屋顶 PAR 透射系数 []
    params[70] = 0.  # tauRfFir:        屋顶远红外透射系数 []
    params[71] = 1.05  # lambdaRf:        屋顶热传导率 [W m-1 K-1]
    params[72] = 840.  # cPRf:            屋顶比热容 [J kg-1 K-1]
    params[73] = 4e-3  # hRf:             屋顶厚度 [m]

    # 热遮阳网 (Thermal Screen) 参数
    params[74] = 0.67  # epsThScrFir:     热遮阳网远红外发射率 []
    params[75] = 200.  # rhoThScr:        热遮阳网密度 [kg m-3]
    params[76] = 0.35  # rhoThScrNir:     热遮阳网 NIR 反射系数 []
    params[77] = 0.35  # rhoThScrPar:     热遮阳网 PAR 反射系数 []
    params[78] = 0.18  # rhoThScrFir:     热遮阳网远红外反射系数 []
    params[79] = 0.75  # tauThScrNir:     热遮阳网 NIR 透射系数 []
    params[80] = 0.75  # tauThScrPar:     热遮阳网 PAR 透射系数 []
    params[81] = 0.15  # tauThScrFir:     热遮阳网远红外透射系数 []
    params[82] = 1800.  # cPThScr:         热遮阳网比热容 [J kg-1 K-1]
    params[83] = 0.35e-3  # hThScr:          热遮阳网厚度 [m]
    params[84] = 5.e-4  # kThScr:          热遮阳网通量系数

    # 黑体遮阳网 (Blackout Screen) 参数
    params[85] = 0.67  # epsBlScrFir:     黑体遮阳网远红外发射率 []
    params[86] = 200.  # rhoBlScr:        黑体遮阳网密度 [kg m-3]
    params[87] = 0.35  # rhoBlScrNir:     黑体遮阳网 NIR 反射系数 []
    params[88] = 0.35  # rhoBlScrPar:     黑体遮阳网 PAR 反射系数 []
    params[89] = 0.01  # tauBlScrNir:     黑体遮阳网 NIR 透射系数 []
    params[90] = 0.01  # tauBlScrPar:     黑体遮阳网 PAR 透射系数 []
    params[91] = 0.7  # tauBlScrFir:     黑体遮阳网远红外透射系数 []
    params[92] = 1800.  # cPBlScr:         黑体遮阳网比热容 [J kg-1 K-1]
    params[93] = 0.35e-3  # hBlScr:          黑体遮阳网厚度 [m]
    params[94] = 5.e-4  # kBlScr:          黑体遮阳网通量系数

    # 地面参数
    params[95] = 1.  # epsFlr:          地面远红外发射率 []
    params[96] = 2300.  # rhoFlr:          地面密度 [kg m-3]
    params[97] = 0.5  # rhoFlrNir:       地面 NIR 反射系数 []
    params[98] = 0.65  # rhoFlrPar:       地面 PAR 反射系数 []
    params[99] = 1.7  # lambdaFlr:       地面热传导率 [W m-1 K-1]
    params[100] = 880.  # cPFlr:           地面比热容 [J kg-1 K-1]
    params[101] = 0.02  # hFlr:            地面厚度 [m]

    params[102] = 1_730_000.  # rhoCpSo:         土壤体积热容量 [J m-3 K-1]
    params[103] = 0.85  # lambdaSo:        土壤热传导率 [W m-1 K-1]

    # 加热管道参数
    params[104] = 0.88  # epsPipe:         加热管道远红外发射系数
    params[105] = 51.e-3  # phiPipeE:        加热管道外径 [m]
    params[106] = 51.e-3 - 2.25e-3  # phiPipeI:        加热管道内径 [m]
    params[107] = 1.3375  # lPipe:           加热管道长度（每平方米）[m]
    params[108] = 130. * params[46]  # pBoil:           锅炉向加热系统的最大能量输入 [W]

    params[109] = 5.0 * params[46]  # phiExtCo2:       外部 CO2 源容量 [mg s-1]

    # 计算加热管道热容量 [J m-2 K-1]
    params[110] = 0.25 * np.pi * params[107] * (
                (params[105] ** 2 - params[106] ** 2) * params[12] * params[24] + params[106] ** 2 * params[13] *
                params[25])

    # 热容量计算
    params[111] = params[9] * np.exp(
        params[26] * params[36] * params[54] / (params[39] * 293.15))  # rhoAir: 空气密度 [kg m-3]
    params[112] = params[48] * params[111] * params[23]  # capAir:    空气热容量 [J m-3 K-1]
    params[113] = params[101] * params[96] * params[100]  # capFlr:    地面热容量 [J m-2 K-1]
    params[114] = params[27] * params[102]  # capSo1:   土壤第 1 层热容量 [J m-2 K-1]
    params[115] = params[28] * params[102]  # capSo2:   土壤第 2 层热容量 [J m-2 K-1]
    params[116] = params[29] * params[102]  # capSo3:   土壤第 3 层热容量 [J m-2 K-1]
    params[117] = params[30] * params[102]  # capSo4:   土壤第 4 层热容量 [J m-2 K-1]
    params[118] = params[31] * params[102]  # capSo5:   土壤第 5 层热容量 [J m-2 K-1]
    params[119] = params[83] * params[75] * params[82]  # capThScr: 热遮阳网热容量 [J m-2 K-1]
    params[120] = (params[49] - params[48]) * params[111] * params[23]  # capTop: 顶部隔间空气热容量 [J m-3 K-1]
    params[121] = params[93] * params[86] * params[92]  # capBlScr: 黑体遮阳网热容量 [J m-2 K-1]

    # CO2 容量计算
    params[122] = params[48]  # capCo2Air: 主隔间 CO2 容量 [m]
    params[123] = params[49] - params[48]  # capCo2Top: 顶部隔间 CO2 容量 [m]

    params[124] = np.pi * params[107] * params[105]  # aPipe:    每平方米地面面积对应的管道表面积 [m2 m-2]
    params[125] = 1 - 0.49 * np.pi * params[107] * params[105]  # fCanFlr:  冠层对地面的视角因子 []

    params[126] = 101325 * pow((1 - 2.5577e-5 * params[54]), 5.25588)  # pressure: 空气压力 [Pa]
    params[127] = 31.65  # energyContentGas: 天然气能量含量 [MJ m-3]

    # 作物生理参数
    params[128] = 2.3  # globJtUmol:      全球辐射到 PAR 的转换因子
    params[129] = 210.  # j25LeafMax:      25 度时的最大电子传递速率 [umol m-2 s-1]
    params[130] = 1.7  # cGamma:          冠层温度对 CO2 补偿点的影响
    params[131] = 0.67  # etaCo2AirStom:   温室空气 CO2 浓度到气孔内浓度的转换比
    params[132] = 37000  # eJ:              Jpot 计算的活化能
    params[133] = 298.15  # t25k:            Jpot 基准温度 (25°C)
    params[134] = 710  # S:               Jpot 计算的熵因子
    params[135] = 220_000  # H:               Jpot 计算的去活性能
    params[136] = 0.7  # theta:           电子传递速率曲线的弯曲度
    params[137] = 0.385  # alpha:           光子到电子的转换效率
    params[138] = 30e-3  # mCh2o:           CH2O 摩尔质量 [kg mol-1]
    params[139] = 44e-3  # mCo2:            CO2 摩尔质量 [kg mol-1]
    params[140] = 4.6  # parJtoUmolSun:   PAR 到 umol m-2 s-1 的转换因子
    params[141] = 3.0  # laiMax:          最大叶面积指数
    params[142] = 2.66e-5  # sla:             比叶面积 [m2 kg-1]
    params[143] = 3e-6  # rgr:             相对生长率 [kg m-2 s-1]
    params[144] = params[141] / params[142]  # cLeafMax:     最大叶片含碳量 [kg m-2]
    params[145] = 3_000_000  # cFruitMax:       最大果实含碳量 [mg m-2]
    params[146] = 0.27  # cFruitG:         果实生长呼吸系数
    params[147] = 0.28  # cLeafG:          叶片生长呼吸系数
    params[148] = 0.3  # cStemG:          茎干生长呼吸系数
    params[149] = 2_850_000  # cRgr:            维持呼吸函数中的回归系数
    params[150] = 2.  # q10m:            温度对维持呼吸影响的 Q10 值

    params[151] = 1.16e-7  # cFruitM:         果实维持呼吸系数
    params[152] = 3.47e-7  # cLeafM:          叶片维持呼吸系数
    params[153] = 1.47e-7  # cStemM:          茎干维持呼吸系数

    params[154] = 0.328  # rgFruit:         果实潜在生长率系数
    params[155] = 0.095  # rgLeaf:          叶片潜在生长率系数
    params[156] = 0.074  # rgStem:          茎干潜在生长率系数

    params[157] = 20e3  # cBufMax:         最大缓冲库容量 [J m-2 K-1]
    params[158] = 1e3  # cBufMin:         最小缓冲库容量 [J m-2 K-1]
    params[159] = 24.5  # tCan24Max:       24 小时最大冠层温度 [°C]
    params[160] = 15  # tCan24Min:       24 小时最小冠层温度 [°C]
    params[161] = 34  # tCanMax:         最大冠层温度 [°C]
    params[162] = 10  # tCanMin:         最小冠层温度 [°C]
    params[163] = 1035  # tEndSum:         达到最大潜力生长的积温阈值 [day °C]
    params[164] = 1250  # tEndSumGrowth:   停止生长的积温终值 [day °C]

    # 生长管道 (Growth Pipe) 参数
    params[165] = 0  # epsGroPipe:      生长管道远红外发射系数
    params[166] = 1.655  # lGroPipe:        生长管道长度 [m]
    params[167] = 35e-3  # phiGroPipeE:     生长管道外径 [m]
    params[168] = 35e-3 - 1.2e-3  # phiGroPipeI:     生长管道内径 [m]
    params[169] = np.pi * params[166] * params[167]  # aGroPipe: 每平方米地面面积对应的管道表面积 [m2 m-2]
    params[170] = 0  # pBoilGro:        锅炉向生长管道系统的最大输入能量 [W/m2]
    params[171] = 0.25 * np.pi * params[166] * (
                (params[167] ** 2 - params[168] ** 2) * params[12] * params[24] + params[168] ** 2 * params[13] *
                params[25])  # capGroPipe: 生长管道热容量 [J m-2 K-1]

    # LED 灯参数
    params[172] = 116  # thetaLampMax:    灯具最大能量输入 [W/m2]
    params[173] = 0  # heatCorrection:  灯具热量校正因子 []
    params[174] = 0.31  # etaLampPar:      灯具 PAR 效率 []
    params[175] = 0.02  # etaLampNir:      灯具 NIR 效率 []
    params[176] = 0.95  # tauLampPar:      灯具 PAR 透射系数 []
    params[177] = 0.95  # tauLampNir:      灯具 NIR 透射系数 []
    params[178] = 0.95  # tauLampFir:      灯具远红外 (FIR) 透射系数 []
    params[179] = 0.  # rhoLampPar:      灯具 PAR 反射系数 []
    params[180] = 0.  # rhoLampNir:      灯具 NIR 反射系数 []
    params[181] = 0.05  # aLamp:           每平方米地面面积对应的灯具表面积 [m2 m-2]
    params[182] = 0.88  # epsLampTop:      顶部灯具远红外发射系数 []
    params[183] = 0.88  # epsLampBottom:   底部灯具远红外发射系数 []
    params[184] = 10.  # capLamp:         灯具热容量 [J m-2 K-1]
    params[185] = 2.3  # cHecLampAir:     灯具与空气间的换热系数 [W m-2 K-1]
    params[186] = 0.63  # etaLampCool:     灯具冷却效率 []
    params[187] = 5.2  # zetaLampPar:     灯具 PAR 排放系数 []

    # 内部照明 (Inter Lamp) 参数
    params[188] = 0  # intLamps:        是否存在内部照明灯
    params[189] = 0.5  # vIntLampPos:     内部照明在冠层内的垂直位置 (0-1, 0 为顶部)
    params[190] = 0.5  # fIntLampDown:    灯光向下照射的比例
    params[191] = 10  # capIntLamp:      内部照明灯热容量
    params[192] = 0  # etaIntLampPar:   光合有效辐射转化比例
    params[193] = 0  # etaIntLampNir:   近红外转化比例
    params[194] = 0  # aIntLamp:        内部灯具面积
    params[195] = 0  # epsIntLamp:      灯具发射率
    params[196] = 0  # thetaIntLampMax: 内部照明最大强度
    params[197] = 0  # zetaIntLampPar:  J 到 umol 的转换系数
    params[198] = 0  # cHecIntLampAir:  热交换系数
    params[199] = 1  # tauIntLampFir:   远红外透射率
    params[200] = 1.4  # k1IntPar:        冠层 PAR 消光系数
    params[201] = 1.4  # k2IntPar:        地面反射光的消光系数
    params[202] = 0.54  # kIntNir:         NIR 消光系数
    params[203] = 1.88  # kIntFir:         FIR 消光系数

    # 杂项参数
    params[204] = 0.9  # cLeakTop:        从顶部流出的通风泄漏比例
    params[205] = 0.25  # minWind:         风速影响泄漏的起始阈值
    params[206] = 0.0627  # dmfm:            干物质到鲜物质的转化率
    params[207] = 1e-6  # eps:             数值稳定性极小值 (Epsilon)
    return params


def init_state(d0, rhMax=90, time_in_days=0):
    """
    初始化温室的 28 个物理状态变量。
    """
    state = np.zeros(28)

    state[0] = d0[3]  # co2Air:    空气 CO2 浓度
    state[1] = state[0]  # co2Top:    顶部隔间 CO2 浓度
    state[2] = 18.5  # tAir:      空气温度
    state[3] = state[2]  # tTop:      顶部温度
    state[4] = state[2] + 2  # tCan:      作物冠层温度
    state[5] = state[2]  # tCovIn:    内覆盖层温度
    state[6] = state[2]  # tCovE:     外覆盖层温度
    state[7] = state[2]  # tThScr:    热遮阳网温度
    state[8] = state[2]  # tFlr:      地面温度
    state[9] = state[2]  # tPipe:     加热管道温度
    state[10] = state[2]  # tSoil1:    土壤第 1 层温度
    state[11] = 0.25 * (3. * state[2] + d0[6])  # tSoil2: 土壤第 2 层温度
    state[12] = 0.25 * (2. * state[2] + 2 * d0[6])  # tSoil3: 土壤第 3 层温度
    state[13] = 0.25 * (state[2] + 3 * d0[6])  # tSoil4: 土壤第 4 层温度
    state[14] = d0[6]  # tSoil5:    土壤第 5 层温度
    state[15] = rhMax / 100. * satVp(state[2])  # vpAir:  空气水蒸气压
    state[16] = state[15]  # vpTop:     顶部水蒸气压
    state[17] = state[2]  # tLamp:     灯具温度
    state[18] = state[2]  # tIntLamp:  内部灯具温度
    state[19] = state[2]  # tGroPipe:  生长管道温度
    state[20] = state[2]  # tBlScr:    黑体遮阳网温度
    state[21] = state[4]  # tCan24:    24 小时平均冠层温度
    state[22] = 1000.  # cBuf:      缓冲碳水化合物库
    state[23] = 26000.  # cLeaf:     叶片总碳量
    state[24] = 18000.  # cStem:     茎干总碳量
    state[25] = 0.  # cFruit:    果实总碳量 (起始为 0)
    state[26] = 0.  # tCanSum:   累计积温
    state[27] = time_in_days  # time:      当前模拟天数

    return state