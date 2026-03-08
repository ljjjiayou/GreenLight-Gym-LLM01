import numpy as np


def parametric_crop_uncertainty(parameters, uncertainty, RNG):
    """
    根据不确定性参数向参数向量中添加随机噪声。

    参数:
        parameters (np.ndarray): 原始参数向量。
        uncertainty (float): 要添加的不确定性水平（比例）。
        RNG: 随机数生成器 (Random Number Generator)。

    返回:
        np.ndarray: 添加了不确定性后的参数向量。
    """
    # 参数向量中以下索引对应的作物参数将被扰动 (索引 128 到 161)
    indices = np.arange(128, 162)
    parameters = np.array(parameters)

    # 生成均匀分布的噪声，范围在 [-uncertainty/2, uncertainty/2] 之间
    noise = RNG.uniform(-uncertainty / 2, uncertainty / 2, size=indices.shape)

    # 将噪声应用到参数上：新参数 = 原参数 + 原参数 * 噪声
    parameters[indices] += noise * parameters[indices]

    # 维护参数间的物理依赖关系：
    # cLeafMax（最大叶片含碳量）取决于 laiMax（最大叶面积指数）和 sla（比叶面积）
    # 公式：cLeafMax = laiMax / sla
    parameters[144] = parameters[141] / parameters[142]

    return parameters