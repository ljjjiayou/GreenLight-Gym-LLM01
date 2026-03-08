import pandas as pd
from scipy.io import loadmat


if __name__ == "__main__":
    # 1. 加载 CSV 文件数据（通常包含 Python 环境生成的控制量）
    csv_file_path = 'data/AgriControl/comparison/gl_u_data.csv'
    csv_data = pd.read_csv(csv_file_path)

    # 从 CSV 中选择特定的列（控制指令）
    # boil: 锅炉, extCo2: 二氧化碳, thScr: 热遮阳网, roof: 通风, lamp: 补光, blScr: 黑幕
    csv_selected_columns = csv_data[['boil', 'extCo2', 'thScr', 'roof', 'lamp', 'blScr']]

    # 2. 加载 MAT 文件数据（通常包含 MATLAB 原始模型生成的控制量）
    mat_file_path = 'data/AgriControl/comparison/gl_controls.mat'
    mat_data = loadmat(mat_file_path)

    # 提取其中的 'controls' 变量矩阵
    controls_data = mat_data['controls']

    # 提取矩阵的第五列（索引为 4），该列代表管道温度 (Pipe Temperature)
    pipe_temperature = controls_data[:, 4]

    # 将提取出的管道温度转换为 DataFrame 格式，方便后续合并
    pipe_temperature_df = pd.DataFrame(pipe_temperature, columns=['pipe_temperature'])

    # 3. 组合两个数据框
    # 将 CSV 的控制指令与 MAT 的管道温度按轴对齐合并
    merged_data = pd.concat([csv_selected_columns, pipe_temperature_df], axis=1)

    # 4. 将合并后的数据保存到新的 CSV 文件中，供后续分析或训练使用
    output_file_path = 'merged_data.csv'
    merged_data.to_csv(output_file_path, index=False)