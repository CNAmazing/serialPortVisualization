
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import json
import yaml
from PIL import Image
from tools import *
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
def get_bayer_color(x, y):
    if y % 2 == 0:    # 偶数行
        if x % 2 == 0:
            return 'R' # 偶数列 → 红
        else:
            return 'Gr' # 奇数列 → 绿
    else:             # 奇数行
        if x % 2 == 0:
            return 'Gb' # 偶数列 → 绿
        else:
            return 'B' # 奇数列 → 蓝
def read_pgm_with_opencv(file_path):
    # 使用 cv2.imread 读取 PGM 文件
    # 第二个参数 flags 可以是:
    # cv2.IMREAD_COLOR (默认，转换为3通道BGR)
    # cv2.IMREAD_GRAYSCALE (以灰度模式读取)
    # cv2.IMREAD_UNCHANGED (保留原样，包括alpha通道)
    img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取文件: {file_path}")
        return None
    
    return img
def read_pgm_with_pillow(file_path):
    img = Image.open(file_path) 
    if img is None:
        print(f"无法读取文件: {file_path}")
        return None
    return img
def lsc_calib_rgb(image, mesh_h_nums=12, mesh_w_nums=16, sample_size=5,maxFactor=1.0):
    """
    RGB图像的Lens Shading Correction校准
    参数:
        image: 输入RGB图像 (H, W, 3)
        mesh_h_nums: 垂直方向网格数 (默认49)
        mesh_w_nums: 水平方向网格数 (默认33)
        sample_size: 采样区域大小 (默认7x7)
    返回:
        mesh_r: R通道增益网格
        mesh_g: G通道增益网格
        mesh_b: B通道增益网格
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是RGB图像 (H, W, 3)")
    
    height, width = image.shape[:2]
    mesh_box_h = int(np.ceil(height / mesh_h_nums))
    mesh_box_w = int(np.ceil(width / mesh_w_nums))
    
    # 初始化网格 (存储局部平均值)
    mesh_r = np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)
    mesh_g = np.zeros_like(mesh_r)
    mesh_b = np.zeros_like(mesh_r)
    
    # 记录各通道全局最大值
    max_r, max_g, max_b = 1.0, 1.0, 1.0
    
    offset = sample_size // 2
    
    for i in range(mesh_h_nums + 1):
        for j in range(mesh_w_nums + 1):
            center_y = min(i * mesh_box_h, height - 1)
            center_x = min(j * mesh_box_w, width - 1)
            
            # 定义采样区域边界
            y_start = max(center_y - offset, 0)
            y_end = min(center_y + offset + 1, height)
            x_start = max(center_x - offset, 0)
            x_end = min(center_x + offset + 1, width)
            
            # 提取采样区域
            sample_patch = image[y_start:y_end, x_start:x_end]
            
            # 计算各通道平均值
            avg_r = np.mean(sample_patch[:, :, 2])
            avg_g = np.mean(sample_patch[:, :, 1])
            avg_b = np.mean(sample_patch[:, :, 0])
            
            # 更新网格和全局最大值
            mesh_r[i, j] = avg_r
            mesh_g[i, j] = avg_g
            mesh_b[i, j] = avg_b
            
            max_r = max(max_r, avg_r)
            max_g = max(max_g, avg_g)
            max_b = max(max_b, avg_b)
    
    # 计算增益值 (全局最大值/局部平均值)
    
    # mesh_r = max_r / np.maximum(mesh_r, 1.0)  # 避免除以0
    # mesh_g = max_g / np.maximum(mesh_g, 1.0)
    # mesh_b = max_b / np.maximum(mesh_b, 1.0)
    mesh_r = max_r *maxFactor/ mesh_r  # 避免除以0
    mesh_g = max_g *maxFactor/ mesh_g
    mesh_b = max_b *maxFactor/ mesh_b
    return [mesh_r, mesh_g, mesh_b]
def lsc_calib(image, mesh_h_nums=24, mesh_w_nums=32, sample_size=3,maxFactor=1.0):
    """
    RGB图像的Lens Shading Correction校准
    参数:
        image: 输入RGB图像 (H, W, 3)
        mesh_h_nums: 垂直方向网格数 (默认49)
        mesh_w_nums: 水平方向网格数 (默认33)
        sample_size: 采样区域大小 (默认7x7)
    返回:
        mesh_r: R通道增益网格
        mesh_g: G通道增益网格
        mesh_b: B通道增益网格
    """
    if image.ndim != 2 :
        raise ValueError("输入必须是RGB图像 (H, W)")
    
    height, width = image.shape[:2]
    mesh_box_h = int(np.ceil(height / mesh_h_nums))
    mesh_box_w = int(np.ceil(width / mesh_w_nums))
    
    # 初始化网格 (存储局部平均值)
    mesh = np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)

    mesh_R= np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)
    mesh_Gr= np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)
    mesh_Gb= np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)
    mesh_B= np.zeros((mesh_h_nums + 1, mesh_w_nums + 1), dtype=np.float32)
    

    max_R = 1.0
    max_Gr = 1.0
    max_Gb = 1.0
    max_B = 1.0
    # 记录各通道全局最大值
    max_ = 1.0
    offset = sample_size // 2
    
    for i in range(mesh_h_nums + 1):
        for j in range(mesh_w_nums + 1):
            center_y = min(i * mesh_box_h, height - 1)
            center_x = min(j * mesh_box_w, width - 1)
            
            # 定义采样区域边界
            y_start = max(center_y - offset, 0)
            y_end = min(center_y + offset + 1, height)
            x_start = max(center_x - offset, 0)
            x_end = min(center_x + offset + 1, width)
            
            # 提取采样区域
            # sample_patch = image[y_start:y_end, x_start:x_end]
            R=[]
            Gr=[]
            Gb=[]
            B=[]
            for p_i in range(y_start, y_end):
                for p_j in range(x_start, x_end):
                    pixelShape=get_bayer_color(p_i, p_j)
                    match pixelShape:
                        case 'R':
                            R.append(image[p_i, p_j])
                        case 'Gr':
                            Gr.append(image[p_i, p_j])
                        case 'Gb':
                            Gb.append(image[p_i, p_j])
                        case 'B':
                            B.append(image[p_i, p_j])
                        case _:
                            pass

            # 计算各通道平均值
            R= np.array(R)
            Gr= np.array(Gr)
            Gb= np.array(Gb)
            B= np.array(B)

            avg_R = np.mean(R) if R.size > 0 else 0
            avg_Gr = np.mean(Gr) if Gr.size > 0 else 0
            avg_Gb = np.mean(Gb) if Gb.size > 0 else 0
            avg_B = np.mean(B) if B.size > 0 else 0

            # avg_ = np.mean(sample_patch)
            
            
            # 更新网格和全局最大值
            # mesh[i, j] = avg_
            mesh_R[i, j] = avg_R
            mesh_Gr[i, j] = avg_Gr
            mesh_Gb[i, j] = avg_Gb
            mesh_B[i, j] = avg_B

      
            
            # max_ = max(max_, avg_)
            max_R = max(max_R, avg_R)
            max_Gr = max(max_Gr, avg_Gr)
            max_Gb = max(max_Gb, avg_Gb)
            max_B = max(max_B, avg_B)

       
           
    
    # 计算增益值 (全局最大值/局部平均值)
    
    # mesh_r = max_r / np.maximum(mesh_r, 1.0)  # 避免除以0
    # mesh_g = max_g / np.maximum(mesh_g, 1.0)
    # mesh_b = max_b / np.maximum(mesh_b, 1.0)
    mesh_R = max_R *maxFactor/ mesh_R
    mesh_Gb=  max_Gb *maxFactor/ mesh_Gb
    mesh_Gr = max_Gr *maxFactor/ mesh_Gr
    mesh_B = max_B *maxFactor/ mesh_B
    return [mesh_R, mesh_Gr, mesh_Gb, mesh_B]


def visualize_2d_array_multi(data_list, name):
    """
    可视化多个二维数组，以2×2子图形式显示
    
    参数:
        data_list: 包含4个二维NumPy数组的列表或元组
        name: 输出文件名前缀
    """
    if len(data_list) != 4:
        raise ValueError("输入数据必须是包含4个二维数组的列表或元组！")
    
    # 创建2×2子图
    typeList= ['R','Gr','Gb','B']

    nameList=[f"Gain_{name}_{t}" for t in typeList]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),constrained_layout=True)
    
    # 遍历4个子图
    for idx, (ax, data) in enumerate(zip(axes.flat, data_list)):
        # 绘制热力图
        im = ax.imshow(data, cmap='coolwarm', interpolation='nearest')
        
        # 添加颜色条（共享同一个colorbar）
        if idx == 3:  # 只在最后一个子图添加colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), label='数值大小', shrink=0.6)
        
        # 设置标题和角点值
        height, width = data.shape
        ax.set_title(
            f"子图 {nameList[idx]}\n"
            f"左上: {data[0, 0]:.2f}, 右上: {data[0, width-1]:.2f}\n"
            f"左下: {data[height-1, 0]:.2f}, 右下: {data[height-1, width-1]:.2f}"
        )
        ax.set_xlabel("W (Width)")
        ax.set_ylabel("H (Height)")
    
    # 调整子图间距
    # plt.tight_layout()
    
    # 保存图像
    plt.savefig(
        f'{name}_2x2.png',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()
def visualize_2d_array_multi_png(data_list, name):
    """
    可视化多个二维数组，以2×2子图形式显示
    
    参数:
        data_list: 包含4个二维NumPy数组的列表或元组
        name: 输出文件名前缀
    """
    if len(data_list) != 3:
        raise ValueError("输入数据必须是包含3个二维数组的列表或元组！")
    
    # 创建2×2子图
    typeList= ['R','G','B']
     
    nameList=[f"Gain_{name}_{t}" for t in typeList]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),constrained_layout=True)
    # 隐藏第4个子图
    axes[1, 1].axis('off')
    # 遍历4个子图
    for idx, (ax, data) in enumerate(zip(axes.flat, data_list)):
        # 绘制热力图
        im = ax.imshow(data, cmap='coolwarm', interpolation='nearest')
        
        # 添加颜色条（共享同一个colorbar）
        if idx == 2:  # 只在最后一个子图添加colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), label='数值大小', shrink=0.6)
        
        # 设置标题和角点值
        height, width = data.shape
        ax.set_title(
            f"子图 {nameList[idx]}\n"
            f"左上: {data[0, 0]:.2f}, 右上: {data[0, width-1]:.2f}\n"
            f"左下: {data[height-1, 0]:.2f}, 右下: {data[height-1, width-1]:.2f}"
        )
        ax.set_xlabel("W (Width)")
        ax.set_ylabel("H (Height)")
    
    # 调整子图间距
    # plt.tight_layout()
    
    # 保存图像
    plt.savefig(
        f'{name}_2x2.png',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()
def visualize_2d_array(data,name='output'):
    """
    可视化二维数组，数值越小越蓝，越大越红
    
    参数:
        data: 输入的二维NumPy数组
    """
    plt.figure(figsize=(12, 6))
    
    # 使用热力图显示，设置colormap为从蓝到红的渐变
    im = plt.imshow(data, cmap='coolwarm', interpolation='nearest')
    
    # 添加颜色条
    plt.colorbar(im, label='数值大小')
    
    # 添加标题和坐标轴标签
    plt.xlabel("W(W*H)")
    plt.ylabel("H(W*H)")


    # 获取四个角点的坐标和值
    height, width = data.shape
    corners = {
        "左上角": data[0, 0],
        "右上角": data[0, width-1],
        "左下角": data[height-1, 0],
        "右下角": data[height-1, width-1]
    }
    
    plt.title(f"{name}\n左上: {data[0, 0]:.2f}, 右上: {data[0, width-1]:.2f}, 左下: {data[height-1, 0]:.2f}, 右下: {data[height-1, width-1]:.2f}")
    
    plt.savefig(
        f'{name}.png',  # 保存的文件名
        dpi=600,          # 分辨率
        bbox_inches='tight',  # 去除白边
        pad_inches=0.1    # 内边距
    )
    # 显示图形
    # plt.show()

def saveYaml_RGGB(result, basename):
    """
    保存校准结果到 YAML 文件
    
    参数:
        result: 包含校准结果的列表或数组
        basename: 输出文件的基本名称
    """
    # 将 NumPy 数组转换为 Python 原生数据类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # 转换所有 NumPy 数组为列表
    converted_result = [convert_numpy(arr) for arr in result]
    
    # 准备元数据
    calibration_data = {
        'description': 'LSC (Lens Shading Correction) Calibration Results',
        'type_list': ['R', 'Gr', 'Gb', 'B'],
        'data': {
            'R_channel': converted_result[0],  # R通道
            'Gr_channel': converted_result[1],
            'Gb_channel': converted_result[2],
            'B_channel': converted_result[3]
        },
        'shape': {
            'mesh_h_nums': 12,
            'mesh_w_nums': 16,
            'sample_size': 13
        }
    }
    
    # 保存为 YAML 文件
    with open(f'{basename}.yaml', 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, sort_keys=False,width=float("inf"))
    
    print(f"Calibration results saved to {basename}.yaml")
def saveYaml_RGB(result, basename):
    """
    保存校准结果到 YAML 文件
    
    参数:
        result: 包含校准结果的列表或数组
        basename: 输出文件的基本名称
    """
    # 将 NumPy 数组转换为 Python 原生数据类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # 转换所有 NumPy 数组为列表
    converted_result = [convert_numpy(arr) for arr in result]
    
    # 准备元数据
    calibration_data = {
        'description': 'LSC (Lens Shading Correction) Calibration Results',
        'type_list': ['R', 'G', 'B'],
        'data': {
            'R_channel': converted_result[0],  # R通道
            'G_channel': converted_result[1],
            'B_channel': converted_result[2]
        },
        'shape': {
            'mesh_h_nums': 12,
            'mesh_w_nums': 16,
            'sample_size': 13
        }
    }
    
    # 保存为 YAML 文件
    with open(f'{basename}.yaml', 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, sort_keys=False,width=float("inf"))
    
    print(f"Calibration results saved to {basename}.yaml")

def adjust_raw_by_blocks_vectorized_pillow(image, gainList):
    """
    向量化版本，显著提高速度
    """
    mesh_R, mesh_Gr, mesh_Gb, mesh_B = gainList
    rows,cols  = image.shape[:2]
    m = len(mesh_R) - 1
    n = len(mesh_R[0]) - 1
    
    if m == 0 or n == 0:
        return image
    
    # 创建Bayer模式掩码
    y_coords, x_coords = np.indices((rows, cols))
    bayer_mask = np.empty((rows, cols), dtype=object)
    bayer_mask[::2, ::2] = 'R'     # 偶数行偶数列
    bayer_mask[::2, 1::2] = 'Gr'    # 偶数行奇数列
    bayer_mask[1::2, ::2] = 'Gb'    # 奇数行偶数列
    bayer_mask[1::2, 1::2] = 'B'    # 奇数行奇数列
    
    # 计算块边界
    block_heights = np.linspace(0, rows, m+1, dtype=int)
    block_widths = np.linspace(0, cols, n+1, dtype=int)
    
    # 为每个像素找到对应的块索引
    i_indices = np.searchsorted(block_heights, y_coords, side='right') - 1
    j_indices = np.searchsorted(block_widths, x_coords, side='right') - 1
    
    # 确保边界情况正确
    i_indices = np.clip(i_indices, 0, m-1)
    j_indices = np.clip(j_indices, 0, n-1)
    
    # 计算归一化坐标
    y_norm = (y_coords - block_heights[i_indices]) / (block_heights[i_indices+1] - block_heights[i_indices]).astype(float)
    x_norm = (x_coords - block_widths[j_indices]) / (block_widths[j_indices+1] - block_widths[j_indices]).astype(float)
    
    # 为每个通道创建增益图
    gain_map = np.zeros_like(image, dtype=float)
    
    # 处理每个Bayer模式
    for color, mesh in [('R', mesh_R), ('Gr', mesh_Gr), ('Gb', mesh_Gb), ('B', mesh_B)]:
        mask = (bayer_mask == color)
        if not np.any(mask):
            continue
            
        # 获取四个角点的增益值
        q11 = mesh[i_indices[mask], j_indices[mask]]
        q21 = mesh[i_indices[mask], j_indices[mask]+1]
        q12 = mesh[i_indices[mask]+1, j_indices[mask]]
        q22 = mesh[i_indices[mask]+1, j_indices[mask]+1]
        
        # 双线性插值
        gain_map[mask] = ((1 - y_norm[mask]) * (1 - x_norm[mask]) * q11 +
                          (1 - y_norm[mask]) * x_norm[mask] * q21 +
                          y_norm[mask] * (1 - x_norm[mask]) * q12 +
                          y_norm[mask] * x_norm[mask] * q22)
    
    # 应用增益
    result = np.clip(image * gain_map, 0, 65535)
    return result
def LSC(image, gainList):
    """
    向量化版本，显著提高速度
    """
    mesh_R, mesh_Gr, mesh_Gb, mesh_B = gainList
    rows,cols  = image.shape[:2]
    m = len(mesh_R) - 1
    n = len(mesh_R[0]) - 1
    
    if m == 0 or n == 0:
        return image
    
    # 创建Bayer模式掩码
    y_coords, x_coords = np.indices((rows, cols))
    bayer_mask = np.empty((rows, cols), dtype=object)
    bayer_mask[::2, ::2] = 'R'     # 偶数行偶数列
    bayer_mask[::2, 1::2] = 'Gr'    # 偶数行奇数列
    bayer_mask[1::2, ::2] = 'Gb'    # 奇数行偶数列
    bayer_mask[1::2, 1::2] = 'B'    # 奇数行奇数列
    
    # 计算块边界
    block_heights = np.linspace(0, rows, m+1, dtype=int)
    block_widths = np.linspace(0, cols, n+1, dtype=int)
    
    # 为每个像素找到对应的块索引
    i_indices = np.searchsorted(block_heights, y_coords, side='right') - 1
    j_indices = np.searchsorted(block_widths, x_coords, side='right') - 1
    
    # 确保边界情况正确
    i_indices = np.clip(i_indices, 0, m-1)
    j_indices = np.clip(j_indices, 0, n-1)
    
    # 计算归一化坐标
    y_norm = (y_coords - block_heights[i_indices]) / (block_heights[i_indices+1] - block_heights[i_indices]).astype(float)
    x_norm = (x_coords - block_widths[j_indices]) / (block_widths[j_indices+1] - block_widths[j_indices]).astype(float)
    
    # 为每个通道创建增益图
    gain_map = np.zeros_like(image, dtype=float)
    
    # 处理每个Bayer模式
    for color, mesh in [('R', mesh_R), ('Gr', mesh_Gr), ('Gb', mesh_Gb), ('B', mesh_B)]:
        mask = (bayer_mask == color)
        if not np.any(mask):
            continue
            
        # 获取四个角点的增益值
        q11 = mesh[i_indices[mask], j_indices[mask]]
        q21 = mesh[i_indices[mask], j_indices[mask]+1]
        q12 = mesh[i_indices[mask]+1, j_indices[mask]]
        q22 = mesh[i_indices[mask]+1, j_indices[mask]+1]
        
        # 双线性插值
        gain_map[mask] = ((1 - y_norm[mask]) * (1 - x_norm[mask]) * q11 +
                          (1 - y_norm[mask]) * x_norm[mask] * q21 +
                          y_norm[mask] * (1 - x_norm[mask]) * q12 +
                          y_norm[mask] * x_norm[mask] * q22)
    
    # 应用增益
    result = np.clip(image * gain_map, 0, 1023).astype(np.uint16)
    return result
def AWB(image, red_gain, blue_gain):
    """
    应用红蓝通道白平衡矫正
    :param image: 输入的RAW图像 (Bayer模式)
    :param red_gain: 红色通道增益系数
    :param blue_gain: 蓝色通道增益系数
    :return: 白平衡矫正后的图像
    """
    # 创建输出图像
    balanced = image.copy()
    
    # 矫正红色通道 (R位于偶数行偶数列)
    balanced[::2, ::2] = np.clip(image[::2, ::2] * red_gain, 0, 1023).astype(np.uint16)
    
    # 矫正蓝色通道 (B位于奇数行奇数列)
    balanced[1::2, 1::2] = np.clip(image[1::2, 1::2] * blue_gain, 0, 1023).astype(np.uint16)
    
    return balanced
def adjust_rgb_by_blocks_optimized(image: np.ndarray, gainList) -> np.ndarray:
    # Split RGB channels
    b, g, r = cv2.split(image.astype('float32'))
    block_gains_r= gainList[0]
    block_gains_g= gainList[1]
    block_gains_b= gainList[2]
    # block_gains_r=1+(block_gains_r-1)*0.3
    # block_gains_g=1+(block_gains_g-1)*0.3
    # block_gains_b=1+(block_gains_b-1)*0.3
    rows, cols = image.shape[:2]
    m = len(block_gains_r) - 1
    n = len(block_gains_r[0]) - 1
    
    if m == 0 or n == 0:
        return image
    
    # Calculate block boundaries
    block_heights = np.array([i * rows // m for i in range(m + 1)])
    block_heights[-1] = rows
    
    block_widths = np.array([j * cols // n for j in range(n + 1)])
    block_widths[-1] = cols
    
    # Precompute vertical and horizontal indices
    y_indices = np.arange(rows)
    x_indices = np.arange(cols)
    
    # Find vertical block indices for each row
    i0s = np.searchsorted(block_heights, y_indices, side='right') - 1
    i0s = np.clip(i0s, 0, m-1)
    i1s = i0s + 1
    
    # Find horizontal block indices for each column
    j0s = np.searchsorted(block_widths, x_indices, side='right') - 1
    j0s = np.clip(j0s, 0, n-1)
    j1s = j0s + 1
    
    # Compute normalized coordinates
    v_norms = (y_indices - block_heights[i0s]) / (block_heights[i1s] - block_heights[i0s]).astype(float)
    u_norms = (x_indices - block_widths[j0s]) / (block_widths[j1s] - block_widths[j0s]).astype(float)
    
    # Process each channel
    for channel, block_gains in zip([r, g, b], [block_gains_r, block_gains_g, block_gains_b]):
        # Convert block gains to numpy array for vectorized operations
        gains = np.array(block_gains)
        
        # Compute bilinear interpolation for all pixels at once
        q11 = gains[i0s[:, None], j0s]  # Shape: (rows, cols)
        q21 = gains[i0s[:, None], j1s]
        q12 = gains[i1s[:, None], j0s]
        q22 = gains[i1s[:, None], j1s]
        
        # Bilinear interpolation
        gains_map = (1 - v_norms[:, None]) * ((1 - u_norms) * q11 + u_norms * q21) + \
                   v_norms[:, None] * ((1 - u_norms) * q12 + u_norms * q22)
        
        # Apply gains
        channel[:, :] = np.clip(channel * gains_map, 0, 255)
    
    return cv2.merge([b, g, r]).astype('uint8')

def adjust_raw_by_blocks(image,gainList):
    """
    Adjust RGB channels separately by blocks with bilinear interpolation.
    
    Args:
        image: Input BGR image (H x W x 3)
        block_gains_r: 2D array of gain coefficients for Red channel (m+1 x n+1)
        block_gains_g: 2D array of gain coefficients for Green channel (m+1 x n+1)
        block_gains_b: 2D array of gain coefficients for Blue channel (m+1 x n+1)
    
    Returns:
        Adjusted BGR image
    """
    # Split RGB channels
    mesh_R= gainList[0]
    mesh_Gr= gainList[1]
    mesh_Gb= gainList[2]
    mesh_B= gainList[3]

    rows, cols = image.shape[:2]
    m = len(mesh_R) - 1
    n = len(mesh_R[0]) - 1
    
    if m == 0 or n == 0:
        return image
    
    # Calculate block boundaries
    block_heights = [i * rows // m for i in range(m + 1)]
    block_heights[-1] = rows  # Ensure last boundary is image height
    
    block_widths = [j * cols // n for j in range(n + 1)]
    block_widths[-1] = cols  # Ensure last boundary is image width
    
    # Process each channel
    for y_pos in range(rows):
        # Find vertical block interval [i0, i1]
        i0, i1 = 0, m
        for i in range(1, m + 1):
            if y_pos < block_heights[i]:
                i0 = i - 1
                i1 = i
                break
        
        # Vertical normalized coordinate (0~1)
        v_norm = (y_pos - block_heights[i0]) / float(block_heights[i1] - block_heights[i0])
        
        for x_pos in range(cols):
            # Find horizontal block interval [j0, j1]
            j0, j1 = 0, n
            for j in range(1, n + 1):
                if x_pos < block_widths[j]:
                    j0 = j - 1
                    j1 = j
                    break
            
            # Horizontal normalized coordinate (0~1)
            u_norm = (x_pos - block_widths[j0]) / float(block_widths[j1] - block_widths[j0])
            print(y_pos,x_pos)
            pixelType= get_bayer_color(y_pos, x_pos)

            match pixelType:
                case 'R':
                    block_gains = mesh_R
                case 'Gr':
                    block_gains = mesh_Gr
                case 'Gb':
                    block_gains = mesh_Gb
                case 'B':
                    block_gains = mesh_B
                case _:
                    pass  # 如果不是R、Gr、Gb或B，跳过处理
            # Bilinear interpolation
            q11 = block_gains[i0][j0]  # Top-left
            q21 = block_gains[i0][j1]  # Top-right
            q12 = block_gains[i1][j0]  # Bottom-left
            q22 = block_gains[i1][j1]  # Bottom-right


            gain = (1 - v_norm) * ((1 - u_norm) * q11 + u_norm * q21) + \
                    v_norm * ((1 - u_norm) * q12 + u_norm * q22)
            
            # Apply gain to current channel
            image[y_pos, x_pos] = np.clip(image[y_pos, x_pos] * gain, 0, 1023)
    
    return image
def printCornerValues_RGGB(array):
    channelType= ['R','Gr','Gb','B']
    h,w= array[0].shape
    strTmp=""
    for arr,type_ in zip(array,channelType): 
        leftTop= arr[0, 0]
        rightTop= arr[0, w-1]
        leftBottom= arr[h-1, 0]
        rightBottom= arr[h-1, w-1]
        strTmp+=type_+":["+ f"{leftTop:.2f}, {rightTop:.2f}, {leftBottom:.2f}, {rightBottom:.2f}" +"]"
    print(f" {strTmp}")
def lenShadingCalibration(image_folder):
    full_paths, basenames = get_paths(image_folder,suffix=".pgm")
    for path,basename in zip(full_paths,basenames):
        pgm_image = read_pgm_with_opencv(path)

        if pgm_image is not None:
            print(f"图像尺寸: {pgm_image.shape},数据类型: {pgm_image.dtype},最小值: {pgm_image.min()}, 最大值: {pgm_image.max()}")  # (高度, 宽度)
            resultList=[]
            for s in range(7,15,2):
                resultTmp = lsc_calib(pgm_image, mesh_h_nums=12, mesh_w_nums=16, sample_size=s,maxFactor=1.0)
                print('当前采样点数:',s)
                printCornerValues_RGGB(resultTmp)
                resultList.append(resultTmp)
            stacked = np.stack(resultList)  
            result = np.mean(stacked, axis=0)
            visualize_2d_array_multi(result,basename)
            saveYaml_RGGB(result, basename)

def printCornerValues(array):
    channelType= ['B','G','R']
    h,w= array[0].shape
    strTmp=""
    for arr,type_ in zip(array,channelType): 
        leftTop= arr[0, 0]
        rightTop= arr[0, w-1]
        leftBottom= arr[h-1, 0]
        rightBottom= arr[h-1, w-1]
        strTmp+=type_+":["+ f"{leftTop:.2f}, {rightTop:.2f}, {leftBottom:.2f}, {rightBottom:.2f}" +"]"
    print(f" {strTmp}")


def lenShadingCalibrationFor_Png(image_folder):
    full_paths, basenames = get_paths(image_folder,suffix=".png")
    for path,basename in zip(full_paths,basenames):
        print(f"Processing image: {path}...")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # 检查图像是否成功加载
        if img is None:
            print("无法加载图像，请检查文件路径")
        if img is not None:
            print(f"图像尺寸: {img.shape},数据类型: {img.dtype},最小值: {img.min()}, 最大值: {img.max()}")  # (高度, 宽度)
            resultList=[]
            for s in range(7,15,2):
                resultTmp = lsc_calib_rgb(img, mesh_h_nums=12, mesh_w_nums=16, sample_size=s,maxFactor=1.0)
                print('当前采样点数:',s)
                printCornerValues(resultTmp)
                resultList.append(resultTmp)
            stacked = np.stack(resultList)  
            result = np.mean(stacked, axis=0)
            visualize_2d_array_multi_png(result,basename)
            saveYaml_RGB(result, basename)

def writePgm(image, basename):
    filename = f"{basename}_NewCalib.pgm"
    with open(filename, "wb") as f:
        # 写入文件头
        f.write(b"P5\n")
        f.write(b"#METADATA BEGIN\n")
        f.write(b"#BIT_PER_SAMPLE,0,10\n")
        f.write(b"#CFA_PATTERN,3,RGGB\n")
        f.write(b"#__data_type__,3,Bayer_R\n")
        f.write(b"#METADATA END\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())  # 宽度 高度
        f.write(b"1023\n")  # 关键：声明 maxval=1023
        # 写入像素数据（16 位小端序）
        f.write(image.astype(">u2").tobytes())  # "<u2" 表示小端序 uint16 ">u2" 表示小端序 uint16

def Demosaic(bayer_pgm):
    # 常见选项：COLOR_BAYER_BG2RGB, COLOR_BAYER_RG2RGB, COLOR_BAYER_GB2RGB 等
    rgb = cv2.cvtColor(bayer_pgm, cv2.COLOR_BAYER_RGGB2RGB)
    return rgb

def lenShadingCorrection(image_folder):
    full_paths, basenames = get_paths(image_folder,suffix=".pgm")
    for path,basename in zip(full_paths,basenames):
        print(f"Processing image: {path}...")   
        keyCT= getCTstr(path)
        yaml_file = fr'C:\serialPortVisualization\data\0815_1_Config\isp_sensor_raw{keyCT}.yaml'
        dataYaml = loadYaml(yaml_file)
        gainList=[]
        for key,value in dataYaml.items():
            # print(f"{key}: {value}")
            if key=='data':
                mesh_R = np.array(value['R_channel'])
                mesh_Gr = np.array(value['Gr_channel'])
                mesh_Gb = np.array(value['Gb_channel'])
                mesh_B = np.array(value['B_channel'])
                gainList.append(mesh_R)
                gainList.append(mesh_Gr)
                gainList.append(mesh_Gb)
                gainList.append(mesh_B)
        '''=============================pillow============================='''
        # pgm_image = read_pgm_with_pillow(path)
        # # pixels = pgm_image.load() 
        # data = np.array(pgm_image)
        # # print(f"图像尺寸: {data.shape}")  # (高度, 宽度)
        # print(data[0,0],data[0,1],'\n',data[1,0],data[1,1])
        
        # pixels_10bit = data >> 6
        # print(pixels_10bit[0,0],pixels_10bit[0,1],'\n',pixels_10bit[1,0],pixels_10bit[1,1])
        # result=adjust_raw_by_blocks_vectorized_pillow(pixels_10bit,gainList)

        # result = Image.fromarray(result.astype(np.uint16))

        # result.save(f'{basename}_NewCalib.pgm')
        """
        =============================opencv=============================
        """
        pgm_image = read_pgm_with_opencv(path)
        awbList={'H':(1.793,0.825),'A':(1.750,0.923),'U30':(1.897,1.031),'CWF':(1.619,1.280),'D50':(1.393,1.312),'D60':(1.311,1.385)}
        # awbList={'H':(1.818,0.863),'U30':(1.897,1.031),'CWF':(1.619,1.280),'D50':(1.406,1.321)}
        rGain,bGain=awbList[keyCT]
        result=LSC(pgm_image,gainList)
        rgb_white= AWB(result, rGain, bGain)  # 假设红蓝通道增益为1.0
        rgb=Demosaic(rgb_white)

        raw_8bit = rgb/1023 * 255  # 将16位数据转换为8位
        raw_8bit = raw_8bit.astype(np.uint8)
        cv2.imwrite(f'{basename}_rgb.jpg', raw_8bit)
        writePgm(result, basename)  
        
def lenShadingCorrectionFor_Png(image_folder,yaml_file=None):
    full_paths, basenames = get_paths(image_folder,suffix=".png")
    for path,basename in zip(full_paths,basenames):
        print(f"Processing image: {path}...")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # 检查图像是否成功加载
        if img is None:
            print("无法加载图像，请检查文件路径")
        
        key= getCTstr(path)
        yaml_file = fr'C:\serialPortVisualization\data\0815_1_A_LSC\isp_sensor_raw{key}.yaml'
        dataYaml = loadYaml(yaml_file)
        gainList=[]
        for key,value in dataYaml.items():
            # print(f"{key}: {value}")
            if key=='data':
                mesh_R = np.array(value['R_channel'])
                mesh_G = np.array(value['G_channel'])
                mesh_B = np.array(value['B_channel'])
        
                gainList.append(mesh_R)
                gainList.append(mesh_G)  # 使用平均值代替单独的Gr和Gb
                gainList.append(mesh_B)
                        
        result=adjust_rgb_by_blocks_optimized(img,gainList)
        cv2.imwrite(f'{basename}_AfterCalib.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
def readRgm2AWB(imagePath,roi=None):
    pgmImage=read_pgm_with_opencv(imagePath)
    
    if pgmImage is None:
        raise ValueError("无法读取图像文件: " + imagePath)
    
   
    height, width = pgmImage.shape
    
    # 设置ROI区域
    if roi is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = roi
        # 确保ROI在图像范围内并转换为有效坐标
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        # 确保x2 > x1且y2 > y1
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
    
    # 初始化统计变量为64位无符号整数
    r_sum = np.uint64(0)
    g1_sum = np.uint64(0)  # R行G列
    g2_sum = np.uint64(0)  # G行B列
    b_sum = np.uint64(0)
    r_count = np.uint64(0)
    g1_count = np.uint64(0)
    g2_count = np.uint64(0)
    b_count = np.uint64(0)
    
    # 遍历ROI区域内的像素
    for row in range(y1, y2):
        for col in range(x1, x2):
            pixel = np.uint64(pgmImage[row, col])  # 转换为64位无符号整数
            if row % 2 == 0:  # 偶数行
                if col % 2 == 0:  # 偶数列 - R
                    r_sum += pixel
                    r_count += 1
                else:  # 奇数列 - G
                    g1_sum += pixel
                    g1_count += 1
            else:  # 奇数行
                if col % 2 == 0:  # 偶数列 - G
                    g2_sum += pixel
                    g2_count += 1
                else:  # 奇数列 - B
                    b_sum += pixel
                    b_count += 1
    
    # 计算均值（转换为浮点数避免整数除法）
    r_mean = float(r_sum) / max(float(r_count), 1.0)
    g1_mean = float(g1_sum) / max(float(g1_count), 1.0)
    g2_mean = float(g2_sum) / max(float(g2_count), 1.0)
    b_mean = float(b_sum) / max(float(b_count), 1.0)
    
    # 合并两个G通道的均值
    g_mean = (g1_mean + g2_mean) / 2.0 if (g1_count + g2_count) > 0 else 0.0
    rGain= g_mean / r_mean if r_mean > 0 else 0.0
    bGain= g_mean / b_mean if b_mean > 0 else 0.0
    # print(f"R_mean: {r_mean:2f}, G_mean: {g_mean:2f}, B_mean: {b_mean:2f}, G1_mean: {g1_mean:2f}, G2_mean: {g2_mean:2f}")
    # print(f"rGain={rGain:3f}, bGain={bGain:3f}")
    return rGain,bGain
    # return {
    #     'R_mean': r_mean,
    #     'G_mean': g_mean,
    #     'B_mean': b_mean,
    #     'G1_mean': g1_mean,
    #     'G2_mean': g2_mean
    
    # }
def folderPocessingAWB(image_folder):
    rois=[(962,1440,1200,1672)]
    full_paths, basenames = get_paths(image_folder,suffix=".pgm")
    for path,basename in zip(full_paths,basenames):
        print(f"Processing image: {path}...")
        ct=getCTstr(basename)
        print('========================',ct,'===================')
        rbGain=[]
        for roi in rois:
            print(f"Processing ROI: {roi}")
            rGain,bGain=readRgm2AWB(path,roi)
            rbGain.append((rGain,bGain))
            print(f"ROI {roi} - R Gain: {rGain:.3f}, B Gain: {bGain:.3f}")
        # 计算平均增益
        rbGain = np.mean(rbGain, axis=0)

        print(f"Average R Gain: {rbGain[0]:.3f}, Average B Gain: {rbGain[1]:.3f}")
def main():

    """"=============================标定代码============================="""
    # folder_path= r'C:\serialPortVisualization\data\0815_2_LSC'
    # lenShadingCalibration(folder_path)
    """"=============================应用代码============================="""
    folder_path=r'C:\serialPortVisualization\data\0815_2' 
    lenShadingCorrection(folder_path)

    

    """=====================================pngLSC====================================="""
    # folder_path=r'C:\serialPortVisualization\data\0812_1'
    # lenShadingCorrectionFor_Png(folder_path)

    ''''=====================png标定========================'''
    # folder_path=r'C:\serialPortVisualization\data\0812_1_Calib'
    # lenShadingCalibrationFor_Png(folder_path)
    '''=====================pgm转raw========================'''

    """===============AWB标定=================="""
    # folder_path=r'C:\serialPortVisualization\data\0815_2_AfterWhite'
    # folderPocessingAWB(folder_path)
    







main()
