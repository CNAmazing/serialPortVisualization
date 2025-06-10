# 读取txt文件并按等号分割内容的脚本
import matplotlib.pyplot as plt
import numpy as np
def read_and_split_by_equal(filename):
    """
    读取txt文件并按等号分割每行内容
    
    参数:
        filename: 要读取的txt文件路径
        
    返回:
        包含分割后内容的列表，每个元素是一个(key, value)元组
    """
    result = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                # 跳过空行和注释行(以#开头的行)
                if not line or line.startswith('#'):
                    continue
                # 按等号分割
                if '=' in line:
                    key, value = line.split('=', 1)  # 只分割第一个等号
                    key = key.strip()
                    value = value.strip()
                    result.append((key, value))
                else:
                    print(f"警告: 行 '{line}' 不包含等号，已跳过")
        return result
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    file_path = "ReceivedTofile-COM3-2025_6_9_15-35-38.txt"  # 替换为你的文件路径
    parsed_data = read_and_split_by_equal(file_path)
    newGain=[]
    curGain=[]
    newExposure=[]
    curExposure=[]
    if parsed_data:
        for key, value in parsed_data:
            value=float(value)
            match key:
                case 'curExposure':
                    curExposure.append(value)
                case 'newExposure':
                    newExposure.append(value)
                case 'curGain':
                    curGain.append(value)
                case 'newGain':
                    newGain.append(value)
    # errorExposure=np.array(newExposure)-np.array(curExposure)
    
    plt.figure(figsize=(12, 8))
    plt.plot(newExposure, marker='o', linestyle='-', color='r')
    plt.plot(curExposure, marker='o', linestyle='-', color='b')
    
    # plt.title(title)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
