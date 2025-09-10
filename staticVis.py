# 读取txt文件并按等号分割内容的脚本
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict
def read_and_split_by_equal1(filename):
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
def read_and_split_by_equal(filename):
    """
    读取日志文件并按等号分割每行内容，处理带有时间戳和前缀的格式
    
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
                
                # 处理带有时间戳和前缀的行
                parts = line.split()
                if len(parts) >= 5 and '=' in parts[-1]:
                    # 最后一部分应该是 key=value
                    key_value = parts[-1].split('=', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        result.append((key, value))
                elif '=' in line:
                    # 普通格式的处理（原逻辑）
                    key, value = line.split('=', 1)  # 只分割第一个等号
                    key = key.strip()
                    value = value.strip()
                    result.append((key, value))
                else:
                    print(f"警告: 行 '{line}' 不包含有效的键值对，已跳过")
        return result
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
def generateDict(keyList):
    dataDict={}
    for key in keyList:
        
        dataDict[key]=[]
    return dataDict
def updateDict(dataDict,parsedData):
    if not parsedData:
        raise ValueError('parsedData为空')
    for key,value in parsedData:
        try:
            value=float(value)
            if key in dataDict:
                dataDict[key].append(value)
        except:
            pass
def pltSingleImage(dataDict):
    plt.figure(figsize=(12, 8))
    for i, (key,value) in enumerate(dataDict.items()):
        plt.plot(value, marker='o', linestyle='-', label=f" {key}")
        # plt.plot(dataDict.values[1], marker='o', linestyle='-', color='b')
        # 在每个数据点上标注数值
        for x, y in enumerate(value):
            plt.text(x, y, f"{x,int(y)}", ha='center', va='bottom')  
    # plt.title(title)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
def pltMultiImage(dataDict,tital='tital',isSavePltImage=False):
    fig, axes = plt.subplots(1, len(dataDict), figsize=(16, 8))
    if len(dataDict) == 1:
        axes = [axes]  
    for ax, (name, value), color in zip(axes, dataDict.items(), plt.cm.tab10.colors):
        line, = ax.plot(value,marker='o', linestyle='-' ,label=name, color=color)
        for x, y in enumerate(value):
            if x==0:
             ax.text(x, y, f"{x},{float(y):.2f}", ha='center', va='bottom')  

            if x>=1 and value[x-1]!=y:
             ax.text(x, y, f"{x},{float(y):.2f}", ha='center', va='bottom')  
        # ax.set_title(name)
        ax.set_xlabel("time")
        # ax.set_ylabel(f"{name}")
        ax.legend()
    fig.suptitle(tital)
    plt.tight_layout()
    if isSavePltImage:
        plt.savefig(f"{tital}.png")
    plt.show()
def singleDatInfer(file_path, title=None,isSavePltImage=False):
    parsed_data = read_and_split_by_equal(file_path)
    keyList = ['curExposure','curGain','avgL']
    # keyList = ['curExposure','curGain','avgL_d','L','roundedCCT']
    # keyList = ['roundedCCT','isgainR','isgainG','isgainB',]
    # keyList = ['roundedCCT','noGainR','noGainG','noGainB',]
    dataDict = generateDict(keyList)
    updateDict(dataDict, parsed_data)
    pltMultiImage(dataDict, tital=title,isSavePltImage=isSavePltImage)

def get_paths(folder_name,ext=".DAT"):
    """
    获取指定文件夹下images子目录中的所有.jpg图片路径及不带后缀的文件名
    
    参数:
        folder_name (str): 目标文件夹名称（如"x"）
        
    返回:
        tuple: (完整路径列表, 不带后缀的文件名列表)，如(
                ["x/images/pic1.jpg", "x/images/pic2.jpg"], 
                ["pic1", "pic2"]
               )
    """
    full_paths = []
    basenames = []
    
    try:
        images_dir = folder_name
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
            
        for f in os.listdir(images_dir):
            if f.endswith(ext):
                file_path = os.path.join(images_dir, f)
                if os.path.isfile(file_path):
                    full_paths.append(file_path)
                    # 去掉.jpg后缀获取纯文件名
                    basenames.append(os.path.splitext(f)[0])
                    
        return full_paths, basenames
        
    except Exception as e:
        print(f"错误: {e}")
        return [], []

def logVis(log_file_path):
    frame_data = []
    current_frame = defaultdict(str)
    
    # 定义要捕捉的模式
    patterns = {
        'avgL_d': r'avgL_d = (\d+\.\d+)',
        'curGain': r'curGain =(\d+)',
        'curExposure': r'curExposure =(\d+)',
        'newExposure': r'newExposure =(\d+)',
        'isEqualPrevExposure': r'isEqualPrevExposure =(\d+)',
        'frame_count': r'framecnt = (\d+)',
        # 'luma_target': r'LumaTar=(\d+)',
        # 'current_luma': r'CurLuma=(\d+)'
        'Converged':r'Converged = (Yes|No)',
        'OutFlag': r' out_img_flag = (\d+)',
    }
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # 检查是否是新的帧开始
            frame_match = re.search(patterns['frame_count'], line)
            if frame_match:
                if current_frame:  # 如果已经有数据，保存当前帧
                    frame_data.append(dict(current_frame))
                    current_frame = defaultdict(str)
                current_frame['frame_count'] = frame_match.group(1)
            
            # 检查其他关键信息
            for key, pattern in patterns.items():
                if key == 'frame_count':
                    continue  # 已经处理过了
                match = re.search(pattern, line)
                if match:
                    current_frame[key] = match.group(1)
    
    # 添加最后一帧数据
    if current_frame:
        frame_data.append(dict(current_frame))
    
 
    for i, frame in enumerate(frame_data):
        if not frame.values():
            continue
        print(f"Frame {i + 1}:")
        for key, value in frame.items():
            if key=="frame_count":
                continue
            print(f" {key}: {value}")

def fitC(G,EV,t):
    c=G*t/2**(EV)
    print(f"c={c}")
    print(f"c={c/(G*t)}")
# 使用示例
if __name__ == "__main__":
    # file_path = r"C:\serialPortVisualization\data\0626\SAVE2025_6_26_9-12-39.DAT"  # 替换为你的文件路径
    # singleDatInfer(file_path)

    """
    vis
    """
    full_paths,basenames = get_paths(r"C:\serialPortVisualization\data\0908")
    for path, name in zip(full_paths, basenames):
        singleDatInfer(path,name,isSavePltImage=True)


    """
    logVis
    """
    # logVis(r"C:\serialPortVisualization\data\0711_8\SAVE2025_7_11_16-54-59.DAT")
    # parsed_data = read_and_split_by_equal(file_path)
    # # keyList=['curExposure','newExposure','curGain','newGain',"avgL"]
    # keyList=['curExposure','curGain',"avgL"]
    # dataDict=generateDict(keyList)
    # updateDict(dataDict,parsed_data)
    # # errorExposure=np.array(newExposure)-np.array(curExposure)
    # # pltSingleImage(dataDict)
    # pltMultiImage(dataDict)

    """
    fitC
    """
    # data=[[4000,56/8,33173],
    #       [8800,56/13,33173],
    #       [1000,56/9,12544]
    #       ]
    # for d in data:
    #     fitC(*d)