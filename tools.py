import os
from pathlib import Path
import yaml
import numpy as np
def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
def saveYaml(dictData, basename):
    
    
    # 保存为 YAML 文件
    with open(f'{basename}.yaml', 'w') as f:
        yaml.dump(dictData, f, default_flow_style=None, sort_keys=False,width=float("inf"))
    
    print(f"Calibration results saved to {basename}.yaml")
def loadYaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data
def print_file_tree(directory, indent=''):
    """递归打印目录的文件树结构"""
    path = Path(directory)
    if not path.exists():
        print(f"错误：目录 '{directory}' 不存在")
        return
    
    # 打印当前目录名
    print(f"{indent}📂 {path.name}/")
    
    # 获取所有子项并排序（目录在前，文件在后）
    items = sorted(os.listdir(path))
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
    files = [item for item in items if not os.path.isdir(os.path.join(path, item))]
    
    # 递归打印子目录
    for d in dirs:
        print_file_tree(os.path.join(path, d), indent + "    ")
    
    # 打印文件
    for f in files:
        print(f"{indent}    📄 {f}")
def getCTstr(file_path):
    file_path=str(file_path)
    if 'U30' in file_path:
        return 'U30'
    elif 'CWF' in file_path:
        return 'CWF'
    elif 'D50' in file_path:
        return 'D50'
    elif 'H_' in file_path:
        return 'H'
    elif 'A_' in file_path:
        return 'A'
    elif 'D60' in file_path:
        return 'D60'
def get_paths(folder_name, suffix=".csv"):
    """
    递归获取指定文件夹及其子目录中的所有suffix图片路径及不带后缀的文件名
    
    参数:
        folder_name (str): 目标文件夹名称（如"x"）
        suffix (str): 文件后缀，默认".jpg"
        
    返回:
        tuple: (完整路径列表, 不带后缀的文件名列表)，如(
                ["x/images/pic1.jpg", "x/subdir/pic2.jpg"], 
                ["pic1", "pic2"]
               )
    """
    full_paths = []
    basenames = []
    
    try:
        if not os.path.exists(folder_name):
            raise FileNotFoundError(f"目录不存在: {folder_name}")
            
        # 使用 os.walk 递归遍历所有子目录
        for root, dirs, files in os.walk(folder_name):
            for f in files:
                if f.lower().endswith(suffix):
                    file_path = os.path.join(root, f)
                    if os.path.isfile(file_path):
                        full_paths.append(file_path)
                        basenames.append(os.path.splitext(f)[0])
                        
        return full_paths, basenames
        
    except Exception as e:
        print(f"错误: {e}")
        return [], []
def npToString(arr):
    return np.array2string(arr, suppress_small=True, precision=4, floatmode='fixed')