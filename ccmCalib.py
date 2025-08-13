import numpy as np
from scipy.optimize import minimize
import cv2
import os
import yaml
IDEAL_RGB = np.array([
      [0.447	,0.317	,0.265],
      [0.764	,0.58	,0.501],
      [0.364	,0.48	,0.612],
      [0.355	,0.422	,0.253],
      [0.507	,0.502	,0.691],
      [0.382	,0.749	,0.67 ],
      [0.867	,0.481	,0.187],
      [0.277	,0.356	,0.668],
      [0.758	,0.322	,0.382],
      [0.361	,0.225	,0.417],
      [0.629	,0.742	,0.242],
      [0.895	,0.63	,0.162],
      [0.155	,0.246	,0.576],
      [0.277	,0.588	,0.285],
      [0.681	,0.199	,0.223],
      [0.928	,0.777	,0.077],
      [0.738	,0.329	,0.594],
      [0	    ,0.54	,0.66 ],
      [0.96	    ,0.962	,0.95 ],
      [0.786	,0.793	,0.793],
      [0.631	,0.639	,0.64 ],
      [0.474	,0.475	,0.477],
      [0.324	,0.33	,0.336],
      [0.191	,0.194	,0.199],
    ])  
# IDEAL_LINEAR_RGB = IDEAL_RGB * 255.0  # 逆Gamma处理后的理想RGB值
IDEAL_LINEAR_RGB = np.power(IDEAL_RGB, 2.2) * 255.0  # 逆Gamma处理后的理想RGB值
def npToString(arr):
    return np.array2string(arr, suppress_small=True, precision=4, floatmode='fixed')
def readCSV(file_path):
    import csv
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)  # 将CSV转换为列表
        
        # 获取第3行第2列(索引从0开始)
        colorArr=[]
        startRow= 52
        for i in range(24):
            rowIdx= startRow + i
            R = float(rows[rowIdx][1])*255.0
            G = float(rows[rowIdx][2])*255.0
            B = float(rows[rowIdx][3])*255.0
            colorArr.append([float(R), float(G), float(B)])
        print(colorArr)
        return np.array(colorArr, dtype=np.float32)
def RGB2Lab(rgb):
    """将RGB颜色转换为Lab颜色空间"""
    # 这里可以使用OpenCV或其他库进行转换
    if (rgb.ndim != 2) or (rgb.shape[1] != 3):
        raise ValueError("输入必须是形状为(N, 3)的RGB数组")
    rgb = rgb[np.newaxis,:, : ]
    rgb = rgb.astype(np.float32) / 255.0  # 转换为0-1范围
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0]
class CCM_3x3:
    def __init__(self,input,output):
        self.input = input
        self.output = output
        self.ccm = np.eye(3, 3)  # 初始化为单位矩阵
        # 仅用shape进行判断
        if input.shape != output.shape:
            raise ValueError("input和output的形状必须相同")
        if input.shape[1] != 3:
            raise ValueError("最后一个维度必须是RGB颜色")
        self.m, self.n = input.shape[:2]  # 获取输入图像的形状
       
    def loss(self, x, input, output):
        input=input.T
        output=output.T
        ccm = x.reshape(3, 3)  # 将扁平化的参数恢复为3x3矩阵
        predicted = np.dot(ccm,input )  # 应用颜色校正
        # print(f"===predicted===\n{npToString(predicted)}")
        # predicted_Lab = RGB2Lab(predicted)    #N*3
        # print(f"===predicted_Lab===\n{npToString(predicted_Lab)}")
        # output_Lab = RGB2Lab(output)            #N*3
        sumTmp=np.sqrt(np.sum((predicted - output)**2,axis=0))
        # print(f"===sumTmp===\n{npToString(sumTmp)}")
        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值
        C = np.zeros((3, 9))
        for i in range(3):
            C[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # 约束条件: CCM矩阵的每一行之和为1
        constraints = {
            'type': 'eq', 
            'fun': lambda x: C @ x - np.ones(3),
        } if len(C) > 0 else None
        bounds = [(-10, 10) for _ in range(9)]
        result = minimize(
            self.loss,  # 包装loss函数
            x,  
            args=(self.input, self.output),
            constraints=constraints,
            bounds=bounds,
            method='trust-constr',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'disp': True}
        )
        # 打印优化结果的详细信息
        print("\n=== Optimization Result ===")
        print(f"Success: {result.success}")  # 是否成功收敛
        print(f"Message: {result.message}")  # 状态描述
        print(f"Final loss value: {result.fun}")  # 最终损失值
        print(f"Iterations: {result.nit}")  # 迭代次数
        print(f"Function evaluations: {result.nfev}")  # 损失函数调用次数

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 3)
        print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
    


def ccmInfer(csv_path):
    input_arr=readCSV(csv_path)
    ccmCalib= CCM_3x3(input_arr, IDEAL_LINEAR_RGB)
    ccm= ccmCalib.infer_image()
    return ccm

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
def getCTstr(file_path):
    file_path=str(file_path)
    if 'U30' in file_path:
        return 'U30'
    elif 'CWF' in file_path:
        return 'CWF'
    elif 'D50' in file_path:
        return 'D50'
    elif 'H' in file_path:
        return 'H'
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
def folder_to_csv(folder_path):
    full_paths, basenames = get_paths(folder_path, suffix=".csv")
    ccmDict = {}
    for path, basename in zip(full_paths, basenames):

        ct_str = getCTstr(basename)
        print(f"正在处理: {path}，CT类型: {ct_str}")
        ccmTmp=ccmInfer(path)
        ccmTmp_oneline=ccmTmp.flatten()
        ccmDict[ct_str] =convert_numpy(ccmTmp) 
        ccmDict[f"{ct_str}_Nx1"] = convert_numpy(ccmTmp_oneline)

    saveYaml(ccmDict, 'ccmDict')
    
def main():
        
    folder_path= r'C:\serialPortVisualization\data\0812_4'
    folder_to_csv(folder_path)

   
    
    # input_arr=readCSV(r'C:\serialPortVisualization\data\0811ccmColor\ResultsU30\current_isp_configU30__summary.csv')

    # ccmCalib= CCM_3x3(input_arr, IDEAL_RGB)
    # ccm= ccmCalib.infer_image()
    # print(f"===A@B的颜色校正矩阵===\n{npToString((A@ccm))}")

main()  