import numpy as np
from scipy.optimize import minimize
import cv2
import os
import yaml
from tools import *
def reverseGamma(img):
    mask = img <= 0.04045
    linear = np.zeros_like(img)
    linear[mask] = img[mask] / 12.92
    linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return linear
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
IDEAL_LINEAR_RGB = np.power(IDEAL_RGB,1/2.2) # 逆Gamma处理后的理想RGB值
IDEAL_LINEAR_RGB = reverseGamma(IDEAL_RGB) # 逆Gamma处理后的理想RGB值

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
            R = float(rows[rowIdx][1])
            G = float(rows[rowIdx][2])
            B = float(rows[rowIdx][3])
            colorArr.append([float(R), float(G), float(B)])
        print(colorArr)
        return np.array(colorArr, dtype=np.float64)
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
     
        sumTmp=np.sqrt(np.sum((predicted - output)**2,axis=0))
        # sumTmp[20]*=3
        # sumTmp[23]*=3
        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        # return (predicted - output).flatten()
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值


        C = np.zeros((3, 9))
        for i in range(3):
            C[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # Cin=np.zeros((1, 9))
        # Cin[0, 4]=-1
        # Cin[0, 8]=1

        constraints = []
        # 约束条件: CCM矩阵的每一行之和为1
        constraints.append( {
            'type': 'eq', 
            'fun': lambda x: C @ x - np.ones(3),
        } )
        

        # constraints.append( {
        #     'type': 'ineq', 
        #     'fun': lambda x: Cin @ x,
        # } )
        '''=====================最小二乘============================='''
        # from scipy.optimize import least_squares

        # 假设你的 self.loss 函数返回残差（residuals）而不是标量损失值
        # result = least_squares(
        #     self.loss,  # 这个函数现在应该返回残差向量而不是标量
        #     x,
        #     args=(self.input, self.output),
        #     # bounds=bounds,  # least_squares 支持 bounds
        #     method='trf',  # 或 'lm'（Levenberg-Marquardt，无约束时使用）
        #     max_nfev=100000,
        #     verbose=2
        # )
        '''=====================能量项优化============================='''
        # bounds = [(-4, 4) for _ in range(9)]
        result = minimize(
            self.loss,  # 包装loss函数
            x,  
            args=(self.input, self.output),
            constraints=constraints,
            # bounds=bounds,
            method='trust-constr',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'disp': True}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 3)
        print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
    


def ccmInfer(csv_path):
    input_arr=readCSV(csv_path)
    # input_arr = input_arr/input_arr[18]
    # input_arr = input_arr*255
    ccmCalib= CCM_3x3(input_arr, IDEAL_RGB)
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
        
    folder_path= r'C:\serialPortVisualization\data\0815_1_ColorChecker_Total_'
    folder_to_csv(folder_path)

   
    
    # input_arr=readCSV(r'C:\serialPortVisualization\data\0811ccmColor\ResultsU30\current_isp_configU30__summary.csv')

    # ccmCalib= CCM_3x3(input_arr, IDEAL_RGB)
    # ccm= ccmCalib.infer_image()
    # print(f"===A@B的颜色校正矩阵===\n{npToString((A@ccm))}")

main()  