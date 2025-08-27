import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import least_squares
import cv2
import os
import yaml
from tools import *
from skimage.color import rgb2lab
import csv

def reverseGamma(img):
    mask = img <= 0.04045
    linear = np.zeros_like(img)
    linear[mask] = img[mask] / 12.92
    linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    linear=np.clip(linear,0,1)

    return linear
def Gamma(img):
    
    # img_encoded = cv2.imread(folderPath)
    mask = img <= 0.0031308
    srgb = np.zeros_like(img)
    srgb[mask] = img[mask] * 12.92
    srgb[~mask] = 1.055 * (img[~mask] ** (1/2.4)) - 0.055
    return srgb
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
# IDEAL_LINEAR_RGB = np.power(IDEAL_RGB,1/2.2) # 逆Gamma处理后的理想RGB值
# IDEAL_LINEAR_RGB = IDEAL_RGB**(2.2)# 逆Gamma处理后的理想RGB值
# IDEAL_LINEAR_RGB=np.clip(IDEAL_LINEAR_RGB,0,1)  # 确保在0-1范围内
IDEAL_LINEAR_RGB = reverseGamma(IDEAL_RGB) # 逆Gamma处理后的理想RGB值
gammaRGB=Gamma(IDEAL_LINEAR_RGB) # Gamma处理后的理想RGB值
print(f"===IDEAL_LINEAR_RGB===\n{npToString(IDEAL_LINEAR_RGB)}")
print(f"===gammaRGB===\n{npToString(gammaRGB)}")

def readCSV(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)  # 将CSV转换为列表
        
        # 获取第3行第2列(索引从0开始)
        colorArr=[]
        startRow=58
        for i in range(24):
            rowIdx= startRow + i
            R = float(rows[rowIdx][1])
            G = float(rows[rowIdx][2])
            B = float(rows[rowIdx][3])
            colorArr.append([float(R), float(G), float(B)])
        print(colorArr)
        return np.array(colorArr, dtype=np.float64)
def rgb_to_lab(rgb):
    """
    将sRGB值转换为CIE Lab
    输入: 
        rgb : [R, G, B] (0-255整数或0-1浮点数)
    输出: 
        Lab: [L, a, b] (L: 0-100, a/b: -128~127)
    """
    # # 如果输入是0-255整数，先归一化为0-1浮点
    # if isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8:
    #     rgb = rgb.astype(np.float32) / 255.0
    # elif isinstance(rgb, (list, tuple)):
    #     rgb = np.array(rgb, dtype=np.float32) / 255.0
    
    # 调用scikit-image的转换函数
    lab = rgb2lab(rgb[np.newaxis, np.newaxis, :]).squeeze()
    return lab
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
        # self.ccm = np.ones(3, 3)  # 初始化为单位矩阵
        self.ccm = np.eye(3)  # 初始化为单位矩阵
        
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
        predicted = np.dot(ccm,input)  # 应用颜色校正

        # labPredicted= rgb_to_lab(predicted.T)  # 转换为Lab颜色空间
        # labOutput = rgb_to_lab(output.T)  # 转换为Lab颜色空间
        # labPredicted=labPredicted.T
        # labOutput=labOutput.T
        # sumTmp=np.sum((labPredicted - labOutput)**2,axis=0)
        # sumTmp[18]*=3
        # sumTmp[19]*=3
        sumTmp=np.sum((predicted - output)**2,axis=0)
        

        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        # return((labPredicted - labOutput)**2).flatten()
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
            # constraints=constraints,
            # bounds=bounds,
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-12,'disp': True}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 3)
        print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
    

class CCM_3x4:
    def __init__(self,input,output):
        self.input = input
        self.output = output
        # self.ccm = np.ones(3, 3)  # 初始化为单位矩阵
        self.ccmR = np.eye(3)  # 初始化为单位矩阵
        self.ccmT=np.zeros((1,3)) # 初始化为零向量
        self.ccm = np.vstack((self.ccmR, self.ccmT))
        # 仅用shape进行判断
        if input.shape != output.shape:
            raise ValueError("input和output的形状必须相同")
        if input.shape[1] != 3:
            raise ValueError("最后一个维度必须是RGB颜色")
        self.m, self.n = input.shape[:2]  # 获取输入图像的形状
       
    def loss(self, x, input, output):
        input=input.T
        output=output.T
        ccm = x.reshape(4, 3)  # 将扁平化的参数恢复为3x3矩阵
        predicted = np.dot(ccm[:3,:],input) +ccm[3,:].reshape(-1,1) # 应用颜色校正

        # labPredicted= rgb_to_lab(predicted.T)  # 转换为Lab颜色空间
        # labOutput = rgb_to_lab(output.T)  # 转换为Lab颜色空间
        # labPredicted=labPredicted.T
        # labOutput=labOutput.T
        # sumTmp=np.sqrt(np.sum((labPredicted - labOutput)**2,axis=0))
        sumTmp=np.sqrt(np.sum((predicted - output)**2,axis=0))
        sumTmp[20]*=3
        sumTmp[1]*=3
      
      
        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        # return((labPredicted - labOutput)**2).flatten()
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值


        C = np.zeros((3, 12))
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
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-12,'disp': True}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(4, 3)
        print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
    

def ccmInfer(csv_path):
    input_arr=readCSV(csv_path)
    # scale = 1.0/input_arr[18]
    # input_arr = input_arr * scale  # 将输入数据缩放到0-1范围
    # input_arr = input_arr*255

    # ccmCalib= CCM_3x4(input_arr, IDEAL_LINEAR_RGB)
    # ccmCalib= CCM_3x4(input_arr, IDEAL_RGB)
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

def readCCM3x3(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)  # 将CSV转换为列表
        
        # 获取第3行第2列(索引从0开始)
        ccmArr=[]
        startRow= 125
        for i in range(3):
            rowIdx= startRow + i
            Ci0 = float(rows[rowIdx][0])
            Ci1 = float(rows[rowIdx][1])
            Ci2  = float(rows[rowIdx][2])
            ccmArr.append([float(Ci0), float(Ci1), float(Ci2)])
        print(ccmArr)
        return np.array(ccmArr, dtype=np.float64)
def readCCM3x4(folderPath):
    full_paths, basenames = get_paths(folderPath, suffix=".csv")
    
def folder_to_csv(folder_path):
    full_paths, basenames = get_paths(folder_path, suffix=".csv")
    ccmDict = {}
    for path, basename in zip(full_paths, basenames):

        ct_str = getCTstr(basename)
        print(f"正在处理: {path}，CT类型: {ct_str}")
        """标定算法"""
        ccmTmp=ccmInfer(path)

        """读取CSV文件"""
        # ccmTmp=readCCM3x3(path)
        # ccmTmp=ccmTmp.T
        '''保存ccm'''
        ccmTmp_oneline=ccmTmp.flatten()
        ccmDict[ct_str] =convert_numpy(ccmTmp) 
        ccmDict[f"{ct_str}_Nx1"] = convert_numpy(ccmTmp_oneline)

    saveYaml(ccmDict, './config/ccmYaml')
    
def main():
        
    folder_path= r'C:\WorkSpace\serialPortVisualization\data\0827_ColorChecker'
    folder_to_csv(folder_path)

   
    
    # input_arr=readCSV(r'C:\serialPortVisualization\data\0811ccmColor\ResultsU30\current_isp_configU30__summary.csv')

    # ccmCalib= CCM_3x3(input_arr, IDEAL_RGB)
    # ccm= ccmCalib.infer_image()
    # print(f"===A@B的颜色校正矩阵===\n{npToString((A@ccm))}")

main()  