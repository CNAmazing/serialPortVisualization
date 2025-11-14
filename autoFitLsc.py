import sys
import os
from scipy.optimize import minimize, curve_fit,least_squares
import cv2
import numpy as np
from tools import *
from skimage.color import rgb2lab,deltaE_ciede2000,deltaE_cie76
from deap import base, creator, tools, algorithms
from functools import partial
import time
from functools import wraps
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000, delta_e_cie1976
from colormath.color_conversions import convert_color
import bm3d
from scipy.ndimage import median_filter
#stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
#stage_arg=bm3d.BM3DStages.ALL_STAGES
def rgbError_detalE2000(rgb1, rgb2):
    lab1=rgb2lab(rgb1)
    lab2=rgb2lab(rgb2)
    delta_e = deltaE_ciede2000(lab1, lab2)
    return delta_e
def rgbError_detalEcie76(rgb1, rgb2):
    lab1=rgb2lab(rgb1)
    lab2=rgb2lab(rgb2)
    delta_e = deltaE_cie76(lab1, lab2)
    return delta_e
def median_filter_NR(arr, size=5):
    """
    使用scipy实现中值滤波
    
    参数:
    arr: 输入numpy数组
    size: 滤波核大小（奇数）
    
    返回:
    滤波后的数组
    """
    arr[::2, ::2] =median_filter(arr[::2, ::2], size=size)
    arr[::2, 1::2] =median_filter(arr[::2, 1::2], size=size)
    arr[1::2, ::2] =median_filter(arr[1::2, ::2], size=size)
    arr[1::2, 1::2] =median_filter(arr[1::2, 1::2], size=size)
    return arr
def bm3d_denoise(img_float, sigma=10, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING): #stage_arg=bm3d.BM3DStages.   ALL_STAGES     HARD_THRESHOLDING
    """
    使用BM3D算法进行10位精度图像去噪
    :param img_float: 输入图像，应为10位精度的浮点型数组 (范围0-1023)
    :param sigma: 噪声标准差估计 (基于10位范围)
    :param stage_arg: 处理阶段 (HARD_THRESHOLDING 或 ALL_STAGES)
    :return: 去噪后的10位图像 (范围0-1023)
    """
    # 将10位图像归一化到[0,1]范围
    img_normalized = img_float.astype(np.float32) / 1023.0
    
    # 使用BM3D去噪 (sigma_psd需要相应调整)
    denoised_img = bm3d.bm3d(img_normalized, 
                             sigma_psd=sigma/1023,  # 噪声标准差相对于10位范围
                             stage_arg=stage_arg)
    
    # 转换回10位图像
    denoised_img = np.clip(denoised_img * 1023, 0, 1023).astype(np.uint16)
    
    return denoised_img

def AWB_RGB(image, awbParam):
    """
    应用红蓝通道白平衡矫正
    :param image: 输入的RAW图像 (Bayer模式)
    :param red_gain: 红色通道增益系数
    :param blue_gain: 蓝色通道增益系数
    :return: 白平衡矫正后的图像
    """
    rGain,grGain,gbGain,bGain = awbParam
    # 创建输出图像
    balanced = image.copy()
    balanced[:,:,0]= np.clip(image[:,:,0] * bGain, 0, 255).astype(np.float64)
    balanced[:,:,1]= np.clip(image[:,:,1] * ((grGain+gbGain)/2), 0, 255).astype(np.float64)
    balanced[:,:,2]= np.clip(image[:,:,2] * rGain, 0, 255).astype(np.float64)
    return balanced
def AWB(image, awbParam):
    """
    应用红蓝通道白平衡矫正
    :param image: 输入的RAW图像 (Bayer模式)
    :param red_gain: 红色通道增益系数
    :param blue_gain: 蓝色通道增益系数
    :return: 白平衡矫正后的图像
    """
    rGain,grGain,gbGain,bGain = awbParam
    # 创建输出图像
    balanced = image.copy()
    
    # 矫正红色通道 (R位于偶数行偶数列)
    balanced[::2, ::2] = np.clip(image[::2, ::2] * rGain, 0, 1023).astype(np.float64)
    balanced[::2, 1::2] = np.clip(image[::2, 1::2] * grGain, 0, 1023).astype(np.float64)
    balanced[1::2, ::2] = np.clip(image[::2, 1::2] * gbGain, 0, 1023).astype(np.float64)
    # 矫正蓝色通道 (B位于奇数行奇数列)
    balanced[1::2, 1::2] = np.clip(image[1::2, 1::2] * bGain, 0, 1023,).astype(np.float64)
    
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
def LSC(image, gainList, strength=[1.0,1.0,1.0,1.0],):
    """
    向量化版本，显著提高速度
    """

    mesh_R, mesh_Gr, mesh_Gb, mesh_B = gainList
    mesh_R=1+(mesh_R-1)*strength[0]
    mesh_Gr=1+(mesh_Gr-1)*strength[1]
    mesh_Gb=1+(mesh_Gb-1)*strength[2]
    mesh_B=1+(mesh_B-1)*strength[3]

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
    result = np.clip(image * gain_map, 0, 1023).astype(np.float64)
    return result
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
def Demosaic(bayer_pgm):
    # 常见选项：COLOR_BAYER_BG2RGB, COLOR_BAYER_RG2RGB, COLOR_BAYER_GB2RGB 等
    bayer_pgm= bayer_pgm.astype(np.uint16)  # 确保数据类型为 uint16
    rgb = cv2.cvtColor(bayer_pgm, cv2.COLOR_BAYER_BGGR2RGB)
    return rgb
def BLC(img,blcParam=16):
    img-= blcParam 
    # img = np.clip(img, 0, 1023)  # 确保像素值在有效范围内
    np.clip(img, 0, 1023,out=img)  # 确保像素值在有效范围内
    return img
def ccmApply(img,ccm):
    
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    np.dot(rgb_flat, ccm.T,out=rgb_flat)  # 等价于 (ccm @ rgb_flat.T).T
    rgb_flat = rgb_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return rgb_flat
class CCM_3x3_6variables:
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

        Cons1 = np.zeros((3, 9))
        for i in range(3):
            Cons1[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # Cin=np.zeros((1, 9))
        # Cin[0, 4]=-1
        # Cin[0, 8]=1

        self.constraints = []
        # 约束条件: CCM矩阵的每一行之和为1
        self.constraints.append( {
            'type': 'eq', 
            'fun': lambda x: Cons1 @ x - np.ones(3),
        } )
    def loss(self, x, input, output):
        input=input.T
        output=output.T
        # print(input.shape,output.shape)
        ccm=np.zeros((3,3))
        ccm[:,:2]= x.reshape(3,2)
        ccm[:,2]= 1-ccm[:,0]-ccm[:,1]
        # print('ccm',ccm)
        # ccm = x.reshape(3, 3)  # 将扁平化的参数恢复为3x3矩阵
        # predicted=np.dot(ccm,input)  # 应用颜色校正
        predicted=np.dot(ccm,input)  # 应用颜色校正

        labPredicted= rgb_to_lab(predicted.T)  # 转换为Lab颜色空间
        labOutput = rgb_to_lab(output.T)  # 转换为Lab颜色空间
        labPredicted=labPredicted.T
        labOutput=labOutput.T
        # sumTmp=np.sum((labPredicted - labOutput)**2,axis=0)
 
        # sumTmp=np.sum((predicted - output)**2,axis=0)
        # error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        return((labPredicted - labOutput)**2).flatten()
        # return error

    def infer_image(self):
        ccm_six= self.ccm[:, :2].flatten()
        # print( '初始猜测值 ccm_six:',npToString(ccm_six) )
        x = ccm_six  
        

       
            # constraints.append( {
        #     'type': 'ineq', 
        #     'fun': lambda x: Cin @ x,
        # } )
        '''=====================最小二乘============================='''
        # from scipy.optimize import least_squares

        # 假设你的 self.loss 函数返回残差（residuals）而不是标量损失值
        result = least_squares(
            self.loss,  # 这个函数现在应该返回残差向量而不是标量
            x,
            args=(self.input, self.output),
            # bounds=bounds,  # least_squares 支持 bounds
            method='trf',  # 或 'lm'（Levenberg-Marquardt，无约束时使用）
            max_nfev=100000,
            verbose=0
        )
        '''=====================能量项优化============================='''
        # bounds = [(-4, 4) for _ in range(9)]
        # result = minimize(
        #     self.loss,  # 包装loss函数
        #     x,  
        #     args=(self.input, self.output),
        #     # constraints=self.constraints,

        #     # bounds=bounds,
        #     method='L-BFGS-B',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
        #     options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-9,'disp': False}
        # )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 2)
        optimized_ccm_full=np.zeros((3,3))
        optimized_ccm_full[:,:2]= optimized_ccm
        optimized_ccm_full[:,2]= 1-optimized_ccm_full[:,0]-optimized_ccm_full[:,1]
        # print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm_full
class CCM_3x3_saturation:
    def __init__(self,input,output):
        self.input = input
        self.output = output
        # self.ccm = np.ones(3, 3)  # 初始化为单位矩阵
        self.ccm = np.eye(3)  # 初始化为单位矩阵
        self.saturation = 1.0
        # 仅用shape进行判断
        if input.shape != output.shape:
            raise ValueError("input和output的形状必须相同")
        if input.shape[1] != 3:
            raise ValueError("最后一个维度必须是RGB颜色")
        self.m, self.n = input.shape[:2]  # 获取输入图像的形状

        Cons1 = np.zeros((3, 9))
        for i in range(3):
            Cons1[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # Cin=np.zeros((1, 9))
        # Cin[0, 4]=-1
        # Cin[0, 8]=1

        self.constraints = []
        # 约束条件: CCM矩阵的每一行之和为1
        self.constraints.append( {
            'type': 'eq', 
            'fun': lambda x: Cons1 @ x[:-1] - np.ones(3),
        } )
    def loss(self, x, input, output):
        input=input.T
        output=output.T
        ccm_x=x[:-1]
        saturation=x[-1]
        gray_matrix=np.array([[0.2126,0.7152,0.0722],[0.2126,0.7152,0.0722],[0.2126,0.7152,0.0722]])
        saturation_matrix=np.eye(3)*saturation+(1-saturation)*gray_matrix
        
        # print(input.shape,output.shape)
        ccm = ccm_x.reshape(3, 3)  # 将扁平化的参数恢复为3x3矩阵
        # predicted=np.dot(ccm,input)  # 应用颜色校正
        predicted=np.dot(ccm,input)  # 应用颜色校正
        predicted=np.dot(saturation_matrix,predicted)
        #predicted.shape=(3,24)
        #output.shape=(3,24)

        error=rgbError_detalE2000(predicted.T,output.T)
        return error
        sumTmp=np.sum((predicted - output)**2,axis=0)
        # print('sumTmp',sumTmp)
        # sumTmp[1]*=3
        # sumTmp[23]*=3
        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        # return((labPredicted - labOutput)**2).flatten()
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值
        x = np.concatenate((x, [self.saturation]))

       
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
            constraints=self.constraints,

            # bounds=bounds,
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-12,'disp': False}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x[:-1].reshape(3, 3)
        optimized_saturation = result.x[-1]
        # print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
        
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

        Cons1 = np.zeros((3, 9))
        for i in range(3):
            Cons1[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # Cin=np.zeros((1, 9))
        # Cin[0, 4]=-1
        # Cin[0, 8]=1

        self.constraints = []
        # 约束条件: CCM矩阵的每一行之和为1
        self.constraints.append( {
            'type': 'eq', 
            'fun': lambda x: Cons1 @ x - np.ones(3),
        } )
    def loss(self, x, input, output):
        input=input.T
        output=output.T
        # print(input.shape,output.shape)
        ccm = x.reshape(3, 3)  # 将扁平化的参数恢复为3x3矩阵
        # predicted=np.dot(ccm,input)  # 应用颜色校正
        predicted=np.dot(ccm,input)  # 应用颜色校正
        #predicted.shape=(3,24)
        #output.shape=(3,24)
        predicted=Gamma(predicted)
        error=rgbError_detalE2000(predicted.T,output.T)
        error=np.mean(error)
        return error
        sumTmp=np.sum((predicted - output)**2,axis=0)
        # print('sumTmp',sumTmp)
        # sumTmp[1]*=3
        # sumTmp[23]*=3
        error = np.mean(sumTmp)  # MSE误差
        # print(f"error:{error}")
        # return((labPredicted - labOutput)**2).flatten()
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值


       
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
            constraints=self.constraints,

            # bounds=bounds,
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-12,'disp': False}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 3)
        # print("===ccm===\n",npToString(optimized_ccm))

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
        # sumTmp[20]*=3
        # sumTmp[1]*=3
      
      
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
            # constraints=constraints,
            # bounds=bounds,
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 100000,'rhobeg': 1.0, 'rhoend': 1e-12,'disp': True}
        )
        # 打印优化结果的详细信息
  

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(4, 3)
        print("===ccm===\n",npToString(optimized_ccm))

        return optimized_ccm
   
def reverseGamma(img):
    mask = img <= 0.04045
    linear = np.zeros_like(img)
    linear[mask] = img[mask] / 12.92
    linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return linear
def Gamma(img):
    GAMMA_EXP = 1.0 / 2.4  # 预计算常数
    mask = img <= 0.0031308
    img = np.where(mask, img * 12.92, 1.055 * (img ** GAMMA_EXP) - 0.055)
    return img
def evaluate_transform(individual,matrix1,matrix2):
    """评估3×3变换矩阵的适应度"""
    transform_matrix = np.array(individual).reshape(3, 3)
    predicted = np.dot(matrix1, transform_matrix.T)
    # 计算每一行的RMSE然后求和
    row_errors = np.sqrt(np.mean((predicted - matrix2)**2, axis=1))
    total_error = np.sum(row_errors)
    return total_error,
def timeit(func):
    start_time = time.time()
    call_count = 0
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / call_count if call_count > 0 else 0
        print(f"   [{call_count}] 总耗时: {elapsed:.2f}s, 平均耗时: {avg_time:.2f}s")
        
        return result
    
    return wrapper
# 3. 遗传算法实现
@timeit
def run_ga(matrix1, matrix2, n_pop=100, n_gen=500, cxpb=0.7, mutpb=0.2):
    # 创建类型（如果尚未创建）
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    # 3×3矩阵有9个元素，每个元素在[-5,5]范围内
    toolbox.register("attr_float", np.random.uniform, -4, 4)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=9)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=n_pop)

    # 关键修改：使用partial绑定matrix1和matrix2到评估函数
    toolbox.register("evaluate", partial(evaluate_transform, matrix1=matrix1, matrix2=matrix2))
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'min', 'avg']
    
    for gen in range(n_gen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = toolbox.map(toolbox.evaluate, pop)  # 这里会自动传递individual
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        pop = toolbox.select(pop, k=len(pop))

    best_matrix = np.array(hof[0]).reshape(3, 3)
    best_error = evaluate_transform(hof[0], matrix1, matrix2)[0]  # 注意这里也要传参数
    return best_matrix, best_error, logbook 
ColorCheckerRGB= np.array([
    [110.32,  75.67,  62.16],
    [194.341, 144.622,  127.28],
    [88.127, 121.004,  153.175],
    [90.554, 107.889,  61.814],
    [131.657, 130.747,  178.76],
    [97.749, 186.483,  164.892],
    [218.739, 121.619,  44.752],
    [65.658, 84.871,  164.328],
    [195.608, 80.417,  97.23],
    [92.127, 55.334,  104.895],
    [158.591, 187.796,  60.912],
    [215.039, 154.824,  37.636],
    [46.377, 64.473,  147.159],
    [57.653,150.051,69.11],
    [173.337,52.375,53.257],
    [233.164,199.629,0.065],
    [183.087,73.554,146.152],
    [0, 136.56,166.328],
    [240.077,240.746,247.666],
    [201.043,202.9,211.196],
    [155.285,158.521,164.586],
    [116.495,118.199,121.993],
    [84.003,85.364,87.504],
    [48.704,48.612,48.055],
])/255.0
CC_D50_2=np.array([
    [119,  81,  68],   # Dark skin
    [203, 147, 130],   # Light skin
    [ 85, 123, 157],   # Blue sky
    [ 99, 110,  65],   # Foliage
    [127, 130, 177],   # Blue flower
    [100, 193, 178],   # Bluish green
    [229, 124,  50],   # Orange
    [ 52,  94, 169],   # Purplish blue
    [204,  82,  97],   # Moderate red
    [ 94,  60, 105],   # Purple
    [168, 188,  64],   # Yellow green
    [237, 163,  47],   # Orange yellow
    [  0,  65, 147],   # Blue
    [ 74, 148,  73],   # Green
    [182,  56,  57],   # Red
    [241, 197,  18],   # Yellow
    [191,  81, 145],   # Magenta
    [  0, 136, 169],   # Cyan
    [240, 240, 236],   # White
    [202, 204, 203],   # Neutral 8
    [163, 164, 163],   # Neutral 6.5
    [124, 123, 122],   # Neutral 5
    [ 84,  85,  85],   # Neutral 3.5
    [ 48,  49,  51],   # Black
])/255.0
IDEAL_COLORCHECKER_RGB = np.array([
    [0.45098, 0.32157, 0.26667],    # 深肤色
    [0.76078, 0.58824, 0.50980],    # 浅肤色
    [0.38431, 0.47843, 0.61569],    # 蓝天空
    [0.34118, 0.42353, 0.26275],    # 绿叶
    [0.52157, 0.50196, 0.69412],    # 蓝花
    [0.40392, 0.74118, 0.66667],    # 蓝光叶
    [0.83922, 0.49412, 0.17255],    # 橙色
    [0.31373, 0.35686, 0.65098],    # 紫蓝色
    [0.75686, 0.35294, 0.38824],    # 中红色
    [0.36863, 0.23529, 0.42353],    # 紫色
    [0.61569, 0.73725, 0.25098],    # 黄绿色
    [0.87843, 0.63922, 0.18039],    # 橙色
    [0.21961, 0.23922, 0.58824],    # 蓝色
    [0.27451, 0.58039, 0.28627],    # 绿色
    [0.68627, 0.21176, 0.23529],    # 红色
    [0.90588, 0.78039, 0.12157],    # 黄色
    [0.73333, 0.33725, 0.58431],    # 洋红色
    [0.03137, 0.52157, 0.63137],    # 青色
    [0.95294, 0.95294, 0.95294],    # 白色
    [0.78431, 0.78431, 0.78431],    # 中性8
    [0.62745, 0.62745, 0.62745],    # 中性6.5
    [0.47843, 0.47843, 0.47843],    # 中性5
    [0.33333, 0.33333, 0.33333],    # 中性3.5
    [0.20392, 0.20392, 0.20392]     # 黑色
], dtype=np.float32)
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
A_ColorChecker_Vivo=np.array([
  [0.630, 0.370, 0.283], [0.922, 0.659, 0.575], [0.539, 0.561, 0.628], [0.513, 0.511, 0.315], [0.700, 0.587, 0.678], [0.542, 0.785, 0.738],
  [0.938, 0.561, 0.297], [0.392, 0.386, 0.612], [0.898, 0.384, 0.387], [0.517, 0.256, 0.361], [0.707, 0.748, 0.371], [0.960, 0.672, 0.352],
  [0.213, 0.240, 0.511], [0.346, 0.592, 0.363], [0.815, 0.261, 0.236], [0.941, 0.766, 0.364], [0.842, 0.351, 0.496], [0.114, 0.546, 0.643],
  [0.925, 0.849, 0.814], [0.847, 0.766, 0.729], [0.724, 0.645, 0.610], [0.596, 0.515, 0.476], [0.412, 0.332, 0.303], [0.234, 0.168, 0.151]
])
# IDEAL_LINEAR_RGB = reverseGamma(IDEAL_RGB) # 逆Gamma处理后的理想RGB值
IDEAL_LINEAR_RGB = reverseGamma(CC_D50_2) # 逆Gamma处理后的理想RGB值
def ccmApply_3x4(img,ccm):
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    corrected_flat = np.dot(rgb_flat, ccm[:3,:].T)+ccm[3,:]  # 等价于 (ccm @ rgb_flat.T).T
    corrected_image = corrected_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return corrected_image
def calColor(img,area):
    """计算色块的平均RGB值"""
    # 批量处理：一次性计算每个区域所有通道的均值
    avg_colors = np.array([
        np.mean(img[y1:y2, x1:x2], axis=(0, 1))  # axis=(0,1) 表示在高度和宽度维度上求均值
        for x1, y1, x2, y2 in area
    ])
    return avg_colors  
def softmax_max(x, beta=10.0):
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * x)))
def calColorError(img,area):
    """计算色块的平均RGB值"""
    # 批量处理：一次性计算每个区域所有通道的均值
    avg_colors = np.array([
        np.mean(img[y1:y2, x1:x2], axis=(0, 1))  # axis=(0,1) 表示在高度和宽度维度上求均值
        for x1, y1, x2, y2 in area
    ])
    error=rgbError_detalE2000(avg_colors,CC_D50_2)
    
    error=np.mean(error)
    return error
def calColorError_ColorChecker(img,area,):
    """计算色块的平均RGB值"""
    # 批量处理：一次性计算每个区域所有通道的均值
    avg_colors = np.array([
        np.mean(img[y1:y2, x1:x2], axis=(0, 1))  # axis=(0,1) 表示在高度和宽度维度上求均值
        for x1, y1, x2, y2 in area
    ])
    
    min_vals = np.min(avg_colors, axis=1)  
    max_vals = np.max(avg_colors, axis=1)  
    
    # 避免除以0：如果max为0，结果为0；否则计算 1 - min/max
    saturation = np.where(max_vals == 0, 0, 1 - min_vals / max_vals)
    
    errors=rgbError_detalE2000(avg_colors,CC_D50_2)
    return errors,saturation  
def save_yaml_config(image,area,awb,ccm,save_path,basename):
    errors,saturation=calColorError_ColorChecker(image,area)
    avgError=np.mean(errors)
    minError=np.min(errors)
    maxError=np.max(errors)
    
    # 按十位数区间统计errors的个数
    # 定义区间：[0, 10), [10, 20), [20, 30), [30, 40), ...
    max_bin = int(np.ceil(maxError / 10) * 10) + 10  # 向上取整到十位数，再加10
    bins = np.arange(0, max_bin + 1, 10)  # 生成0, 10, 20, 30...的区间边界
    hist_counts, bin_edges = np.histogram(errors, bins=bins)
    
    # 创建区间标签和统计字典
    error_stats = {}
    for i in range(len(hist_counts)):
        start = int(bin_edges[i])
        end = int(bin_edges[i+1]) if i+1 < len(bin_edges) else int(max_bin)
        # 区间显示格式：0-10 表示 [0, 10)，即包含0但不包含10
        label = f"{start}-{end}"
        count = int(hist_counts[i])
        if count > 0:  # 只记录有数据的区间
            error_stats[label] = count
    
    errors = errors.reshape(4,6)  
    saturation=saturation.reshape(4,6)
    bestParamx1e8=np.array(awb)*100000000
    bestCCMx1e8= ccm*100000000
    bestParamx1e8=bestParamx1e8.astype(np.int64)
    bestCCMx1e8=bestCCMx1e8.astype(np.int64)
    awbNormalized=awb/awb[1]
    ccmNormalized= ccm*awb[1]
    awbx1e8=awbNormalized*100000000
    ccmx1e8= ccmNormalized*100000000
    awbx1e8=awbx1e8.astype(np.int64)
    ccmx1e8=ccmx1e8.astype(np.int64)
    yamlConfig={
        'awb': {
            'R': float(f"{awb[3]:.4f}"),
            'Gr': float(f"{awb[1]:.4f}"),
            'Gb': float(f"{awb[2]:.4f}"),
            'B': float(f"{awb[0]:.4f}"),
        },
        'ccm': np.round(ccm, 4).tolist(),
        'awb_x': {
            'R': int(f"{bestParamx1e8[3]}"),
            'Gr':int(f"{bestParamx1e8[1]}"),
            'Gb':int(f"{bestParamx1e8[2]}"),
            'B': int(f"{bestParamx1e8[0]}"),
        },
        'ccm_x': bestCCMx1e8.tolist(),
        'awbNormalized': {
            'R': float(f"{awbNormalized[3]:.4f}"),
            'Gr': float(f"{awbNormalized[1]:.4f}"),
            'Gb': float(f"{awbNormalized[2]:.4f}"),
            'B': float(f"{awbNormalized[0]:.4f}"),
        },
        'ccmNormalized': np.round(ccmNormalized, 4).tolist(),
        'awb_x1e8': {
            'R': int(f"{awbx1e8[3]}"),
            'Gr':int(f"{awbx1e8[1]}"),
            'Gb':int(f"{awbx1e8[2]}"),
            'B': int(f"{awbx1e8[0]}"),
        },
        'ccm_x1e8': ccmx1e8.tolist(),
        'errors': np.round(errors, 2).tolist(),
        'avgError': float(f"{avgError:.2f}"),
        'minError': float(f"{minError:.2f}"),
        'maxError': float(f"{maxError:.2f}"),
        'errorStats': error_stats,  
        'saturation':np.round(saturation, 2).tolist(),
    }
    yamlSavePath=os.path.join(save_path, f"{basename}_avgError{avgError:.2f}")
    saveYaml(yamlConfig,yamlSavePath)

def autoFitAwb(image_folder,lscyamlFolder):
    fixGreen= True
    @timeit
    def loss (x,  imgLSC,area,fixGreen):
        if fixGreen:
            rGain, bGain = x
            gGain= 1.0
        else:
            rGain, gGain, bGain = x
        awbParam=[bGain,gGain,gGain,rGain]
        imgTmp=AWB(imgLSC,awbParam)
        imgTmp=Demosaic(imgTmp)
        # imgTmp=np.clip(imgTmp,0,1023)
        imgTmp = imgTmp.astype(np.float64) 
        np.clip(imgTmp, 0, 1023, out=imgTmp)  
        imgTmp /= 1023 # 归一化
        # imgTmp= AWB_RGB(imgTmp,awbParam)  # 假设红蓝通道增益为1.0
        # print(f"去马赛克后图像尺寸: {imgTmp.shape},数据类型: {imgTmp.dtype},最小值: {imgTmp.min()}, 最大值: {imgTmp.max()},均值_8bit:{imgTmp.mean()/1023*255}")  # (高度, 宽度)
        # imgTmp=imgTmp[...,::-1] # BGR转RGB

        color_means= calColor(imgTmp,area)
        # print(f"计算的色块平均RGB值: {color_means}...")
        # ccmCalib= CCM_3x4(color_means, IDEAL_LINEAR_RGB) 
        # ccm, best_error, logbook = run_ga(color_means, IDEAL_LINEAR_RGB)
        ccmCalib= CCM_3x3(color_means, CC_D50_2) 
        ccm= ccmCalib.infer_image()
        # imgTmp= ccmApply_3x4(imgTmp,ccm)
        imgTmp= ccmApply(imgTmp,ccm)
        np.clip(imgTmp, 0, 1, out=imgTmp)
        imgTmp= Gamma(imgTmp)       
        error=calColorError(imgTmp,area)
        print(f"rGain:{rGain:.2f}, gGain:{gGain:.2f}, bGain:{bGain:.2f}, error:{error:.2f}" ,end='')
        return  error
    
    full_paths, basenames = get_paths(image_folder,suffix=".raw")
    #G07S5
    area=[[735, 490, 855, 610], [990, 490, 1110, 610], [1245, 490, 1365, 610], [1500, 490, 1620, 610], [1755, 490, 1875, 610], [2010, 490, 2130, 610], [735, 745, 855, 865], [990, 745, 1110, 865], [1245, 745, 1365, 865], [1500, 745, 1620, 865], [1755, 745, 1875, 865], [2010, 745, 2130, 865], [735, 1000, 855, 1120], [990, 1000, 1110, 1120], [1245, 1000, 1365, 1120], [1500, 1000, 1620, 1120], [1755, 1000, 1875, 1120], [2010, 1000, 2130, 1120], [735, 1255, 855, 1375], [990, 1255, 1110, 1375], [1245, 1255, 1365, 1375], [1500, 1255, 1620, 1375], [1755, 1255, 1875, 1375], [2010, 1255, 2130, 1375]]
    #G12
    # area=[[312, 285, 382, 355], [497, 285, 567, 355], [682, 285, 752, 355], [867, 285, 937, 355], [1052, 285, 1122, 355], [1237, 285, 1307, 355], [312, 470, 382, 540], [497, 470, 567, 540], [682, 470, 752, 540], [867, 470, 937, 540], [1052, 470, 1122, 540], [1237, 470, 1307, 540], [312, 655, 382, 725], [497, 655, 567, 725], [682, 655, 752, 725], [867, 655, 937, 725], [1052, 655, 1122, 725], [1237, 655, 1307, 725], [312, 840, 382, 910], [497, 840, 567, 910], [682, 840, 752, 910], [867, 840, 937, 910], [1052, 840, 1122, 910], [1237, 840, 1307, 910]]
    #no d50 d75
    # area=[[709, 692, 929, 912], [1194, 692, 1414, 912], [1679, 692, 1899, 912], [2164, 692, 2384, 912], [2649, 692, 2869, 912], [3134, 692, 3354, 912], [709, 1177, 929, 1397], [1194, 1177, 1414, 1397], [1679, 1177, 1899, 1397], [2164, 1177, 2384, 1397], [2649, 1177, 2869, 1397], [3134, 1177, 3354, 1397], [709, 1662, 929, 1882], [1194, 1662, 1414, 1882], [1679, 1662, 1899, 1882], [2164, 1662, 2384, 1882], [2649, 1662, 2869, 1882], [3134, 1662, 3354, 1882], [709, 2147, 929, 2367], [1194, 2147, 1414, 2367], [1679, 2147, 1899, 2367], [2164, 2147, 2384, 2367], [2649, 2147, 2869, 2367], [3134, 2147, 3354, 2367]]
    #d50 
    # area=[[628, 537, 888, 797], [1133, 537, 1393, 797], [1638, 537, 1898, 797], [2143, 537, 2403, 797], [2648, 537, 2908, 797], [3153, 537, 3413, 797], [628, 1042, 888, 1302], [1133, 1042, 1393, 1302], [1638, 1042, 1898, 1302], [2143, 1042, 2403, 1302], [2648, 1042, 2908, 1302], [3153, 1042, 3413, 1302], [628, 1547, 888, 1807], [1133, 1547, 1393, 1807], [1638, 1547, 1898, 1807], [2143, 1547, 2403, 1807], [2648, 1547, 2908, 1807], [3153, 1547, 3413, 1807], [628, 2052, 888, 2312], [1133, 2052, 1393, 2312], [1638, 2052, 1898, 2312], [2143, 2052, 2403, 2312], [2648, 2052, 2908, 2312], [3153, 2052, 3413, 2312]]
    #d75

    for path,basename in zip(full_paths,basenames):
        keyCT= getCTstr(path)
       
        print(f"Processing image: {path},colorTemp:{keyCT}...")   


        yaml_files,_= get_paths(lscyamlFolder,suffix=".yaml")
        for yf in yaml_files:
            if keyCT in yf:
                yaml_file=yf
                break
        if yaml_file == '':
            print(f"未找到对应的yaml文件，跳过处理: {keyCT}")
            continue
        print(f"Using yaml file: {yaml_file}...")
        dataYaml = loadYaml(yaml_file)
        gainList=[]
        
        mesh_R = np.array(dataYaml['R'])
        mesh_Gr = np.array(dataYaml['Gr'])
        mesh_Gb = np.array(dataYaml['Gb'])
        mesh_B = np.array(dataYaml['B'])

        gainList.append(mesh_R)
        gainList.append(mesh_Gr)
        gainList.append(mesh_Gb)
        gainList.append(mesh_B)
   
        img = readRaw(path,h=1944,w=2592)  # 读取为numpy数组
        print(f"图像尺寸: {img.shape},数据类型: {img.dtype},最小值: {img.min()}, 最大值: {img.max()},均值_10bit:{img.mean()},均值_8bit:{img.mean()/1023*255}")  # (高度, 宽度)
        img= BLC(img,blcParam=64)
        # img=bm3d_denoise(img)
        lsc_strength=1
        # imgLSC=LSC(img,gainList,strength=[lsc_strength,lsc_strength,lsc_strength,lsc_strength])
        imgLSC=img
        awbParam=None
        saveFolderName='ispResults'
        savePath=os.path.join(image_folder,saveFolderName)
        if fixGreen:
            x=[1,1] #rGain, bGain
            bounds=[(0.3, 5), (0.3, 5)]
        else:    
            x=[1,1,1] #rGain, gGain, bGain
            bounds=[(0.3, 5), (0.3,5), (0.3, 5)]
        cur_method='Powell'
        awbResult=minimize(
            loss,  # 包装loss函数
            x,  
            args=(imgLSC,area,fixGreen),
            # constraints=constraints,
            bounds=bounds,
            method=cur_method,#trust-constr SLSQP  L-BFGS-B TNC COBYLA_ Nelder-Mead Powell
            options={'maxiter': 10000,'disp': True}
        )
        if fixGreen:
            r,b=awbResult.x
            g=1.0
        else:
            r,g,b=awbResult.x
        awbParam=[b,g,g,r]
        awbParam=np.array(awbParam)
        print(f"优化结果: {awbResult}")
        # imgTmp= AWB(imgLSC,bestParam)  # 假设红蓝通道增益为1.0
        imgTmp= AWB(imgLSC,awbParam)  # 假设红蓝通道增益为1.0
        imgTmp=Demosaic(imgTmp)
        imgTmp = imgTmp.astype(np.float64)
        np.clip(imgTmp, 0, 1023, out=imgTmp)
        imgTmp /=1023 # 归一化
        # imgTmp=AWB_RGB(imgTmp,bestParam)
        color_means= calColor(imgTmp,area)
        # ccmCalib= CCM_3x4(color_means, IDEAL_LINEAR_RGB) 
        ccmCalib= CCM_3x3(color_means, CC_D50_2) 
        ccm= ccmCalib.infer_image()
        # imgTmp= ccmApply_3x4(imgTmp,ccm)
        imgTmp= ccmApply(imgTmp,ccm)

        np.clip(imgTmp, 0, 1, out=imgTmp)
        imgTmp= Gamma(imgTmp)
        os.makedirs(savePath, exist_ok=True)
        save_yaml_config(imgTmp,area,awbParam,ccm,savePath,basename)
        

        imgTmp=imgTmp[...,::-1] # RGB转BGR
        imgTmp = np.clip(imgTmp * 255, 0, 255)
        imgTmp = imgTmp.astype(np.uint8)

        imgSavePath=os.path.join(savePath, f"{basename}_{cur_method}_{'fixGreen' if fixGreen else 'nofixGreen'}.jpg")
        cv2.imwrite(imgSavePath, imgTmp)

        
       
def autoFitLsc(image_folder):

    full_paths, basenames = get_paths(image_folder,suffix=".raw")
    Rrange= np.arange(0.8, 1.8, 0.15)  
    Brange= np.arange(1.2, 1.8, 0.15)  
    yamlFolder= r'C:\WorkSpace\serialPortVisualization\data\0901lscConfig2'
    area=[[710, 471, 840, 601], [965, 471, 1095, 601], [1220, 471, 1350, 601], [1475, 471, 1605, 601], [1730, 471, 1860, 601], [1985, 471, 2115, 601], [710, 726, 840, 856], [965, 726, 1095, 856], [1220, 726, 1350, 856], [1475, 726, 1605, 856], [1730, 726, 1860, 856], [1985, 726, 2115, 856], [710, 981, 840, 1111], [965, 981, 1095, 1111], [1220, 981, 1350, 1111], [1475, 981, 1605, 1111], [1730, 981, 1860, 1111], [1985, 981, 2115, 1111], [710, 1236, 840, 1366], [965, 1236, 1095, 1366], [1220, 1236, 1350, 1366], [1475, 1236, 1605, 1366], [1730, 1236, 1860, 1366], [1985, 1236, 2115, 1366]]

    for path,basename in zip(full_paths,basenames):
        keyCT= getCTstr(path)
       
        print(f"Processing image: {path},colorTemp:{keyCT}...")   
        # yaml_file = fr'C:\serialPortVisualization\data\0815_1_Config\isp_sensor_raw{keyCT}.yaml'
        # yaml_file = ''
        # yaml_files,_= get_paths(yamlFolder,suffix=".yaml")
        # for yf in yaml_files:
        #     if keyCT in yf:
        #         yaml_file=yf
        #         break
        # if yaml_file == '':
        #     print(f"未找到对应的yaml文件，跳过处理: {keyCT}")
        #     continue
        # print(f"Using yaml file: {yaml_file}...")
        # dataYaml = loadYaml(yaml_file)
        # gainList=[]
        
        # mesh_R = np.array(dataYaml['R'])
        # mesh_Gr = np.array(dataYaml['Gr'])
        # mesh_Gb = np.array(dataYaml['Gb'])
        # mesh_B = np.array(dataYaml['B'])


        # gainList.append(mesh_R)
        # gainList.append(mesh_Gr)
        # gainList.append(mesh_Gb)
        # gainList.append(mesh_B)
   
        # img = read_pgm_with_opencv(path)
        img = readRaw(path,h=1944,w=2592) 
        print(f"图像尺寸: {img.shape},数据类型: {img.dtype},最小值: {img.min()}, 最大值: {img.max()},均值_10bit:{img.mean()},均值_8bit:{img.mean()/1023*255}")  # (高度, 宽度)
        imgLSC= BLC(img,blcParam=64)
        # imgLSC=LSC(img,gainList,strength=[0.5,0.5,0.5,0.5])
        imgLSC=bm3d_denoise(imgLSC)
        minError=float('inf')
        bestParam=None
        bestCCM=None
        saveFolderName='ispResults'
        savePath=os.path.join(image_folder,saveFolderName)

        for rGain in Rrange:
            for bGain in Brange:
                    awbParam=[bGain,1,1,rGain]
                    print(f"tring awb Param:R_{rGain:.2f}, B_{bGain:.2f}",end=' ')
                    # awbParam=AWBList[keyCT]
                    # print(imgLSC[0:4,0:4])

                    imgTmp= AWB(imgLSC,awbParam)  # 假设红蓝通道增益为1.0
                    imgTmp=np.clip(imgTmp,0,1023)
                    imgTmp=Demosaic(imgTmp)
                    imgTmp = imgTmp/1023 # 归一化
                    # print(f"去马赛克后图像尺寸: {imgTmp.shape},数据类型: {imgTmp.dtype},最小值: {imgTmp.min()}, 最大值: {imgTmp.max()},均值_8bit:{imgTmp.mean()/1023*255}")  # (高度, 宽度)
                    # imgTmp=imgTmp[...,::-1] # BGR转RGB

                    color_means= calColor(imgTmp,area)
                    # print(f"计算的色块平均RGB值: {color_means}...")
                    # ccmCalib= CCM_3x4(color_means, IDEAL_LINEAR_RGB) 
                    # ccmCalib= CCM_3x3(color_means, IDEAL_LINEAR_RGB) 
                    # ccm= ccmCalib.infer_image()
                    ccm, best_error, logbook = run_ga(color_means, IDEAL_LINEAR_RGB)

                    # imgTmp= ccmApply_3x4(imgTmp,ccm)
                    imgTmp= ccmApply(imgTmp,ccm)
                    
                    imgTmp= Gamma(imgTmp)
                    error=calColorError(imgTmp,area)
                    if error < minError:
                        minError=error
                        bestParam=awbParam
                        bestCCM=ccm
                    
                    print(f"当前误差: {error:.2f}, 最小误差: {minError:.2f}")

                    imgTmp=imgTmp[...,::-1] # RGB转BGR
                    imgTmp = np.clip(imgTmp * 255, 0, 255)
                    imgTmp = imgTmp.astype(np.uint8)
                    os.makedirs(savePath, exist_ok=True)
                    imgSavePath=os.path.join(savePath, f"{keyCT}_R{rGain:.2f}_B{bGain:.2f}_error{error:.2f}.jpg")
                    cv2.imwrite(imgSavePath, imgTmp)
                    # writePgm(imgLSCTmp, basename)  
        yamlConfig={
            'awbParam': {
                'R': float(f"{bestParam[3]:.4f}"),
                'Gr': float(f"{bestParam[1]:.4f}"),
                'Gb': float(f"{bestParam[2]:.4f}"),
                'B': float(f"{bestParam[0]:.4f}"),
            },
            'CCM': bestCCM.tolist(),
            'Error': float(f"{minError:.4f}"),
        }
        yamlSavePath=os.path.join(savePath, f"{keyCT}_ispConfig")
        saveYaml(yamlConfig,yamlSavePath)
        print(f"最佳awb参数: {bestParam}, 最小色彩误差: {minError},最佳CCM:\n{npToString(bestCCM)}")    
def main():
    # autoFitLsc(folderPath)
    # error=rgbError_detalE2000(IDEAL_RGB,ColorCheckerRGB)
    # print(f"初始色彩误差(2000): {error}")

    folderPath= r'C:\WorkSpace\serialPortVisualization\data\g07s5ColorChecker'
    lscyamlFolder= r'C:\WorkSpace\serialPortVisualization\data\1011LSC'
    autoFitAwb(folderPath,lscyamlFolder)

main()