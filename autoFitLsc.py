import sys
import os
from scipy.optimize import minimize, curve_fit
import cv2
import numpy as np
from tools import *
from skimage.color import rgb2lab

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
    balanced[1::2, 1::2] = np.clip(image[1::2, 1::2] * bGain, 0, 1023).astype(np.float64)
    
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
    rgb = cv2.cvtColor(bayer_pgm, cv2.COLOR_BAYER_RGGB2RGB)
    return rgb
def BLC(img,blcParam=16):
    img= img - blcParam 
    img = np.clip(img, 0, 1023)  # 确保像素值在有效范围内
    return img
def ccmApply(img,ccm):
    
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    corrected_flat = np.dot(rgb_flat, ccm.T)  # 等价于 (ccm @ rgb_flat.T).T
    corrected_image = corrected_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return corrected_image
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
        from scipy.optimize import least_squares

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
    
    # img_encoded = cv2.imread(folderPath)
    mask = img <= 0.0031308
    srgb = np.zeros_like(img)
    srgb[mask] = img[mask] * 12.92
    srgb[~mask] = 1.055 * (img[~mask] ** (1/2.4)) - 0.055
    return srgb
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
    [0.73333, 0.33725, 0.58431],    # magenta
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
IDEAL_LINEAR_RGB = reverseGamma(IDEAL_RGB) # 逆Gamma处理后的理想RGB值
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
    avg_colors = []
    for (x1, y1, x2, y2) in area:
        patch = img[y1:y2, x1:x2]
        # mean_color = cv2.mean(patch)[:3]  # 计算BGR的均值，忽略Alpha通道
        Rmean= np.mean(patch[:,:,0])
        Gmean= np.mean(patch[:,:,1])
        Bmean= np.mean(patch[:,:,2])
        mean_color = [Rmean, Gmean, Bmean]
        avg_colors.append(mean_color)
    return np.array(avg_colors)  # 返回一个包含所有色块平均颜色的数组
def autoFitLsc(image_folder):

    full_paths, basenames = get_paths(image_folder,suffix=".pgm")
    Rrange= np.arange(1.55, 1.9, 0.03)  # 红色通道增益范围
    Brange= np.arange(0.75, 1.08, 0.03)  
    Grange= np.arange(0.85, 1.05, 0.03)  
    yamlFolder= r'C:\WorkSpace\serialPortVisualization\data\0826_LSC_low'
    area=[[400, 481, 550, 631], [710, 481, 860, 631], [1020, 481, 1170, 631], [1330, 481, 1480, 631], [1640, 481, 1790, 631], [1950, 481, 2100, 631], [400, 791, 550, 941], [710, 791, 860, 941], [1020, 791, 1170, 941], [1330, 791, 1480, 941], [1640, 791, 1790, 941], [1950, 791, 2100, 941], [400, 1101, 550, 1251], [710, 1101, 860, 1251], [1020, 1101, 1170, 1251], [1330, 1101, 1480, 1251], [1640, 1101, 1790, 1251], [1950, 1101, 2100, 1251], [400, 1411, 550, 1561], [710, 1411, 860, 1561], [1020, 1411, 1170, 1561], [1330, 1411, 1480, 1561], [1640, 1411, 1790, 1561], [1950, 1411, 2100, 1561]]
    for path,basename in zip(full_paths,basenames):
        keyCT= getCTstr(path)
       
        print(f"Processing image: {path},colorTemp:{keyCT}...")   
        if keyCT !='U30':
            continue
        # yaml_file = fr'C:\serialPortVisualization\data\0815_1_Config\isp_sensor_raw{keyCT}.yaml'
        yaml_file = ''
        yaml_files,_= get_paths(yamlFolder,suffix=".yaml")
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
   
        img = read_pgm_with_opencv(path)
        print(f"图像尺寸: {img.shape},数据类型: {img.dtype},最小值: {img.min()}, 最大值: {img.max()},均值_10bit:{img.mean()},均值_8bit:{img.mean()/1023*255}")  # (高度, 宽度)
        img= BLC(img,blcParam=16)
        imgLSC=LSC(img,gainList,strength=[0.5,0.5,0.5,0.5])

        for rGain in Rrange:
            for bGain in Brange:
                    awbParam=[rGain,1,1,bGain]
                    print(f"尝试awb参数: {awbParam}...")
                    # awbParam=AWBList[keyCT]
                    imgTmp= AWB(imgLSC,awbParam)  # 假设红蓝通道增益为1.0
                    imgTmp=np.clip(imgTmp,0,1023)
                    imgTmp=Demosaic(imgTmp)
                    imgTmp = imgTmp/1023 # 归一化
                    print(f"去马赛克后图像尺寸: {imgTmp.shape},数据类型: {imgTmp.dtype},最小值: {imgTmp.min()}, 最大值: {imgTmp.max()},均值_8bit:{imgTmp.mean()/1023*255}")  # (高度, 宽度)
                    imgTmp=imgTmp[...,::-1] # BGR转RGB

                    color_means= calColor(imgTmp,area)
                    print(f"计算的色块平均RGB值: {color_means}...")
                    # ccmCalib= CCM_3x4(color_means, IDEAL_LINEAR_RGB) 
                    ccmCalib= CCM_3x3(color_means, IDEAL_LINEAR_RGB) 
                    ccm= ccmCalib.infer_image()
                    # imgTmp= ccmApply_3x4(imgTmp,ccm)
                    imgTmp= ccmApply(imgTmp,ccm)
                    imgTmp= Gamma(imgTmp)
                    imgTmp=imgTmp[...,::-1] # BGR转RGB

                    imgTmp = np.clip(imgTmp * 255, 0, 255)
                    imgTmp = imgTmp.astype(np.uint8)
                    savePath=os.path.join(image_folder,'demosaicResults')
                    os.makedirs(savePath, exist_ok=True)
                    imgSavePath=os.path.join(savePath, f"{basename}_R{rGain:.2f}_B{bGain:.2f}.jpg")
                    cv2.imwrite(imgSavePath, imgTmp)
                    # writePgm(imgLSCTmp, basename)  
            


def main():
    if len(sys.argv) != 2:
        print("Usage: python rawLsc.py <folder_path>")
        sys.exit(1)
    
    folderPath = sys.argv[1]
    autoFitLsc(folderPath)

main()