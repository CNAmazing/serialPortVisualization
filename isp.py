from tools import *
import cv2
def Gamma(img):
    GAMMA_EXP = 1.0 / 2.4  # 预计算常数
    mask = img <= 0.0031308
    img = np.where(mask, img * 12.92, 1.055 * (img ** GAMMA_EXP) - 0.055)
    return img
def ccmApply(img,ccm):
    
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    np.dot(rgb_flat, ccm.T,out=rgb_flat)  # 等价于 (ccm @ rgb_flat.T).T
    rgb_flat = rgb_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return rgb_flat
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
def LSC_rgb(image, gainList, strength=[1.0, 1.0, 1.0]):
    """
    对RGB图像进行Lens Shading Correction (LSC)
    参数:
        image: np.ndarray, shape=(H, W, 3)，RGB图像
        gainList: [mesh_R, mesh_Gr, mesh_Gb, mesh_B]，每个为2D增益矩阵
        strength: 每个通道的校正强度，默认全为1
    """
    mesh_R, mesh_Gr, mesh_Gb, mesh_B = gainList
    tmp = mesh_B
    mesh_B = mesh_R
    mesh_R =tmp
    # 合并G通道（通常Gr和Gb差别不大）
    mesh_G = (mesh_Gr + mesh_Gb) / 2.0

    # 应用strength调节
    mesh_R = 1 + (mesh_R - 1) * strength[0]
    mesh_G = 1 + (mesh_G - 1) * strength[1]
    mesh_B = 1 + (mesh_B - 1) * strength[2]

    rows, cols = image.shape[:2]
    m = len(mesh_R) - 1
    n = len(mesh_R[0]) - 1
    if m == 0 or n == 0:
        return image

    # 计算块边界
    block_heights = np.linspace(0, rows, m + 1, dtype=int)
    block_widths = np.linspace(0, cols, n + 1, dtype=int)

    # 网格坐标
    y_coords, x_coords = np.indices((rows, cols))
    i_indices = np.searchsorted(block_heights, y_coords, side='right') - 1
    j_indices = np.searchsorted(block_widths, x_coords, side='right') - 1
    i_indices = np.clip(i_indices, 0, m - 1)
    j_indices = np.clip(j_indices, 0, n - 1)

    # 归一化坐标
    y_norm = (y_coords - block_heights[i_indices]) / (
        (block_heights[i_indices + 1] - block_heights[i_indices]).astype(float)
    )
    x_norm = (x_coords - block_widths[j_indices]) / (
        (block_widths[j_indices + 1] - block_widths[j_indices]).astype(float)
    )

    # 定义辅助函数：对一个mesh执行双线性插值
    def bilinear_interp(mesh):
        q11 = mesh[i_indices, j_indices]
        q21 = mesh[i_indices, j_indices + 1]
        q12 = mesh[i_indices + 1, j_indices]
        q22 = mesh[i_indices + 1, j_indices + 1]
        return ((1 - y_norm) * (1 - x_norm) * q11 +
                (1 - y_norm) * x_norm * q21 +
                y_norm * (1 - x_norm) * q12 +
                y_norm * x_norm * q22)

    # 分别对R、G、B插值
    gain_R = bilinear_interp(mesh_R)
    gain_G = bilinear_interp(mesh_G)
    gain_B = bilinear_interp(mesh_B)

    # 构建gain_map并应用
    gain_map = np.stack([gain_R, gain_G, gain_B], axis=-1)
    result = np.clip(image * gain_map, 0, 1023).astype(np.float64)
    return result
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
    balanced[1::2, 1::2] = np.clip(image[1::2, 1::2] * rGain, 0, 1023).astype(np.float64)
    balanced[::2, 1::2] = np.clip(image[::2, 1::2] * grGain, 0, 1023).astype(np.float64)
    balanced[1::2, ::2] = np.clip(image[1::2, ::2] * gbGain, 0, 1023).astype(np.float64)
    # 矫正蓝色通道 (B位于奇数行奇数列)
    balanced[::2, ::2] = np.clip(image[::2, ::2] * bGain, 0, 1023).astype(np.float64)
    
    return balanced

def Demosaic(bayer_pgm):
    # 常见选项：COLOR_BAYER_BG2RGB, COLOR_BAYER_RG2RGB, COLOR_BAYER_GB2RGB 等
    bayer_pgm= bayer_pgm.astype(np.uint16)  # 确保数据类型为 uint16
    rgb = cv2.cvtColor(bayer_pgm, cv2.COLOR_BAYER_BGGR2RGB)
    return rgb
def BLC(img,blcParam=64):
    img= img - blcParam 
    img = np.clip(img, 0, 1023)  # 确保像素值在有效范围内
    return img
def wbestParamFromYaml(keyCT,yamlPath):
    yaml_files,_= get_paths(yamlPath,suffix=".yaml")
    for yf in yaml_files:
        if keyCT in yf:
            yaml_file=yf
            break
  
    print(f"Using lsc yaml file: {yaml_file}...")
    dataYaml = loadYaml(yaml_file)
    gainParam=dataYaml['awbParam']
    wb_R = np.array(gainParam['R'])
    wb_Gr = np.array(gainParam['Gr'])
    wb_Gb = np.array(gainParam['Gr'])
    wb_B = np.array(gainParam['B'])

    ccm=dataYaml['CCM']
    # print(gainParam)
    return np.array([wb_R,wb_Gr,wb_Gb,wb_B]),np.array(ccm)
def ispPipe(image_folder,lsc_yaml_folder,awb_yaml_folder):
    
    full_paths, basenames = get_paths(image_folder,suffix=".raw")
    saveFolderName='ispResults'
    savePath=os.path.join(image_folder,saveFolderName)
    x=[1,1,1] #rGain, gGain, bGain
    for path,basename in zip(full_paths,basenames):
        keyCT= getCTstr(path)
       
        print(f"Processing image: {path},colorTemp:{keyCT}...")   
        yaml_files,_= get_paths(lsc_yaml_folder,suffix=".yaml")
        for yf in yaml_files:
            if keyCT in yf:
                yaml_file=yf
                break
        if yaml_file == '':
            print(f"未找到对应的yaml文件，跳过处理: {keyCT}")
            continue
        print(f"Using lsc yaml file: {yaml_file}...")
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
        bestParam,ccm=wbestParamFromYaml(keyCT,awb_yaml_folder)
        
        img = readRaw(path,h=1944,w=2592)  # 读取为numpy数组
        print(f"图像尺寸: {img.shape},数据类型: {img.dtype},最小值: {img.min()}, 最大值: {img.max()},均值_10bit:{img.mean()},均值_8bit:{img.mean()/1023*255}")  # (高度, 宽度)
        img= BLC(img,blcParam=64)
        # img=LSC(img,gainList,strength=[1,1,1,1])
       
        img= AWB(img,bestParam)  # 假设红蓝通道增益为1.0
        img=Demosaic(img)
        img = img.astype(np.float64)
        img=LSC_rgb(img,gainList,strength=[1,1,1])
        np.clip(img, 0, 1023, out=img)
        # imgTmp=np.clip(imgTmp,0,1023)
        img /=1023 # 归一化
        # imgTmp=AWB_RGB(imgTmp,bestParam)
        # imgTmp= ccmApply_3x4(imgTmp,ccm)
        img= ccmApply(img,ccm)
        img= Gamma(img)

        img=img[...,::-1] # RGB转BGR
        img = np.clip(img * 255, 0, 255)
        img = img.astype(np.uint8)
        os.makedirs(savePath, exist_ok=True)
        imgSavePath=os.path.join(savePath, f"{basename}_.jpg")
        cv2.imwrite(imgSavePath, img)
       
if __name__ == "__main__":
    image_folder=r'C:\WorkSpace\serialPortVisualization\data\0901LSC'
    lsc_yaml_folder=r'C:\WorkSpace\serialPortVisualization\data\1009LSC'
    awb_yaml_folder=r'C:\WorkSpace\serialPortVisualization\data\g07s5ColorChecker\ispResults10'
    ispPipe(image_folder,lsc_yaml_folder,awb_yaml_folder)