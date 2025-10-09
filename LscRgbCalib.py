from tools import *
import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.optimize import curve_fit
import scienceplots  # 导入科学绘图样式
import matplotlib as mpl
plt.style.use(['science', 'no-latex'])  # 不使用LaTeX渲染

plt.rcParams['font.size'] = 14          # 全局字体大小
mpl.rcParams['font.family'] = 'Microsoft YaHei'
def vignetting_theta(theta, n=4.0, k=0.0):
    """
    以角度 theta（弧度）为自变量计算亮度衰减:
        I(theta) = I0 * cos(theta)**n * exp(-k * theta**2)

    参数:
      theta : float or ndarray, 单位为弧度
      I0    : 中心亮度 (default 1.0)
      n     : cos 的幂指数 (default 4.0)
      k     : 指数衰减系数 (default 0.0)

    返回:
      I : 与 theta 同形状的亮度值
    """
    theta = np.asarray(theta)
    cos_t = np.cos(theta)
    # 为了数值稳定性，cos_t 可能小于0（极大视场时），这里允许负值的幂运算：
    # 如果你期望物理上小于0的 cos 不出现，应该先裁剪: cos_t = np.clip(cos_t, 0, None)
    return np.power(cos_t, n) * np.exp(-k * theta * theta)
def func_exp(x, a, k, c):
    return a * np.exp(-k * x )+ c
def multi_exp(x, a1, k1, a2, k2, c):
    return a1 * np.exp(-k1 * x) + a2 * np.exp(-k2 * x) + c
def mix_exp_linear(x, a1, k1,a2, k2, b1, b2):
    return a1 * np.exp(-k1 * x)  + a2 * np.exp(-k2 * x)+ b1 * x + b2

def mix_exp_linear_plus(x, a1, k1,a2, k2, b2):
    return a1 * np.exp(-k1 * x)  + a2 * np.exp(-k2 * x)+ b2
# 以2为底的指数衰减函数: a * 2^(-k*x) + c 
def func_exp2(x, a, k, c):
    return a * 2**(-k * x) + c

# 二次多项式函数: a*x² + b*x + c
def func_quad(x, a, b, c):
    return a * x**2 + b * x + c

# 线性函数: a*x + b
def func_linear(x, a, b):
    return a * x + b

# 平方根函数: a*sqrt(x) + b
def func_sqrt(x, a, b):
    return a * np.sqrt(x) + b

def func_1_x(x, a, b, c):
    return a / (x + b) + c

def func_1_r1_r2(x, a, b ):
    return 1+ a * x + b * x**2
def func_1_r2_r4(x, a,b):
    return 1+ a * x**2 +b*x**4
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

    mesh_R = max_R *maxFactor/ mesh_R
    mesh_Gb=  max_Gb *maxFactor/ mesh_Gb
    mesh_Gr = max_Gr *maxFactor/ mesh_Gr
    mesh_B = max_B *maxFactor/ mesh_B

    return np.array([mesh_R, mesh_Gr, mesh_Gb, mesh_B])
def visualize_2d_array_multi(data_list, name,folderPath=''):
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

    nameList=[f"{name}_{t}" for t in typeList]
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
    savePath=os.path.join(folderPath,f'{name}_2x2.png')
    # 保存图像
    plt.savefig(
        savePath,
        dpi=150,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()
def saveYaml_RGGB(result, basename):
    """
    保存校准结果到 YAML 文件
    
    参数:
        result: 包含校准结果的列表或数组
        basename: 输出文件的基本名称
    """
    
    # 转换所有 NumPy 数组为列表
    converted_result = [convert_numpy(arr) for arr in result]
    
    # 准备元数据
    calibration_data = { }
    
    typeList= ['R','Gr','Gb','B']
    for t, arr in zip(typeList, converted_result):
            calibration_data[t]=arr
    
    # 保存为 YAML 文件
    with open(f'{basename}.yaml', 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, sort_keys=False,width=float("inf"))
    
    print(f"Calibration results saved to {basename}.yaml")
def generate_lut(m: int, n: int,  ):
    """
    Generate a lookup table for brightness adjustment.
    
    Args:adjust_brightness_by_blocks_in_yuv
        m: Number of rows in block grid
        n: Number of columns in block grid
        c_strength: Corner strength parameter
    
    Returns:
        2D list of gain values ((m+1) x (n+1))
    """
    m += 1
    n += 1
    lut = [[0.0 for _ in range(n)] for _ in range(m)]
    
    asymmetry = 1.0

    R2 = m * n / 4 * (1 + asymmetry * asymmetry)
    
    for y in range(n):
        for x in range(m):
            dy = y - n / 2 + 0.5
            dx = (x - m / 2 + 0.5) * asymmetry
            r2 = (dx * dx + dy * dy) / R2
            lut[x][y] = r2 # reproduces the cos^4 rule
    
    return np.array(lut)
def curveFit(data,func):
    # 获取函数的参数信息
    sig = inspect.signature(func)
    parameters = list(sig.parameters.keys())
    
    # 提取需要拟合的参数名（去掉第一个变量 'x'）
    fit_param_names = parameters[1:]  # 假设第一个参数是 x，其余是待拟合参数
    
    num_params = len(fit_param_names)
    # 默认初始猜测 p0（可以根据需要调整）
    p0 = [1.0] * num_params  # 例如：[1.0, 1.0, ...] 长度与参数数量一致

    sorted_indices = np.argsort(data[:, 0])  # 获取排序索引
    data = data[sorted_indices]
    x = data[:,0]
    y = data[:,1]


    params, covariance = curve_fit(func, x, y, p0=p0,maxfev=100000)  # 设置边界条件，避免参数为负值
    param_errors = np.sqrt(np.diag(covariance))

    # print(f"param_errors: {param_errors}")
    param_values = {}
    for name, value in zip(fit_param_names, params):
        param_values[name] = value
    

    # 计算拟合值
    y_fit = func(x,*params)

    # plt.figure(figsize=(14,7)) 
    # plt.scatter(x, y, label="原始数据")
    # plt.plot(x, y_fit, 'r-', label="拟合曲线")
    # plt.legend()
    # plt.show()
    # param_str = ", ".join([f"{name}={value}" for name, value in param_values.items()])
    # paramError_str = ", ".join([f"{name}误差={error}" for name, error in zip(fit_param_names, param_errors)])
    # print(f"拟合参数: {param_str}")
    # print(f"拟合误差: {paramError_str}")
    return params
def generater2(x,y,m=24,n=32):
    asymmetry = 1.0
    
    R2 = m * n / 4 * (1 + asymmetry * asymmetry)
    

    dy = y - n / 2 + 0.5
    dx = (x - m / 2 + 0.5) * asymmetry
    r2 = (dx * dx + dy * dy) / R2
    return r2
def matrix_fit(matrix):
    result=[]
    for ch in matrix:
            dataLut= ch
            m = len(dataLut) 
            n = len(dataLut[0]) 
            xIdx = generate_lut(m-1, n-1)
            dataLut = np.array(dataLut)
            # xIdx = xIdx[:, :xIdx.shape[1]//2]  
            # dataLut = dataLut[:, :dataLut.shape[1]//2] 
            dataLut = dataLut.flatten()
            xIdx = xIdx.flatten()
            dataLut = np.column_stack((xIdx, dataLut))
            
            params = curveFit(dataLut, func=func_1_r1_r2)
            
            lut = np.zeros((m , n ))
            for j in range(n ):
                for i in range(m ):
                    lut[i][j] = func_1_r1_r2(generater2(i, j, m, n,), *params)
            result.append( convert_numpy(lut))
    return np.array(result)

def lenShadingCalibrationForRaw(image_folder,h,w):
    full_paths, basenames = get_paths(image_folder,suffix=".raw")
    for path,basename in zip(full_paths,basenames):
        # rawImage = read_pgm_with_opencv(path)
        rawImage=readRaw(path,h,w)
        blcParam=64
        rawImage= rawImage - blcParam
        if rawImage is  None:
            raise ValueError("raw data is None")
        avgLRaw= rawImage.mean()
        avgLRaw=avgLRaw/1023*255
        print(f"图像尺寸: {rawImage.shape},数据类型: {rawImage.dtype},最小值: {rawImage.min()}, 最大值: {rawImage.max()},均值_10bit:{rawImage.mean()},均值_8bit:{avgLRaw}")  # (高度, 宽度)
        
        resultList=[]
        for s in range(4,11,2):
            resultTmp = lsc_calib(rawImage, mesh_h_nums=24, mesh_w_nums=32, sample_size=s,maxFactor=1.0)
            resultList.append(resultTmp)
        stacked = np.stack(resultList)  

        result = np.mean(stacked, axis=0)
        #result.shape=(4,25,33)
        result=matrix_fit(result)

        
        visualize_2d_array_multi(result, basename)
        saveYaml_RGGB(result, basename)


if __name__ == "__main__":
    folderPath=r'C:\WorkSpace\serialPortVisualization\data\0901LSC'
    lenShadingCalibrationForRaw(folderPath,h=1944,w=2592)