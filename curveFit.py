import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import inspect
from tools import *

mpl.rcParams['font.family'] = 'Microsoft YaHei'

# 指数衰减函数: a * e^(-k*x) + c
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

def func_1_r2_r4(x, a, b,c ):
    return 1+ a * x + b * x**3+c* x**4

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

    print(f"param_errors: {param_errors}")
    param_values = {}
    for name, value in zip(fit_param_names, params):
        param_values[name] = value
    

    # 计算拟合值
    y_fit = func(x,*params)

    
    plt.scatter(x, y, label="原始数据")
    plt.plot(x, y_fit, 'r-', label="拟合曲线")
    plt.legend()
    plt.grid(True) 
    plt.show()
    param_str = ", ".join([f"{name}={value}" for name, value in param_values.items()])
    paramError_str = ", ".join([f"{name}误差={error}" for name, error in zip(fit_param_names, param_errors)])
    print(f"拟合参数: {param_str}")
    print(f"拟合误差: {paramError_str}")
    return params
def regularizeData(X):
    for x in X:
        x=int(x//1000)
        x=x*1000
    return X
def generater2(x,y,m=24,n=32):
    asymmetry = 1.0
    
    R2 = m * n / 4 * (1 + asymmetry * asymmetry)
    

    dy = y - n / 2 + 0.5
    dx = (x - m / 2 + 0.5) * asymmetry
    r2 = (dx * dx + dy * dy) / R2
    return r2
def generater2(x, y, m=24, n=32, center_x=0.5, center_y=0.5, asymmetry=1.0):
    """
    Calculate the normalized squared distance from (x,y) to a custom center point.
    
    Args:
        x, y: Coordinates of the point to evaluate.
        m, n: Grid dimensions (default 24x32).
        center_x: X-coordinate of the center (normalized 0-1, default 0.5).
        center_y: Y-coordinate of the center (normalized 0-1, default 0.5).
        asymmetry: Asymmetry factor (default 1.0).
    
    Returns:
        r2: Normalized squared distance from (x,y) to the center.
    """
    
    # Convert normalized center to grid coordinates
    center_x_grid = center_x * m
    center_y_grid = center_y * n
    
    R2 = m * n / 4 * (1 + asymmetry * asymmetry)
    
    # Calculate distance from (x,y) to the custom center
    dy = y - center_y_grid
    dx = (x - center_x_grid) * asymmetry
    r2 = (dx * dx + dy * dy) / R2
    
    return r2
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
def generate_lut2(m: int, n: int, center_x: float = 0.52, center_y: float = 0.52, c_strength: float = 1.0):
    """
    Generate a lookup table for brightness adjustment with adjustable center.
    
    Args:
        m: Number of rows in block grid (original m)
        n: Number of columns in block grid (original n)
        center_x: X coordinate of center point (normalized 0-1, default 0.5)
        center_y: Y coordinate of center point (normalized 0-1, default 0.5)
        c_strength: Corner strength parameter
    
    Returns:
        2D numpy array of gain values ((m+1) x (n+1))
    """
    # Convert grid dimensions to number of points
    m_points = m + 1
    n_points = n + 1
    
    # Set default center if not specified
    
    # Convert normalized center coordinates to grid coordinates
    center_x_grid = center_x * m
    center_y_grid = center_y * n
    
    lut = [[0.0 for _ in range(n_points)] for _ in range(m_points)]
    
    asymmetry = 1.0
    R2 = m * n / 4 * (1 + asymmetry * asymmetry)
    
    for y in range(n_points):
        for x in range(m_points):
            # Calculate distance from current point to custom center
            dy = y - center_y_grid
            dx = (x - center_x_grid) * asymmetry
            r2 = (dx * dx + dy * dy) / R2
            lut[x][y] = r2  # reproduces the cos^4 rule
    
    return np.array(lut)
def valueCal(x, *params,func):

    y= func(x, *params)
    
    plt.scatter(x, y, label="拟合数据")
    for x_i, y_i in zip(x,y):
        plt.text(x_i, y_i, f"{x_i,(y_i)}", ha='center', va='bottom')
     # 打印每个点的坐标  
    plt.plot(x, y, 'r-', )
    print(f"x: {x.tolist()}")  # 打印每个点的坐标  
    y=regularizeData(y)  # 对x进行正则化处理

    print(f"y: {y.astype(int).tolist()}") 
    plt.legend()
    plt.show()
def processLSC(folderPath):
    """处理LSC数据"""
    full_paths, basenames = get_paths(folderPath, suffix=".yaml")
    for path, basename in zip(full_paths, basenames):
        print(f'Processing: {path}...')
        data = loadYaml(path)
        result = {}
        typeChannel = ['R', 'Gr', 'Gb', 'B']
        for t in typeChannel:
            dataLut= data[t]
        
            m = len(dataLut) 
            n = len(dataLut[0]) 
            xIdx = generate_lut2(m-1, n-1)
            dataLut = np.array(dataLut)
            # xIdx = xIdx[:, :xIdx.shape[1]//2]  
            # dataLut = dataLut[:, :dataLut.shape[1]//2] 
            dataLut = dataLut.flatten()
            xIdx = xIdx.flatten()
            dataLut = np.column_stack((xIdx, dataLut))
            
            params = curveFit(dataLut, func=func_1_r2_r4)
            
            lut = np.zeros((m , n ))
            for j in range(n ):
                for i in range(m ):
                    lut[i][j] = func_1_r2_r4(generater2(i, j, m, n), *params)
            result[t]= convert_numpy(lut)
        saveYaml(result, basename)
        
def main():


    # folderPath = r"C:\WorkSpace\serialPortVisualization\data\0901LscConfig"
    # processLSC(folderPath)   

    '''参数拟合部分'''
    data=np.array([[3174,2800],
                   [3232,2800],
                   [3242,2800],
                   [3270,2800],
                   [3894,3500],
                   [3856,3500],
                   [3910,3500],
                   [3904,3500],
                   [4369,4000],
                   [4448,4000],
                   [4469,4000],
                   [4409,4000],
                   [5243,5000],
                   [5158,5000],
                   [5258,5000],
                   [5108,5000],
                   [5951,6000],
                   [5937,6000],
                   [5703,6000],
                   [5860,6000],
                
                
    ])
    params=curveFit(data,func=func_quad)
    """"
    计算给定x值的拟合曲线y值
    """
    x=[]
    for i in range(0,10,1):
        x.append(i/10)
    x=np.array(x)
    # a, b, c = 36984.00,0.41,3431.71
    valueCal(x,*params,func=func_quad)

main()


