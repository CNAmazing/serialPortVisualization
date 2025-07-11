import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import inspect
mpl.rcParams['font.family'] = 'Microsoft YaHei'

# 指数衰减函数: a * e^(-k*x) + c
def func_exp(x, a, k, c):
    return a * np.exp(-k * x )+ c
def multi_exp(x, a1, k1, a2, k2, c):
    return a1 * np.exp(-k1 * x) + a2 * np.exp(-k2 * x) + c
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
    plt.show()
    param_str = ", ".join([f"{name}={value:.2f}" for name, value in param_values.items()])
    paramError_str = ", ".join([f"{name}误差={error:.2f}" for name, error in zip(fit_param_names, param_errors)])
    print(f"拟合参数: {param_str}")
    print(f"拟合误差: {paramError_str}")
    return params
def regularizeData(X):
    for x in X:
        x=int(x//1000)
        x=x*1000
    return X
def valueCal(x, *params,func):

    y= func(x, *params)
    plt.scatter(x, y, label="拟合数据")
    for x_i, y_i in zip(x,y):
        plt.text(x_i, y_i, f"{x_i,int(y_i)}", ha='center', va='bottom')
     # 打印每个点的坐标  
    plt.plot(x, y, 'r-', )
    y=regularizeData(y)  # 对x进行正则化处理
    print(f"x: {x.tolist()}")  # 打印每个点的坐标  

    print(f"y: {y.astype(int).tolist()}") 
    plt.legend()
    plt.show()

def main():
    # data = np.array([
    #     [1, 32000],
    #     [1, 26000],
    #     [3, 15000],
    #     [5, 8000],
    #     [6, 7000],
    #     [7, 5000],
    #     [7, 5000],
    #     [8, 5000],
    #     [8, 5000],

    #     [0, 41000],
    #     [0, 40000],
    #     [1, 26000],
    #     [2, 18000],
    #     [3, 14000],
    #     [4, 12000],
    #     [5, 8000],
    #     [5, 7000],
    #     [6, 7000],
    #     [6, 7000],
     
    #     [11, 4000],
    #     [15, 3000],

    #     [15, 3000],
    #     [7, 5000],
    #     [10, 4000],
    #     [15, 3000],
    #     [2,10000],
    #     [5, 8000],
    #     [5, 8000],
    #     [7,6000],
    #     [5, 8000],
    #     [15,4000],
    #     [12,4000],
    #     [10,4000],
    #     [5, 8000],
    #     [7,5000],
    #     [2,14000],
    #     [2,10000],
    #     [2,16725],
    # ])
    data=np.array([
                  [7.4,9700],
                  [11.2,4000],
                  [15,3000],
                  [18.4,2500],
                  [23.6,2000],
                  [13.9,5000],
                  [9.8,6000],
                  [11.3,5000],
                  [10,6000],
                  [9,6000],
                  [0,62000],
                  [1,56000],
                  [2,38000],
                  [2.5,34000],
                  [1.9,44000],
                  [1.8,50000],
                  [2.8,26000],
                  [4.1,16000],
                  [6,11000],
                  [10.2,6000],
                  [6.7,8000],
                  [9,6000],
                  [6.9,9000],
                  [14.5,4200],
                  [13,4700],
                
              
                  ])
    params=curveFit(data,func=multi_exp)

    """"
    计算给定x值的拟合曲线y值
    """
    x=[]
    for i in range(26):
        x.append(i)
    x=np.array(x)
    # a, b, c = 36984.00,0.41,3431.71
    valueCal(x,*params,func=multi_exp)

main()


