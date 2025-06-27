import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import inspect
mpl.rcParams['font.family'] = 'Microsoft YaHei'

# 指数衰减函数: a * e^(-k*x) + c
def func_exp(x, a, k, c):
    return a * np.exp(-k * x) + c

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
    params, covariance = curve_fit(func, x, y, p0=p0)
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
    print(f"拟合参数: {param_str}")
    
def valueCal(x, a, b, c):

    y= func(x, a, b, c)
    plt.scatter(x, y, label="拟合数据")
    for x_i, y_i in zip(x,y):
        plt.text(x_i, y_i, f"{x_i,int(y_i)}", ha='center', va='bottom')  
    plt.plot(x, y, 'r-', )
    plt.legend()
    plt.show()

def main():
    data = np.array([
           [1, 32000],
        [1, 26000],
        [3, 15000],
        [5, 8000],
        [6, 7000],
        [7, 5000],
        [7, 5000],
        [8, 5000],
        [8, 5000],

        [0, 41000],
        [0, 40000],
        [1, 26000],
        [2, 18000],
        [3, 14000],
        [4, 12000],
        [5, 8000],
        [5, 7000],
        [6, 7000],
        [6, 7000],
     
        [11, 4000],


    ])
    
    curveFit(data,func=func_sqrt)

    """"
    计算给定x值的拟合曲线y值
    """
    # x=[]
    # for i in range(15):
    #     x.append(i)
    # print(x)
    # x=np.array(x)
    # a, b, c = 36876.44,0.42,3594.35
    # valueCal(x, a, b, c)

main()


