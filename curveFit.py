import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Microsoft YaHei'

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
def curveFit(data):
    sorted_indices = np.argsort(data[:, 0])  # 获取排序索引
    data = data[sorted_indices]
    x = data[:,0]
    y = data[:,1]
    params, covariance = curve_fit(func, x, y, p0=[20000, 0.1, 7000])  # p0 是初始猜测

    a, b, c = params

    # 计算拟合值
    y_fit = func(x, a, b, c)

    # 绘图
    plt.scatter(x, y, label="原始数据")
    plt.plot(x, y_fit, 'r-', label=f"拟合曲线: {a:.2f} * exp(-{b:.2f}x) + {c:.2f}")
    plt.legend()
    plt.show()

    # 输出参数
    print(f"拟合参数: a={a:.2f}, b={b:.2f}, c={c:.2f}")
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
    
    curveFit(data)

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


