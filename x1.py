import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
# data = np.array([
#     [0, 1100],
#     [2000, 3000],
#     [9600, 4500],
#     [17000,7000],
#     [22000, 9000],
#     [31854, 14000],
#     [13000, 5300],
#     [11000, 5000],
#     [8500, 4950],
#     [0, 2200],
#     [12000, 5100],
#     [14000, 5700],
#     [27000, 17000],
#     [29000, 22000],
#     [30500, 22000],
#     [1300, 4600],
#     [17000, 9000],
#     [23700, 7000],
#     [3600, 3600],

# ])
data = np.array([
    [1, 10000],
    [10, 8000],
    [15, 5000],
    [20, 1000],
    [25, 0],


])
# 数据：输入色温和真实色温
cct_input = data[:,0]
cct_real = data[:,1]

# 定义二次函数
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c
def quadratic_1(x, a):
    return a * np.sqrt(x)
def quadratic_reverse(x,a,b,c):
    return a/(x+c)+b
def quadratic_ax_b(x,a,b):
    return a*x+b
# 拟合曲线
# params, _ = curve_fit(quadratic_1, cct_input, cct_real)
params, _ = curve_fit(quadratic_ax_b, cct_input, cct_real,maxfev=100000)

# a, b, c = params
a,b = params
# print(f"拟合参数: a={a}, b={b}, c={c}")
print(f"拟合参数: a={a:.10f},b={b:.10f}")
# 绘制拟合曲线
x_vals = np.linspace(min(cct_input), max(cct_input), 100)
# y_vals = quadratic(x_vals, a, b, c)
y_vals = quadratic_ax_b(x_vals, a,b)

plt.scatter(cct_input, cct_real, color='red', label='数据点')
plt.plot(x_vals, y_vals, color='blue', label='拟合曲线')
plt.xlabel('CCT 输入值 (K)')
plt.ylabel('CCT 真实值 (K)')
plt.legend()
plt.grid()
plt.show()