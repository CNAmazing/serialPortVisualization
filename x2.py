import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.array([
    [38000, 8000],
    [40000, 10000],
    [43000, 11000],
    [3600, 3600],
])

cct_input = data[:, 0]
cct_real = data[:, 1]

# 定义翻转的平方根函数
def flipped_sqrt(x, a, C):
    return a * (C - np.sqrt(x))

# 初始化 C 为 sqrt(max(x)) 的估计值
C_guess = np.sqrt(max(cct_input))

# 拟合曲线
params, _ = curve_fit(flipped_sqrt, cct_input, cct_real, p0=[1.0, C_guess])
a, C = params

print(f"拟合参数: a={a:.5f}, C={C:.5f}")

# 绘制拟合曲线
x_vals = np.linspace(min(cct_input), max(cct_input), 100)
y_vals = flipped_sqrt(x_vals, a, C)

plt.scatter(cct_input, cct_real, color='red', label='数据点')
plt.plot(x_vals, y_vals, color='blue', label='翻转平方根拟合')
plt.xlabel('CCT 输入值 (K)')
plt.ylabel('CCT 真实值 (K)')
plt.legend()
plt.grid()
plt.show()