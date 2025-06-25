import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 数据
x = np.array([1, 2, 3, 4, 5, 5, 6, 6])
y = np.array([26000, 18000, 14000, 12000, 8000, 7000, 7000, 7000])

# 定义拟合函数（指数衰减 + 常数）
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# 拟合
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