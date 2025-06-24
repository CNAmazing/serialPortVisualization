import numpy as np
import matplotlib.pyplot as plt

# 示例数据
x_data = np.array([50, 20, 80, 110, 0,60,40])
y_data = np.array([20000, 24000, 16000,8000, 32000,19000,21000])

# 使用 polyfit 进行三次拟合（deg=3）
coefficients = np.polyfit(x_data, y_data, deg=3)
a, b, c, d = coefficients

print(f"拟合参数: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

# 计算拟合曲线
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit + d  # 或者用 np.polyval(coefficients, x_fit)

# 可视化
plt.scatter(x_data, y_data, label="原始数据")
plt.plot(x_fit, y_fit, 'r-', label=f"三次拟合: y = {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("三次多项式拟合")
plt.show()