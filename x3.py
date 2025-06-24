from scipy.interpolate import CubicSpline
import numpy as np

# 原始控制点（可能不平滑）
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 2, 1, 3, 1])

# 生成平滑的三次样条曲线（默认C²连续）
cs = CubicSpline(x, y, bc_type='natural')  # 自然边界条件

# 绘制结果
import matplotlib.pyplot as plt
xs = np.linspace(0, 4, 100)
plt.plot(x, y, 'o', label='控制点')
plt.plot(xs, cs(xs), label='平滑曲线')
plt.legend()
plt.show()