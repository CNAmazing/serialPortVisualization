import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scienceplots  # 导入科学绘图样式

# 使用SCI论文样式
plt.style.use(['science', 'no-latex'])  # 不使用LaTeX渲染
# 使用LaTeX渲染
plt.rcParams['font.size'] = 14          # 全局字体大小
# 数据
x = [55, 85, 120]
y1 = [0.707, 0.908, 1]    # 第一条曲线
y2 = [0.783, 0.942, 1]   # 第二条曲线
y3 = [0.822, 0.942, 1]    # 第三条曲线
y4 = [0.779, 0.916, 1]    # 第四条曲线

# 插值
x_new = np.linspace(min(x), max(x), 300)

def smooth_curve(x_orig, y_orig, kind='quadratic'):
    f = interpolate.interp1d(x_orig, y_orig, kind=kind, fill_value='extrapolate')
    return f(x_new)

y1_smooth = smooth_curve(x, y1)
y2_smooth = smooth_curve(x, y2)
y3_smooth = smooth_curve(x, y3)
y4_smooth = smooth_curve(x, y4)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制曲线
colors = ["#df0d0d", "#5b8039", '#2ca02c', "#080cff"]  # 默认颜色
# line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
labels = ['R', 'Gr', 'Gb', 'B']
for i, (y_smooth, y_data) in enumerate(zip([y1_smooth, y2_smooth, y3_smooth, y4_smooth], 
                                          [y1, y2, y3, y4])):
    ax.plot(x_new, y_smooth, color=colors[i],  
            label=f'{labels[i]}')
    ax.scatter(x, y_data, color=colors[i], s=60, edgecolors='white', 
              linewidth=1, zorder=5, marker=markers[i])

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Value')
ax.legend()

plt.tight_layout()
plt.show()