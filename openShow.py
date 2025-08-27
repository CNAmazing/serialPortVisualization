import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# 设置SCI论文风格 - Times New Roman字体
plt.style.use('default')  # 使用默认样式，更专业
plt.rcParams['font.family'] = 'Times New Roman'  # 使用Times New Roman字体，科学论文常用
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5

# 确保数学符号也使用Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX字体与Times New Roman兼容
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 数据
x = [55, 85, 120]
y1 = [0.707, 0.908, 1]    # 第一条曲线
y2 = [0.783, 0.942, 1]   # 第二条曲线
y3 = [0.822, 0.942, 1]    # 第三条曲线
y4 = [0.779, 0.916, 1]    # 第四条曲线

# 创建插值 - 使用二次插值避免错误
x_new = np.linspace(min(x), max(x), 300)

def smooth_curve(x_orig, y_orig, kind='quadratic'):
    f = interpolate.interp1d(x_orig, y_orig, kind=kind, fill_value='extrapolate')
    return f(x_new)

y1_smooth = smooth_curve(x, y1)
y2_smooth = smooth_curve(x, y2)
y3_smooth = smooth_curve(x, y3)
y4_smooth = smooth_curve(x, y4)

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 使用SCI论文常用配色（更沉稳）
colors = ['#2E86AB', '#A23B72', '#F18F01', '#39603D']  # 蓝色、紫色、橙色、绿色
line_styles = ['-', '--', '-.', ':']  # 不同的线型
markers = ['o', 's', '^', 'D']  # 不同的标记

# 绘制四条曲线
for i, (y_smooth, y_data) in enumerate(zip([y1_smooth, y2_smooth, y3_smooth, y4_smooth], 
                                          [y1, y2, y3, y4])):
    ax.plot(x_new, y_smooth, color=colors[i], linestyle=line_styles[i], 
            label=f'Curve {i+1}')
    ax.scatter(x, y_data, color=colors[i], s=80, edgecolors='white', 
              linewidth=1.5, zorder=5, marker=markers[i])

# 美化图表
ax.set_xlabel('X Coordinate', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Value', fontsize=14, fontweight='bold')
ax.set_title('Four Curves Visualization', fontsize=16, fontweight='bold', pad=20)

# 添加图例
ax.legend(frameon=True, fancybox=False, shadow=False, 
         edgecolor='black', fontsize=11)

# 网格设置（更 subtle）
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# # 设置坐标轴范围
# ax.set_xlim(50, 125)
# ax.set_ylim(0, 12)

# 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 调整刻度
ax.tick_params(axis='both', which='major', width=1.5, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=4)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 保存为高质量图片（SCI论文常用格式）
# plt.savefig('sci_curves.png', dpi=300, bbox_inches='tight')
# plt.savefig('sci_curves.pdf', bbox_inches='tight')  # 矢量图格式