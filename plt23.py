import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text  # 导入自动避让功能

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制曲线
plt.plot(x, y, 'b-', label='sin(x)', alpha=0.3, linewidth=1)

# 存储所有文本对象
texts = []

# 标记点并添加文本（减少密度）
mark_interval = 1  # 适当增大间隔，减少标签数量
for i in range(0, len(x), mark_interval):
    # 标记点（半透明）
    plt.scatter(x[i], y[i], color='red', s=20, alpha=0.7, edgecolor='none', zorder=5)
    
    # 添加文本标签（先不调整位置）
    label_text = f"({x[i]:.1f}, {y[i]:.2f})"
    t = plt.text(x[i], y[i], label_text,
                 ha='center', va='center', color='black',
                 fontsize=2, zorder=10)
    texts.append(t)

# 使用 adjust_text 自动调整文本位置，避免重叠
adjust_text(texts, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
            expand_points=(1.2, 1.2),  # 扩大调整范围
            expand_text=(1.2, 1.2),    # 扩大文本间距
            force_text=0.2,            # 调整文本位置的力度
            force_points=0.2,          # 调整标记点位置的力度
            lim=50)                    # 最大调整次数

plt.title('Sine Wave with Non-overlapping Labels', pad=20)
plt.legend(loc='upper right', framealpha=0.5)
plt.grid(alpha=0.2)
plt.savefig('non_overlapping_labels_sine.svg', format='svg', bbox_inches='tight', dpi=120)
plt.close()