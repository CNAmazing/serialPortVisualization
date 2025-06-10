import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

mpl.rcParams['font.family'] = 'SimHei'  # 设置中文字体

# 初始数据
data1 = [0]  # 第一组数据
data2 = [0]  # 第二组数据
x = [0]      # 时间轴

# 创建图形和坐标轴（2个子图，1行2列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 加宽图形以容纳图例
fig.suptitle("实时数据可视化", fontsize=14)  # 总标题

# 初始化两条折线
line1, = ax1.plot(x, data1, label="随机数据1", color="blue")
line2, = ax2.plot(x, data2, label="随机数据2", color="red")

# 设置子图标题、坐标轴标签
ax1.set_title("数据集1")
ax1.set_xlabel("时间")
ax1.set_ylabel("数值")

ax2.set_title("数据集2")
ax2.set_xlabel("时间")
ax2.set_ylabel("数值")

# 固定图例在图表右侧外部
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 调整子图布局，留出图例空间
plt.tight_layout()

def update(frame):
    # 生成新数据
    new_value1 = np.random.rand() * 10
    new_value2 = np.random.rand() * 5
    
    data1.append(new_value1)
    data2.append(new_value2)
    x.append(frame)
    
    # 更新折线数据
    line1.set_data(x, data1)
    line2.set_data(x, data2)
    
    # 调整坐标轴范围
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    return line1, line2

# 创建动画，每100毫秒更新一次
ani = FuncAnimation(fig, update, frames=range(100), interval=100, blit=False)
plt.show()