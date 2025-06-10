import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import matplotlib as mpl

# 设置更兼容的中文字体
mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 替代SimHei，解决符号缺失问题
# 初始化串口
ser = None
try:
    ser = serial.Serial(
        port='COM3',
        baudrate=115200,
        timeout=1,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    print(f"串口状态: {ser.is_open}")
except Exception as e:
    print(f"初始化错误: {e}")
    exit()

# 数据存储
data_dict = {
    # 'newGain': [0],
    # 'curGain': [0],
    'newExposure': [0],
    'curExposure': [0]
}
x = [0]  # 时间轴
# 创建图形和轴
num_plots = len(data_dict)
fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), squeeze=False)
axes = axes[0]  # 解包axes数组
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("实时数据可视化", fontsize=14)
# lines = {}
colors = plt.cm.tab10.colors  # 使用预定义颜色

lines = []
for ax, (name, data), color in zip(axes, data_dict.items(), colors):
    line, = ax.plot(x, data, label=name, color=color)
    lines.append(line)
    ax.set_title(name)
    ax.set_xlabel("时间")
    ax.set_ylabel("数值")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 固定图例在右侧
    ax.grid(True)  # 添加网格线
plt.tight_layout()

# 初始化四条曲线
# for ax, (title, ylabel, keys) in zip([ax1, ax2], 
#                                    [('增益', '值', ['newGain', 'curGain']), 
#                                     ('曝光', '值', ['newExposure', 'curExposure'])]):
#     ax.set_title(title)
#     ax.set_xlabel('时间')
#     ax.set_ylabel(ylabel)
#     ax.grid(True)
#     for key in keys:
#         line, = ax.plot([], [], label=key)
#         lines[key] = line
#     ax.legend()

# 调整子图间距
MAX_POINTS=100000
def update(frame):
    try:
        # 读取串口数据
        data = ser.readline()
        if data:
            line = data.decode('utf-8', errors='ignore').strip()
            if not line or line.startswith('#'):
                return lines.values()
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in data_dict:
                    try:
                        # 尝试转换为浮点数
                        num_value = float(value)
                        data_dict[key].append(num_value)
                        
                        # 限制数据点数量，防止图表过于拥挤
                        # for k in data_dict:
                        #     if len(data_dict[k]) > max_points:
                        #         data_dict[k] = data_dict[k][-max_points:]
                        min_Len=float('inf')
                        for l in data_dict.values():
                            min_Len=min(min_Len,len(l))
                        x= list(range(min_Len)) 
                        for line, data in zip(lines, data_dict.values()):
                            line.set_data(x[-min_Len:], data[-min_Len:]) 
                        # 更新曲线数据
                        # for k in lines:
                        #     lines[k].set_data(range(len(data_dict[k])), data_dict[k])
                        
                        # 自动调整坐标轴范围
                        for ax in axes:
                            ax.relim()
                            ax.autoscale_view()
                    except ValueError:
                        pass
        
        return lines
    
    except serial.SerialException as e:
        print(f"通信错误: {e}")
        ani.event_source.stop()  # 停止动画
        return lines.values()

# 创建动画
ani = FuncAnimation(fig, 
        update, 
        frames=range(MAX_POINTS), 
        interval=100, 
        blit=False,
        cache_frame_data=False   # 避免内存积累)
)
try:
    plt.show()
finally:
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")