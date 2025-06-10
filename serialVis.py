import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import time
# 设置中文字体
mpl.rcParams['font.family'] = 'Microsoft YaHei'

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

# 配置参数
MAX_POINTS =  10000# 限制显示的数据点数
PLOT_INTERVAL = 50  # 绘图间隔(ms)

# 数据存储
data_dict = {
    'newGain': [],
    'curGain': [],
    'newExposure': [],
    'curExposure': []
}
timestamps = []

# 创建图形
fig, axes = plt.subplots(1, len(data_dict), figsize=(16, 8))
if len(data_dict) == 1:
    axes = [axes]  # 确保axes总是列表
fig.suptitle("实时数据可视化", fontsize=14)

# 初始化曲线
lines = []
for ax, (name, _), color in zip(axes, data_dict.items(), plt.cm.tab10.colors):
    line, = ax.plot([], [], label=name, color=color)
    lines.append(line)
    ax.set_title(name)
    ax.set_xlabel("时间")
    ax.set_ylabel("数值")
    ax.legend()
    ax.grid(True)
plt.tight_layout()

# 初始化时间戳
start_time = None

def update(frame):
    global start_time
    
    try:
        # 读取串口数据
        while ser.in_waiting > 0:
            data = ser.readline()
            if data:
                line = data.decode('utf-8', errors='ignore').strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key in data_dict:
                        try:
                            # 记录当前时间
                            current_time = time.time()
                            if start_time is None:
                                start_time = current_time
                            
                            # 转换为数值
                            num_value = float(value)
                            data_dict[key].append(num_value)
                            timestamps.append(current_time - start_time)
                            
                            # 限制数据点数量
                            if len(timestamps) > MAX_POINTS:
                                del timestamps[0]
                                for k in data_dict:
                                    if len(data_dict[k]) > MAX_POINTS:
                                        del data_dict[k][0]
                            
                        except ValueError:
                            pass
        
        # 更新所有曲线
        if timestamps:
            for line, data in zip(lines, data_dict.values()):
                line.set_data(timestamps[-len(data):], data[-len(data):])
            
            # 自动调整坐标轴范围
            for ax in axes:
                ax.relim()
                ax.autoscale_view(True, True, True)
        
        return lines
    
    except serial.SerialException as e:
        print(f"通信错误: {e}")
        ani.event_source.stop()
        return lines

# 创建动画
ani = FuncAnimation(
    fig, 
    update, 
    interval=PLOT_INTERVAL, 
    blit=False,
    cache_frame_data=False
)

try:
    plt.show()
finally:
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")