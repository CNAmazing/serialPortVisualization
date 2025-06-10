import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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
data_arrays = {
    'newGain': [],
    'curGain': [],
    'newExposure': [],
    'curExposure': []
}

# 创建图形和轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('实时数据监控')
lines = {}

# 初始化四条曲线
for ax, (title, ylabel, keys) in zip([ax1, ax2], 
                                   [('增益', '值', ['newGain', 'curGain']), 
                                    ('曝光', '值', ['newExposure', 'curExposure'])]):
    ax.set_title(title)
    ax.set_xlabel('时间')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    for key in keys:
        line, = ax.plot([], [], label=key)
        lines[key] = line
    ax.legend()

# 调整子图间距
plt.tight_layout()

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
                
                if key in data_arrays:
                    try:
                        # 尝试转换为浮点数
                        num_value = float(value)
                        data_arrays[key].append(num_value)
                        
                        # 限制数据点数量，防止图表过于拥挤
                        max_points = 10000
                        for k in data_arrays:
                            if len(data_arrays[k]) > max_points:
                                data_arrays[k] = data_arrays[k][-max_points:]
                        
                        # 更新曲线数据
                        for k in lines:
                            lines[k].set_data(range(len(data_arrays[k])), data_arrays[k])
                        
                        # 自动调整坐标轴范围
                        for ax in [ax1, ax2]:
                            ax.relim()
                            ax.autoscale_view()
                    except ValueError:
                        pass
        
        return lines.values()
    
    except serial.SerialException as e:
        print(f"通信错误: {e}")
        ani.event_source.stop()  # 停止动画
        return lines.values()

# 创建动画
ani = FuncAnimation(fig, update, interval=100, blit=True)

try:
    plt.show()
finally:
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")