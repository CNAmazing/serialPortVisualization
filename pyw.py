import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import time
import sys

# 设置中文显示（需要系统有中文字体）
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# 初始化串口
ser = None
try:
    ser = serial.Serial(
        port='COM5',
        baudrate=115200,
        timeout=0.5,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    print(f"串口状态: {ser.is_open}")
except Exception as e:
    print(f"初始化错误: {e}")
    sys.exit()

# 配置参数
MAX_POINTS = 1000  # 限制显示的数据点数
PLOT_INTERVAL = 50  # 绘图间隔(ms)

# 数据存储
data_dict = {
    'newGain': [],
    'avgL': [],
    'avgR': [],
    'avgG': [],
    'avgB': [],
}
timestamps = []

# 创建Qt应用
app = pg.mkQApp("Serial Data Monitor")

# 创建主窗口
win = pg.GraphicsLayoutWidget(title="实时数据可视化", size=(1200, 800))
win.show()

# 创建子图
plots = []
curves = []
for i, name in enumerate(data_dict.keys()):
    if i > 0:
        win.nextRow()
    plot = win.addPlot(title=name)
    plot.showGrid(x=True, y=True)
    plot.setLabel('left', '数值')
    plot.setLabel('bottom', '时间', 's')
    curve = plot.plot(pen=pg.mkPen(color=pg.intColor(i), width=2))
    plots.append(plot)
    curves.append(curve)

# 初始化时间戳
start_time = None

def update():
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
            for curve, data in zip(curves, data_dict.values()):
                curve.setData(timestamps[-len(data):], data[-len(data):])
    
    except serial.SerialException as e:
        print(f"通信错误: {e}")
        timer.stop()
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")

# 创建定时器
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(PLOT_INTERVAL)

# 启动应用
if __name__ == '__main__':
    try:
        pg.exec()
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")