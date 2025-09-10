import serial
from datetime import datetime
import re 
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
# 初始化串口
def serial_init(port, baudrate=115200, timeout=None):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        print(f"串口已打开: {ser.is_open}")
        return ser
    except Exception as e:
        print(f"初始化错误: {e}")
        return None
def pltSaveFig(keyDict):
        
    # 创建画布
    plt.figure(figsize=(16, 8))

    # 存储所有标注对象
    texts = []
    now = datetime.now()
    saveName=now.strftime("%Y%m%d_%H_%M_%S")
    # 绘制每条曲线，并添加标记
    for key, values in keyDict.items():
        if not values:
            print(f"key={key}无数据，跳过......")
            continue
        values=np.array(values)
        maxValue=np.max(values)
        line = plt.plot(values/maxValue, label=key, marker='o', markersize=3, linestyle='-')
        for i, v in enumerate(values):
            # 添加标注到列表（暂不显示）
            texts.append(plt.text(i, v/maxValue, f'({i:.1f}, {v:.1f})', fontsize=2, ha='center', va='bottom'))

    
    adjust_text(
        texts, 
        # arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),  # 箭头样式
        # expand_points=(1.2, 1.2),  # 扩大标注移动范围
        # expand_text=(1.2, 1.2),     # 扩大文本间距
        # force_text=0.5,             # 调整文本间的排斥力
        # force_points=0.5            # 调整文本与点的排斥力
    )

    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"{saveName}")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"{saveName}.svg", dpi=300, bbox_inches='tight',format='svg')
    # plt.show()
def processDict(keyDict):
    return keyDict
def main():
    keyDict={
        'curGain':[],
        "newGain":[],
        "curExposure":[],
        "newExposure":[],
        "avgL_d":[],
        "avgL":[],
        "roundedCCT":[],
    }
    captureEndFlag=r"mediax save dng metadata success"
    videoEndFlag=r"mediax_transmit_stop succeeded"
    endFlag=f"{captureEndFlag}|{videoEndFlag}"
    ser = serial_init('COM11', 2000000)  # 修改为你的串口号和波特率
    try:
        print("开始接收数据(按Ctrl+C停止)...")
        while True:
            # 读取串口数据
            if ser.in_waiting > 0:
                data = ser.readline()
                try:
                    line = data.decode('utf-8', errors='ignore').strip()
                    if line:  # 只打印非空行
                        # 使用正则表达式匹配等号前后的内容
                        print(f"{line}")

                        endMatch=re.search(videoEndFlag, line)
                        if endMatch:
                            print("接收结束，保存...")

                            pltSaveFig(keyDict)
                            for k in keyDict:
                                keyDict[k].clear()
                            print("继续接收数据...")
                            continue
                        match = re.search(r'([^\s=,]+)\s*=\s*([^\s=,]+)', line) 
                        if match:
                            # 如果匹配成功，提取键值对
                            key = match.group(1)
                            value = match.group(2)
                            if key in keyDict:
                                try:
                                    keyDict[key].append(float(value))
                                except ValueError:
                                    print(f"无法将值转换为浮点数: {value}")
                        
                except UnicodeDecodeError:
                    print(f"接收非文本数据: {data}")
                    
    except KeyboardInterrupt:
        print("\n用户中断，停止接收数据")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")



if __name__ == "__main__":
    main()