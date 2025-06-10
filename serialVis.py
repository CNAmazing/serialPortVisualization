import serial
# import time

ser = None
newGain=[]
curGain=[]
newExposure=[]
curExposure=[]
try:
    # 串口初始化放在循环外（只打开一次）
    ser = serial.Serial(
        port='COM3',
        baudrate=115200,
        timeout=1,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    print(f"串口状态: {ser.is_open}")

    while True:
        try:
            # 发送数据
            # ser.write(b'Hello, Serial!')
            
            # 接收数据（带超时和错误处理）
            data = ser.readline()
            if data:
                # print(f"接收到的数据: {data.decode('utf-8', errors='ignore')}")
                line= data.decode('utf-8', errors='ignore').strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)  # 只分割第一个等号
                    key = key.strip()
                    value = value.strip()
                    match key:
                        case 'curExposure':
                            curExposure.append(value)
                        case 'newExposure':
                            newExposure.append(value)
                        case 'curGain':
                            curGain.append(value)
                        case 'newGain':
                            newGain.append(value)
                    # result.append((key, value))
            else:
                print("未收到数据")
            
            # time.sleep(0.5)  # 避免CPU跑满
            
        except serial.SerialException as e:
            print(f"通信错误: {e}")
            break  # 发生严重错误时退出循环

except Exception as e:
    print(f"初始化错误: {e}")

finally:
    if ser and ser.is_open:
        ser.close()
        print("串口已关闭")