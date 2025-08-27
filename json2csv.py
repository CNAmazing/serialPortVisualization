import json
import csv
import pandas as pd
import numpy as np
import pandas as pd
from tools import *
def read_json_file(json_file):
    """
    读取JSON文件并返回字典数据
    
    参数:
    json_file: JSON文件路径
    
    返回:
    dict: 包含JSON数据的字典
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"JSON文件读取成功: {json_file}")
        return data
    except Exception as e:
        print(f"读取JSON文件时出现错误: {e}")
        return None

def save_dict_to_csv(data, csv_file):
    """
    将字典数据保存为CSV文件
    
    参数:
    data: 要保存的字典数据
    csv_file: 输出的CSV文件路径
    """
    try:
        # 如果数据是字典而不是列表，将其转换为列表
        if isinstance(data, dict):
            data = [data]
        
        # 获取所有字段名
        fieldnames = set()
        for item in data:
            if isinstance(item, dict):
                fieldnames.update(item.keys())
        
        # 写入CSV文件
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                if isinstance(item, dict):
                    writer.writerow(item)
        
        print(f"数据已成功保存为CSV文件: {csv_file}")
        
    except Exception as e:
        print(f"保存CSV文件时出现错误: {e}")

def save_dict_to_csv_pandas(data, csv_file):
    """
    使用pandas将字典数据保存为CSV文件（推荐方法）
    
    参数:
    data: 要保存的字典数据
    csv_file: 输出的CSV文件路径
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"数据已成功保存为CSV文件: {csv_file}")
        
    except Exception as e:
        print(f"保存CSV文件时出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    input_json = r"C:\WorkSpace\serialPortVisualization\data\0826_LSC_681\#681.json"  # 替换为你的JSON文件路径
    output_csv = "output.csv"  # 替换为你想要的输出CSV文件路径
    
    # 第一部分：读取JSON文件
    json_data = read_json_file(input_json)
    
    if json_data is not None:
        # print("读取到的数据:", json_data)
        LSCdata = json_data['LSC']
        for key, value in LSCdata.items():
            channelList=['r','gr','gb','b']
            # arrList=[]
            # for channel in channelList:
            #     channelData = value[channel]
            #     channelData=np.array(channelData)
            #     channelData=channelData[:825]

            #     channelData=channelData.reshape(25,33)
            #     value[channel]=channelData
            # # save_dict_to_csv(value[channel], key+".csv")
            # df = pd.DataFrame(channelData)
            # df.to_csv(f'{key}.csv', index=False, encoding='utf-8')

            with open(f'#681_{key}K.csv', 'w', encoding='utf-8') as f:
                for channel in channelList:
                    channelData = value[channel]
                    channelData = np.array(channelData)
                    channelData = channelData[:825]
                    channelData = channelData.reshape(25, 33)
                    channelData = channelData.astype(float) / 1000
                    # 更新字典中的值（如果需要）
                    value[channel] = channelData
                    
                    # 在每个通道数据前添加空行（除了第一个通道）
                    if channel != channelList[0]:
                        f.write('\n')
                    
                    # 写入通道标识
                    if channel == 'r':
                        channel_name = "R通道"
                    elif channel == 'gr':
                        channel_name = "GR通道"
                    elif channel == 'gb':
                        channel_name = "GB通道"
                    elif channel == 'b':
                        channel_name = "B通道"
                    f.write(f'{channel_name}\n')
                    
                    # 将数据写入CSV格式，使用特定的行结束符
                    df = pd.DataFrame(channelData)
                    df.to_csv(f, index=False, header=False, lineterminator='\n')
                    # 在每个通道数据后添加空行
                    # f.write('\n')
        # result=json_data['LSC']
        # save_dict_to_csv(result, output_csv) 

        