import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

# 设置更兼容的中文字体
mpl.rcParams['font.family'] = 'Microsoft YaHei'  # 替代SimHei，解决符号缺失问题
def read_and_split_by_equal(filename):
    """
    读取txt文件并按等号分割每行内容
    
    参数:
        filename: 要读取的txt文件路径
        
    返回:
        包含分割后内容的列表，每个元素是一个(key, value)元组
    """
    result = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                # 跳过空行和注释行(以#开头的行)
                if not line or line.startswith('#'):
                    continue
                # 按等号分割
                if '=' in line:
                    key, value = line.split('=', 1)  # 只分割第一个等号
                    key = key.strip()
                    value = value.strip()
                    result.append((key, value))
                else:
                    print(f"警告: 行 '{line}' 不包含等号，已跳过")
        return result
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
def Extract_numerical_values(filename,*args):
    """
    读取txt文件并按等号分割每行内容
    
    参数:
        filename: 要读取的txt文件路径
        
    返回:
        包含分割后内容的列表，每个元素是一个(key, value)元组
    """
    result = {name:[]for name in args}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                # 跳过空行和注释行(以#开头的行)
                if not line or line.startswith('#'):
                    continue
                # 按等号分割
                if '=' in line:
                    key, value = line.split('=', 1)  # 只分割第一个等号
                    if key in result:

                        key = key.strip()
                        value = value.strip()
                        result[key].append(float(value))
                    # result.append((key, value))
                else:
                    print(f"警告: 行 '{line}' 不包含等号，已跳过")
        return result
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def create_dynamic_plots(**kwargs):
    """
    创建动态更新的多子图动画（改进版）
    
    参数:
        kwargs: 键值对形式，键为子图名称，值为初始数据列表。
               例如: `温度=[0], 湿度=[0]` 会生成2个子图。
    
    返回:
        FuncAnimation 对象
    """
    # 初始数据
    data_dict = {name: list(data) for name, data in kwargs.items()}  # 确保数据是可变的列表
    x = [0]  # 时间轴
    MAX_POINTS = 10000  # 最大数据点数
    file_path = r"C:\Users\15696\workSpace\collectedData\ReceivedTofile-COM3-2025_6_10_14-22-50.DAT"  # 替换为你的文件路径
    # parsed_data = read_and_split_by_equal(file_path)
    # 创建图形和子图
    num_plots = len(data_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4), squeeze=False)
    axes = axes[0]  # 解包axes数组
    
    fig.suptitle("实时数据可视化", fontsize=14)
    
    # 初始化折线和颜色
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
    
    def update(frame):
        nonlocal x  # 声明x为非局部变量
        nonlocal file_path
        data_dict=Extract_numerical_values(file_path,'curGain','newGain','curExposure','newExposure')
        # 为每条折线生成新数据
        # for line, data in zip(lines, data_dict.values()):
        #     new_value = np.random.rand() * 10
        #     data=
        min_Len=float('inf')
        # x.append(frame)
        for l in data_dict.values():
            min_Len=min(min_Len,len(l))
        x= list(range(min_Len))  # 确保x的长度与数据一致
        
        
        # 更新所有折线数据（确保x和data长度一致）
        for line, data in zip(lines, data_dict.values()):
            line.set_data(x[-len(data):], data)  # 确保使用相同长度的数据
            
        # 调整所有坐标轴范围
        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        return lines
    
    # 创建动画
    ani = FuncAnimation(
        fig, 
        update, 
        frames=range(MAX_POINTS), 
        interval=100, 
        blit=False,
        cache_frame_data=False   # 避免内存积累
    )
    
    plt.show()
    return ani

# 示例调用

# newGain=[]
# curGain=[]
# newExposure=[]
# curExposure=[]
# if parsed_data:
#         for key, value in parsed_data:
#             value=float(value)
#             match key:
#                 case 'curExposure':
#                     curExposure.append(value)
#                 case 'newExposure':
#                     newExposure.append(value)
#                 case 'curGain':
#                     curGain.append(value)
#                 case 'newGain':
#                     newGain.append(value)

ani = create_dynamic_plots(
    原始增益=[0], 
    新增益=[0], 
    原始曝光=[0],
    新曝光=[0]
)