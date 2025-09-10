import matplotlib.pyplot as plt
from adjustText import adjust_text

# 示例数据
keyDict = {
    'curGain': [1.2, 1.5, 1.8, 2.1, 2.4],
    'newGain': [0.8, 1.0, 1.2, 1.4, 1.6],
    'curExposure': [10, 15, 20, 25, 30],
    'newExposure': [5, 10, 15, 20, 25],
    'avgL_d': [50, 55, 60, 65, 70],
}

def pltSaveFig(keyDict):
        
    # 创建画布
    plt.figure(figsize=(16, 8))

    # 存储所有标注对象
    texts = []

    # 绘制每条曲线，并添加标记
    for key, values in keyDict.items():
        line = plt.plot(values, label=key, marker='o', markersize=8, linestyle='-')
        for i, v in enumerate(values):
            # 添加标注到列表（暂不显示）
            texts.append(plt.text(i, v, f'({i}, {v})', fontsize=8, ha='center', va='bottom'))

    # 自动调整标注位置（避免重叠）
    adjust_text(
        texts, 
        # arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),  # 箭头样式
        # expand_points=(1.2, 1.2),  # 扩大标注移动范围
        # expand_text=(1.2, 1.2),     # 扩大文本间距
        # force_text=0.5,             # 调整文本间的排斥力
        # force_points=0.5            # 调整文本与点的排斥力
    )

    # 添加图例、标题、坐标轴标签
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Name")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("Name.png", dpi=300, bbox_inches='tight')
    plt.show()


pltSaveFig(keyDict)