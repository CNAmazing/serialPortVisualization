import cv2
import matplotlib.pyplot as plt
from tools import readRaw
# 1. 读取一张图像
# image = cv2.imread(r'C:\WorkSpace\serialPortVisualization\data\g07s5ColorChecker\demosaicResults\A.jpg')
image_path=r'C:\WorkSpace\serialPortVisualization\data\1016_671\24\671-A.raw'
image=readRaw(image_path,3072,4096)
# 将BGR格式转换为RGB格式（Matplotlib使用RGB）
image_rgb = image
# image_rgb = cv2.cvtColor(image, cv2.IMREAD_UNCHANGED)

# 2. 定义矩形的参数s
boxSize = 220
interval = 265

# 起始坐标（第一个方块的左上角）
start_x =709
start_y = 692
boxList=[]
# 3. 循环生成24个等距离的矩形（6行4列）
for idx in range(24):

    row = idx // 6  # 行索引（0-3）
    col = idx % 6   # 列索引（0-5）
    
    # 计算当前方块的左上角和右下角坐标
    top_left_corner = (start_x + col * (boxSize + interval), 
                       start_y + row * (boxSize + interval))
    
    bottom_right_corner = (top_left_corner[0] + boxSize, 
                           top_left_corner[1] + boxSize)
    
    # 在图像上绘制矩形（红色边框）
    cv2.rectangle(image_rgb, top_left_corner, bottom_right_corner, (255, 0, 0), 2)
    boxList.append([top_left_corner[0],top_left_corner[1],bottom_right_corner[0],bottom_right_corner[1]])
    # 可选：在每个方块中心添加编号
    center_x = top_left_corner[0] + boxSize // 2
    center_y = top_left_corner[1] + boxSize // 2
  
# 4. 使用Matplotlib显示图像
print(boxList)
plt.figure(figsize=(15, 10))
plt.imshow(image_rgb)
plt.title(f'24 Equal Distance Boxes (Size: {boxSize}, Interval: {interval})')
plt.axis('off')

# 5. 显示图形
plt.tight_layout()
plt.show()