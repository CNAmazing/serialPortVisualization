import cv2
import numpy as np

# 创建一个更大的模拟Bayer图像 (6x6, RGGB模式)
# 模式重复：
# R G R G R G
# G B G B G B
# R G R G R G
# G B G B G B
# R G R G R G
# G B G B G B
bayer_larger = np.array([
    [255, 0, 200, 0, 220,0],  # R, G, R, G, R, G
    [0,  0, 0, 0, 0, 0],  # G, B, G, B, G, B
    [230, 180, 240, 190, 250, 200],  # R, G, R, G, R, G
    [180, 130, 190, 140, 200, 150],  # G, B, G, B, G, B
    [210, 160, 230, 170, 240, 180],  # R, G, R, G, R, G
    [160, 110, 170, 120, 180, 130]   # G, B, G, B, G, B
], dtype=np.uint8)

print("原始Bayer数据 (6x6 RGGB):")
print(bayer_larger)

# 进行解马赛克转换
rgb_larger = cv2.cvtColor(bayer_larger, cv2.COLOR_BAYER_RGGB2RGB)

print("\n转换后的RGB图像形状:", rgb_larger.shape)
print("转换后的图像数据 (注意: OpenCV默认BGR顺序):")
print(rgb_larger)

# 查看中心区域的一个像素（避免边界效应）
print("\n中心区域像素 (2,2) 的BGR值:", rgb_larger[2, 2])
print("这应该对应原始Bayer中R位置(240)的插值结果")

# 查看各个通道
print("\nB通道 (中心4x4区域):")
print(rgb_larger[1:5, 1:5, 0])  # 蓝色通道
print("\nG通道 (中心4x4区域):")
print(rgb_larger[1:5, 1:5, 1])  # 绿色通道  
print("\nR通道 (中心4x4区域):")
print(rgb_larger[1:5, 1:5, 2])  # 红色通道