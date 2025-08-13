import numpy as np
import cv2

# 1. 加载图像
image = cv2.imread(r'C:\serialPortVisualization\data\0812_2\current_isp_configCWF_AfterCalib.png')
if image is None:
    raise ValueError("无法加载图像，请检查路径是否正确")

# 2. 定义CCM矩阵
ccm = np.array([
    [1.5413319658370959, -0.8752036171037689, 0.33387165126667295],
    [-0.6024030123362651, 1.119865991262268, 0.48253702107399715],
    [-0.09472281312418553, -1.108186241209772, 2.2029090543339573]
])

# 3. 转换为RGB并归一化
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_linear = image_rgb.astype(np.float32) / 255.0  # 确保是float32

# 4. 应用CCM矫正
h, w = image_linear.shape[:2]
rgb_flat = image_linear.reshape(-1, 3)
corrected_flat = np.dot(rgb_flat, ccm.T)  # 等价于 (ccm @ rgb_flat.T).T
corrected_image = corrected_flat.reshape(h, w, 3)

# 5. 裁剪并转换到8位
corrected_image = np.clip(corrected_image, 0, 1)
corrected_image_8bit = (corrected_image * 255).astype(np.uint8)

# 6. 保存为PNG（确保无Alpha通道）
if corrected_image_8bit.shape[2] == 4:
    corrected_image_8bit = corrected_image_8bit[:, :, :3]  # 移除Alpha通道
cv2.imwrite('corrected_image555.png', cv2.cvtColor(corrected_image_8bit, cv2.COLOR_RGB2BGR))