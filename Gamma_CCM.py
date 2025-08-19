import cv2
import numpy as np
import yaml
import os
from tools import *
def reverseGamma(img):
    mask = img <= 0.04045
    linear = np.zeros_like(img)
    linear[mask] = img[mask] / 12.92
    linear[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return linear
def Gamma(img):
    
    # img_encoded = cv2.imread(folderPath)
    mask = img <= 0.0031308
    srgb = np.zeros_like(img)
    srgb[mask] = img[mask] * 12.92
    srgb[~mask] = 1.055 * (img[~mask] ** (1/2.4)) - 0.055
    # return srgb
    return img**(1/2.2)

    # gamma = 2.2
    # img_gamma = np.power(img.astype(np.float64) , gamma)
    # imgName= folderPath.replace(".jpg","_reversegamma.jpg")
    # # 保存结果（需转换回 8-bit）
    # cv2.imwrite(imgName, (img_gamma * 255).astype(np.uint8))

def ccmApply(img,ccm):
    
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    corrected_flat = np.dot(rgb_flat, ccm.T)  # 等价于 (ccm @ rgb_flat.T).T
    corrected_image = corrected_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return corrected_image
def ccmApply_3x4(img,ccm):
    # 3. 应用CCM矫正（使用左乘）
    h, w = img.shape[:2] #H*W*3
    rgb_flat = img.reshape(-1, 3) # (h*w, 3)
    corrected_flat = np.dot(rgb_flat, ccm[:3,:].T)+ccm[3,:]  # 等价于 (ccm @ rgb_flat.T).T
    corrected_image = corrected_flat.reshape(h, w, 3)
    # 5. 裁剪并转换到8位
    return corrected_image
def rgb2yuv(rgb):
    if (rgb.ndim != 3) or (rgb.shape[2] != 3):
        raise ValueError("输入必须是形状为(N, 3)的RGB数组")
    r=rgb[:,:,0]
    g=rgb[:,:,1]
    b=rgb[:,:,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b + 0.5
    v =  0.615 * r - 0.515 * g - 0.100 * b + 0.5

    yuv = np.stack((y, u, v), axis=-1)
    yuv = np.clip(yuv, 0, 1)  # 确保YUV值在0-1范围内
    return yuv
def yuv2rgb(yuv):
    if (yuv.ndim != 3) or (yuv.shape[2] != 3):
        raise ValueError("输入必须是形状为(N, 3)的RGB数组")


    y=yuv[:,:,0]
    u=yuv[:,:,1]-0.5
    v=yuv[:,:,2]-0.5

    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u

    rgb = np.stack((r, g, b), axis=-1)
    rgb = np.clip(rgb, 0, 1)  # 确保RGB值在0-1范围内
    return rgb

def Contrast(img):
    contrastConfig = {
    'LUM_0' :50,
    'LUM_32': 80,
    'LUM_64' :100,
    'LUM_96' : 100,
    'LUM_128' : 100,
    'LUM_160' :100,
    'LUM_192' : 110,
    'LUM_224' : 140,
    'LUM_256' : 200,
    }
    config=[50,80,100,100,100,100,110,140,200]
    contrastRange = [0,32,64,96,128,160,192,224,256]
    yuv= rgb2yuv(img)
    y=yuv[:,:,0]
    gains=np.ones_like(y)  
    for i in  range(len(contrastRange)-1):
        rangeLeft=contrastRange[i]
        rangeRight=contrastRange[i+1] if i+1 < len(contrastRange) else 256

        mask=(y*256>=rangeLeft) & (y*256<rangeRight)
        gains[mask]=(y[mask]*256- rangeLeft)/(rangeRight- rangeLeft)* config[i+1]/100.0+(rangeRight-y[mask]*256)/(rangeRight- rangeLeft)* config[i]/100.0

    y=y*gains
    yuv[:, :, 0] = y  # 更新Y通道
    yuv=np.clip(yuv,0,1)

    rgb= yuv2rgb(yuv)

    return rgb
def img_uint8_to_float(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
    image_linear = imgRGB.astype(np.float64) / 255.0  # 归一化到[0,1]
    return image_linear
def main():

    # Gamma(r"C:\serialPortVisualization\corrected_image.jpg")
    folderPath = r"C:\serialPortVisualization\data\0819_4"
    yamlFile='ccmDict.yaml'


    full_paths, basenames= get_paths(folderPath, suffix=".jpg")
    yamlData= loadYaml(yamlFile)

    for path, basename in zip(full_paths, basenames):
        typeCT=getCTstr(basename)
        print(f'Processing: {path}, Type: {typeCT}')

        # print   (yamlData['D50'])
        CCM= np.array(yamlData[typeCT])
        print(npToString(CCM))
        img= cv2.imread(path)
        img=img_uint8_to_float(img)
        # img= ccmApply(img,CCM)
        img= ccmApply_3x4(img,CCM)
        img = np.clip(img, 0, 1) #img_CCM  范围0-1
        img= Gamma(img) #img_Gamma  范围0-1

        img= Contrast(img)
        #
        img = (img * 255).astype(np.uint8) #img_CCM  范围0-1 
        img = np.clip(img, 0, 255)

        imgName= basename+'_CCM.jpg'
        cv2.imwrite(imgName, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

main()
# img_Tmp = cv2.imread(imgName)

# img_reversegamma= np.power(img_Tmp.astype(np.float32) / 255.0, 1/gamma)
# # 保存反 Gamma 处理后的结果
# imgName_reversegamma = "image_reversegamma.png"
# cv2.imwrite(imgName_reversegamma, (img_reversegamma * 255).astype(np.uint8))