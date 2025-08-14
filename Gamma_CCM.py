import cv2
import numpy as np
import yaml
import os
def loadYaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data
def get_paths(folder_name, suffix=".csv"):
    """
    递归获取指定文件夹及其子目录中的所有suffix图片路径及不带后缀的文件名
    
    参数:
        folder_name (str): 目标文件夹名称（如"x"）
        suffix (str): 文件后缀，默认".jpg"
        
    返回:
        tuple: (完整路径列表, 不带后缀的文件名列表)，如(
                ["x/images/pic1.jpg", "x/subdir/pic2.jpg"], 
                ["pic1", "pic2"]
               )
    """
    full_paths = []
    basenames = []
    
    try:
        if not os.path.exists(folder_name):
            raise FileNotFoundError(f"目录不存在: {folder_name}")
            
        # 使用 os.walk 递归遍历所有子目录
        for root, dirs, files in os.walk(folder_name):
            for f in files:
                if f.lower().endswith(suffix):
                    file_path = os.path.join(root, f)
                    if os.path.isfile(file_path):
                        full_paths.append(file_path)
                        basenames.append(os.path.splitext(f)[0])
                        
        return full_paths, basenames
        
    except Exception as e:
        print(f"错误: {e}")
        return [], []
# 读取图片
def test():
    typeDict=["D50", "CWF", "H", "U30"]
    for t in typeDict:
            
        img_encoded = cv2.imread(fr"C:\serialPortVisualization\current_isp_config{t}_AfterCalib.png")

        # 反 Gamma 处理
        gamma = 2.4
        img_gamma = np.power(img_encoded.astype(np.float32) / 255.0, 1/gamma)

        imgName= f"{t}.png"
        # 保存结果（需转换回 8-bit）
        cv2.imwrite(imgName, (img_gamma * 255).astype(np.uint8))

def Gamma(img):
    # img_encoded = cv2.imread(folderPath)

    # 反 Gamma 处理
    gamma = 2.2
    img_gamma = np.power(img.astype(np.float32) , 1/gamma)
    return img_gamma
    # imgName= folderPath.replace(".jpg","_reversegamma.jpg")
    # # 保存结果（需转换回 8-bit）
    # cv2.imwrite(imgName, (img_gamma * 255).astype(np.uint8))
def Save(img, imgName):
    
    cv2.imwrite(imgName, (img * 255).astype(np.uint8))
def getCTstr(file_path):
    file_path=str(file_path)
    if 'U30' in file_path:
        return 'U30'
    elif 'CWF' in file_path:
        return 'CWF'
    elif 'D50' in file_path:
        return 'D50'
    elif 'H' in file_path:
        return 'H'
def ccmApply(img,ccm):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB
    image_linear = imgRGB.astype(np.float64) / 255.0  # 归一化到[0,1]

    # 3. 应用CCM矫正（使用左乘）
    h, w = image_linear.shape[:2] #H*W*3
    rgb_flat = image_linear.reshape(-1, 3) # (h*w, 3)
    corrected_flat = np.dot(rgb_flat, ccm.T)  # 等价于 (ccm @ rgb_flat.T).T
    corrected_image = corrected_flat.reshape(h, w, 3)

    # 5. 裁剪并转换到8位
    return corrected_image
def main():

    # Gamma(r"C:\serialPortVisualization\corrected_image.jpg")
    folderPath = r"C:\serialPortVisualization\data\0814_1_rgb2"
    yamlFile='ccmDict.yaml'


    full_paths, basenames= get_paths(folderPath, suffix=".jpg")
    yamlData= loadYaml(yamlFile)

    for path, basename in zip(full_paths, basenames):
        typeCT=getCTstr(basename)

        # print   (yamlData['D50'])
        CCM= np.array(yamlData[typeCT])
        print (CCM.shape)
        print(CCM)
        img= cv2.imread(path)
        img_CCM= ccmApply(img,CCM)
        img_CCM = np.clip(img_CCM, 0, 1)

        img_Gamma= Gamma(img_CCM)   

        img_Gamma = (img_Gamma * 255).astype(np.uint8)
        img_Gamma = np.clip(img_Gamma, 0, 255)

        imgName= basename+'_CCM.jpg'
        cv2.imwrite(imgName, cv2.cvtColor(img_Gamma, cv2.COLOR_RGB2BGR))

main()
# img_Tmp = cv2.imread(imgName)

# img_reversegamma= np.power(img_Tmp.astype(np.float32) / 255.0, 1/gamma)
# # 保存反 Gamma 处理后的结果
# imgName_reversegamma = "image_reversegamma.png"
# cv2.imwrite(imgName_reversegamma, (img_reversegamma * 255).astype(np.uint8))