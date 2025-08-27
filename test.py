
def readRaw(path,w,h):#path为raw文件所在路径
    type = 'uint16' #得到数据格式，如uint8和uint16等
    imgData = np.fromfile(path, dtype=type)
    imgData = imgData.reshape(-1, w, h)

    return imgData
