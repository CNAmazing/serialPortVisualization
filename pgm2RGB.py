import cv2
import numpy as np
from tools import *
# 读取 Bayer PGM 文件


def RGGB2RGG(path):

    bayer_pgm = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # 常见选项：COLOR_BAYER_BG2RGB, COLOR_BAYER_RG2RGB, COLOR_BAYER_GB2RGB 等
    rgb = cv2.cvtColor(bayer_pgm, cv2.COLOR_BAYER_RGGB2RGB)
    return rgb

def folderProcessing(folderPath):
    print_file_tree(folderPath)
    full_paths, basenames = get_paths(folderPath, suffix=".pgm")
    for path, basename in zip(full_paths, basenames):
        rgb=RGGB2RGG(path)
        cv2.imwrite(f"{basename}_RGB.jpg", rgb)

def main():

    folderPath = r"C:\serialPortVisualization\data\0813_2"
    folderProcessing(folderPath)


main()