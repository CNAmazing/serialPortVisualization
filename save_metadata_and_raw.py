import struct
import os
import cv2
from glob import glob

class DngMeta:
    def __init__(self):
        self.ImageWidth = 0
        self.ImageHeight = 0
        self.BitsPerSample = 0
        self.DateTime = ''
        self.StripOffsets = 0
        self.StripByteCounts = 0
        self.CFAPattern = b''
        self.ExposureTime = [0, 0]
        self.ISO = 0
        self.BlackLevel = 0
        self.ColorMatrix1 = []
        self.AsShotNeutral = []
        self.ColorTemperatureK = 0
        self.BitsPerSampleReal = 0
        self.ADRCGain = 0.0


def parse_dng_metadata(data: bytes, meta: DngMeta) -> int:
    if len(data) < 8:
        return -1

    print(f"DNG file size: {len(data)} bytes")

    # 检查字节序和 magic number
    byte_order, magic = struct.unpack('<HH', data[0:4])
    if byte_order != 0x4949 or magic != 0x002A:
        print("Invalid DNG header")
        return -1

    # 获取第一个 IFD 偏移
    ifd_offset = struct.unpack('<I', data[4:8])[0]
    if ifd_offset >= len(data):
        print("Invalid IFD offset")
        return -1

    # 解析 IFD 条目数量
    ifd = data[ifd_offset:]
    num_entries = struct.unpack('<H', ifd[:2])[0]
    # print(f"Number of entries: {num_entries}")
    ifd = ifd[2:]

    # 遍历所有条目
    for i in range(num_entries):
        entry_data = ifd[i * 12:(i + 1) * 12]
        tag, typ, count, value = struct.unpack('<HHII', entry_data)

        # 计算项长度
        type_sizes = {1:1, 2:1, 3:2, 4:4, 5:8, 11:4}
        item_size = type_sizes.get(typ, 0)
        total_len = item_size * count

        if total_len > 4 and value + total_len <= len(data):
            ptr = data[value:value + total_len]
        elif total_len <= 4:
            ptr = entry_data[8:12]
        else:
            ptr = None

        # 解析 tag 项
        if tag == 256:
            meta.ImageWidth = value
        elif tag == 257:
            meta.ImageHeight = value
        elif tag == 258:
            meta.BitsPerSample = value & 0xFFFF
        elif tag == 306 and ptr:
            meta.DateTime = ptr.rstrip(b'\x00').decode(errors='ignore')
        elif tag == 273:
            meta.StripOffsets = value
        elif tag == 279:
            meta.StripByteCounts = value
        elif tag == 33422 and typ == 1 and count == 4 and ptr:
            meta.CFAPattern = ptr
        elif tag == 33434 and typ == 5 and count == 1 and ptr:
            meta.ExposureTime[0] = struct.unpack('<I', ptr[0:4])[0]
            meta.ExposureTime[1] = struct.unpack('<I', ptr[4:8])[0]
        elif tag == 34855:
            meta.ISO = value & 0xFFFF
        elif tag == 50714:
            meta.BlackLevel = value & 0xFFFF
        elif tag == 50721 and typ == 11 and count == 9 and ptr:
            meta.ColorMatrix1 = list(struct.unpack('<9f', ptr))
        elif tag == 50728 and typ == 11 and count == 3 and ptr:
            meta.AsShotNeutral = list(struct.unpack('<3f', ptr))
        elif tag == 50944:
            meta.ColorTemperatureK = value
        elif tag == 50945:
            meta.BitsPerSampleReal = value
        elif tag == 50946:
            meta.ADRCGain = value/1000.0

    return 0


def save_raw_image_data(data: bytes, meta: DngMeta, output_file: str):
    """保存原始图像数据到文件"""
    offset = meta.StripOffsets
    count = meta.StripByteCounts

    if offset + count > len(data):
        print("Invalid raw data range")
        return

    raw_data = data[offset:offset + count]
    with open(output_file, 'wb') as f:
        f.write(raw_data)
    print(f"Raw image data saved to {output_file}")

def print_dng_metadata(meta: DngMeta):
    """打印 DNG 文件中的所有元数据字段"""
    print("======= DNG 元数据 =======")
    print(f"图像宽度: {meta.ImageWidth}")
    print(f"图像高度: {meta.ImageHeight}")
    print(f"每像素位数: {meta.BitsPerSampleReal}")
    print(f"拍摄时间: {meta.DateTime}")
    print(f"CFA图案 (R:0, G:1, B:2): {list(meta.CFAPattern)}")
    print(f"曝光时间 (单位：秒): {meta.ExposureTime[0]}/{meta.ExposureTime[1]}")
    print(f"AGain: {meta.ISO/100}")
    print(f"黑电平: {meta.BlackLevel}")
    print(f"颜色矩阵1: {meta.ColorMatrix1}")
    print(f"白平衡系数: {meta.AsShotNeutral}")
    print(f"色温: {meta.ColorTemperatureK} K")
    print(f"ADRC Gain: {meta.ADRCGain}")
    print("===========================")

# if __name__ == "__main__":
#     #修改位深度
#     name = "a2AbAfTpRZ_vel_raw_1753256024523"
#     dng_file = f"./data/0723_tmp2/{name}.dng"
#     raw_file = f"./data/0723_tmp2/{name}.raw"
#     # meta_file = f"./data/{name}.json"
    
#     # 加载 DNG 文件
#     with open(dng_file, 'rb') as f:
#         dng_data = f.read()

#     # 创建元数据结构体
#     meta = DngMeta()

#     # 解析 DNG 元数据
#     if parse_dng_metadata(dng_data, meta) == 0:
#         print("解析成功")
#         print_dng_metadata(meta)  # 打印所有元数据
        
#         # 保存原始图像数据
#         # save_raw_image_data(dng_data, meta, raw_file)
#     else:
#         print("解析失败")

def draw_metadata_on_image(image, meta):
    text_lines = [
        f"Resolution: {meta.ImageWidth}x{meta.ImageHeight}",
        f"Exposure: {meta.ExposureTime[0]}/{meta.ExposureTime[1]} s",
        f"Gain (ISO/100): {meta.ISO / 100:.2f}",
        f"Color Temp: {meta.ColorTemperatureK} K",
        f"ADRC Gain: {meta.ADRCGain:.2f}"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0  # 原来是0.7，放大到3.0
    color = (0, 255, 0)
    thickness = 6     # 原来是2，加粗为6

    line_height = int(80 * font_scale)  # 行间距调整
    x_offset = 100
    y_start = 200

    for i, line in enumerate(text_lines):
        y = y_start + i * line_height
        cv2.putText(image, line, (x_offset, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return image

if __name__ == "__main__":
    folder = r"C:\serialPortVisualization\data\0813_5"
    dng_files = glob(os.path.join(folder, "*.dng"))

    for dng_path in dng_files:
        name = os.path.splitext(os.path.basename(dng_path))[0]
        png_path = os.path.join(folder, f"{name}.png")

        if not os.path.exists(png_path):
            print(f"[跳过] PNG 不存在: {png_path}")
            continue

        with open(dng_path, 'rb') as f:
            dng_data = f.read()

        meta = DngMeta()
        if parse_dng_metadata(dng_data, meta) == 0:
            print(f"[成功] 解析: {name}")
            print_dng_metadata(meta)

            img = cv2.imread(png_path)
            if img is not None:
                img_annotated = draw_metadata_on_image(img, meta)
                out_path = os.path.join(folder, f"{name}_annotated.png")
                cv2.imwrite(out_path, img_annotated)
                print(f"[保存] 注释图像保存至: {out_path}")
            else:
                print(f"[错误] 无法读取 PNG 图像: {png_path}")
        else:
            print(f"[失败] 解析 DNG: {name}")