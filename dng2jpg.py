import os
import rawpy
import imageio.v2 as imageio
from tqdm import tqdm

def dng_to_jpg(input_dir, output_dir):
    """
    将文件夹下所有 DNG 图片批量转换为 JPG。
    :param input_dir: 输入目录（包含 DNG 文件）
    :param output_dir: 输出目录（保存 JPG 文件）
    """
    os.makedirs(output_dir, exist_ok=True)
    dng_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.dng')]

    if not dng_files:
        print("❌ 未在输入目录中找到 DNG 文件")
        return

    for file in tqdm(dng_files, desc="转换中"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.jpg')

        try:
            with rawpy.imread(input_path) as raw:
                rgb = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    use_auto_wb=True,      # 使用相机白平衡
                    no_auto_bright=False,     # 关闭自动亮度调整
                    output_bps=8,            # 输出 8-bit JPG
                    output_color=rawpy.ColorSpace.sRGB,
                )
                imageio.imwrite(output_path, rgb)
        except Exception as e:
            print(f"⚠️ 转换失败：{file}，错误：{e}")

    print(f"\n✅ 转换完成！共 {len(dng_files)} 张，输出路径：{output_dir}")

if __name__ == '__main__':
    # 🟢 这里手动设置输入输出路径
    input_dir = r"C:\Users\15696\Desktop\new"       # 替换为你的 DNG 文件夹路径
    output_dir = r"C:\Users\15696\Desktop\new"     # 替换为输出 JPG 文件夹路径

    dng_to_jpg(input_dir, output_dir)