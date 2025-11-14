import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Microsoft YaHei'

def read_video_and_display_histogram(video_path, frame_interval=1):
    """
    读取视频并显示图像和直方图
    
    参数:
        video_path: 视频文件路径
        frame_interval: 帧间隔，每隔多少帧显示一次（默认1，即每帧都显示）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息：")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  显示间隔: 每 {frame_interval} 帧显示一次")
    
    frame_count = 0
    displayed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"\n视频读取完成，共显示 {displayed_count} 帧")
                break
            
            # 根据帧间隔决定是否显示
            if frame_count % frame_interval == 0:
                displayed_count += 1
                
                # 将BGR转换为RGB（matplotlib使用RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为灰度图用于计算亮度直方图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 计算5区间直方图和MSV
                # 将0-255分为5个区间：[0-50], [51-101], [102-152], [153-203], [204-255]
                interval_size = 51
                num_intervals = 5
                hist_5_intervals = np.zeros(num_intervals)
                
                # 计算每个区间的像素数量
                for i in range(num_intervals):
                    start_val = i * interval_size
                    if i == num_intervals - 1:  # 最后一个区间包含到255
                        end_val = 255
                        mask = (gray >= start_val) & (gray <= end_val)
                    else:
                        end_val = (i + 1) * interval_size - 1
                        mask = (gray >= start_val) & (gray <= end_val)
                    hist_5_intervals[i] = np.sum(mask)
                
                # 计算MSV (Mean of Scaled Values)
                total_pixels = gray.size
                msv = np.sum(hist_5_intervals * np.arange(1, num_intervals + 1)) / total_pixels
                
                # 创建图形，包含三个子图：图像、完整亮度直方图、5区间直方图
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                fig.suptitle(f'视频帧分析 - 帧号: {frame_count}/{total_frames}', fontsize=14, fontweight='bold')
                
                # 子图1：显示原始图像
                axes[0].imshow(frame_rgb)
                axes[0].set_title(f'原始图像 (帧 {frame_count})', fontsize=12)
                axes[0].axis('off')
                
                # 子图2：完整亮度直方图
                axes[1].set_title('亮度直方图', fontsize=12)
                hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
                axes[1].plot(hist_gray, color='black', linewidth=1.5)
                axes[1].fill_between(range(256), hist_gray.flatten(), alpha=0.3, color='gray')
                axes[1].set_xlabel('像素值')
                axes[1].set_ylabel('频数')
                axes[1].grid(True, alpha=0.3)
                
                # 子图3：5区间直方图
                axes[2].set_title('5区间亮度直方图', fontsize=12)
                bars = axes[2].bar(range(1, num_intervals + 1), hist_5_intervals, 
                                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'], 
                                  alpha=0.7, edgecolor='black', linewidth=1.5)
                axes[2].set_xlabel('区间编号')
                axes[2].set_ylabel('像素数量')
                axes[2].set_xticks(range(1, num_intervals + 1))
                axes[2].set_xticklabels([f'{i+1}' for i in range(num_intervals)])
                axes[2].grid(True, alpha=0.3, axis='y')
                
                # 在柱状图上显示数值
                for i, (bar, count) in enumerate(zip(bars, hist_5_intervals)):
                    height = bar.get_height()
                    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(count)}\n({count/total_pixels*100:.1f}%)',
                                ha='center', va='bottom', fontsize=9)
                
                # 添加区间范围说明
                axes[2].text(0.5, -0.15, '\n'.join([f'区间{i+1}: [{i*51}-{((i+1)*51-1) if i<4 else 255}]' 
                                                    for i in range(num_intervals)]),
                            transform=axes[2].transAxes, ha='center', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
                
                # 添加统计信息文本（包含MSV）
                stats_text = (f'帧号: {frame_count}/{total_frames}\n'
                            f'分辨率: {width}x{height}\n'
                            f'平均亮度: {np.mean(gray):.2f}\n'
                            f'标准差: {np.std(gray):.2f}\n'
                            f'MSV: {msv:.4f}')
                axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes,
                              fontsize=9, verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                # 使用阻塞模式：关闭窗口后自动继续下一帧
                plt.show()
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        cap.release()
        print("视频资源已释放")


def save_video_frames_as_images(video_path, output_dir=None, frame_interval=1):
    """
    将视频帧保存为图片文件
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录（默认在视频同目录下创建frames文件夹）
        frame_interval: 帧间隔，每隔多少帧保存一次
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 设置输出目录
    if output_dir is None:
        video_path_obj = Path(video_path)
        output_dir = video_path_obj.parent / 'frames'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0
    
    print(f"开始保存视频帧到: {output_dir}")
    print(f"总帧数: {total_frames}, 保存间隔: 每 {frame_interval} 帧")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                output_path = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"已保存 {saved_count} 帧...")
            
            frame_count += 1
        
        print(f"\n完成！共保存 {saved_count} 帧到 {output_dir}")
        
    except Exception as e:
        print(f"保存过程中出错: {e}")
    finally:
        cap.release()


if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 默认视频路径，可以修改为你的视频路径
        video_path = input("请输入视频文件路径: ").strip().strip('"').strip("'")
    
    if not Path(video_path).exists():
        print(f"错误：视频文件不存在: {video_path}")
        sys.exit(1)
    
    # 选择功能
    print("\n请选择功能：")
    print("1. 显示视频帧和直方图（交互式）")
    print("2. 保存视频帧为图片文件")
    
    choice = input("请输入选项 (1 或 2，默认1): ").strip() or "1"
    
    if choice == "1":
        # 询问帧间隔
        interval_input = input("请输入帧间隔（默认1，即每帧都显示）: ").strip()
        frame_interval = int(interval_input) if interval_input else 1
        read_video_and_display_histogram(video_path, frame_interval)
    elif choice == "2":
        # 询问输出目录和帧间隔
        output_dir_input = input("请输入输出目录（留空使用默认）: ").strip()
        output_dir = output_dir_input if output_dir_input else None
        
        interval_input = input("请输入帧间隔（默认1，即每帧都保存）: ").strip()
        frame_interval = int(interval_input) if interval_input else 1
        
        save_video_frames_as_images(video_path, output_dir, frame_interval)
    else:
        print("无效选项")
