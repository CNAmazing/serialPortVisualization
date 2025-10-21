#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from typing import Dict, List, Optional, Any
import time
import cv2
from dataclasses import dataclass

@dataclass
class FrameMeta:
    width: int = 0
    height: int = 0
    raw_bit: int = 10
    bayer_pattern: str = "RGGB"
    wb_gain: Dict[str, float] = None
    ccm_matrix: Any = None
    gamma: float = 2.2
    blc_value: int = 64
    lsc_gain: Dict[str, np.ndarray] = None
    
   

@dataclass
class FrameData:
    """
    图像帧数据，包含图像本身和元信息
    """
    image: np.ndarray
    meta: FrameMeta

class ISPModule:
    """简化的ISP模块基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        """处理图像 - 子类需要重写此方法"""
        return frame_data
class ReadRawModule(ISPModule):
    """raw图读取模块"""
    
    def __init__(self):
        super().__init__("READRAW")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        imgData = np.fromfile(kwargs.get('image_path', ''), dtype=np.uint16)
        imgData = imgData.reshape(meta.height, meta.width)
        result = np.clip(imgData, 0, meta.raw_bit).astype(np.float32)
        
        return FrameData(image=result, meta=meta)

class BLCModule(ISPModule):
    """黑电平校正模块"""
    
    def __init__(self):
        super().__init__("BLC")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        # 从 FrameMeta 直接获取黑电平参数
        blc_value = meta.blc_value
        result = frame_data.image - blc_value
        result = np.clip(result, 0, meta.raw_bit)
        
        return FrameData(image=result, meta=meta)

class LSCRawModule(ISPModule):
    """镜头阴影校正模块"""
    
    def __init__(self):
        super().__init__("LSCRAW")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 从 FrameMeta 直接获取增益图
        gain_map = meta.lsc_gain
        if gain_map is None:
            return frame_data
            
        rows, cols = image.shape[:2]
        gain_R = gain_map['R']
        gain_Gr = gain_map['Gr']
        gain_Gb = gain_map['Gb']
        gain_B = gain_map['B']
        m = len(gain_R) - 1
        n = len(gain_R[0]) - 1
        bayer_mask = self.generate_bayer_mask(rows, cols, pattern=meta.bayer_pattern)
        
        # 块边界
        block_heights = np.linspace(0, rows, m+1, dtype=int)
        block_widths = np.linspace(0, cols, n+1, dtype=int)

        y_coords, x_coords = np.indices((rows, cols))
        i_indices = np.searchsorted(block_heights, y_coords, side='right') - 1
        j_indices = np.searchsorted(block_widths, x_coords, side='right') - 1
        i_indices = np.clip(i_indices, 0, m-1)
        j_indices = np.clip(j_indices, 0, n-1)

        # 归一化坐标（避免除零）
        h_diff = (block_heights[i_indices+1] - block_heights[i_indices]).astype(float)
        w_diff = (block_widths[j_indices+1] - block_widths[j_indices]).astype(float)
        h_diff[h_diff == 0] = 1
        w_diff[w_diff == 0] = 1
        y_norm = (y_coords - block_heights[i_indices]) / h_diff
        x_norm = (x_coords - block_widths[j_indices]) / w_diff

        # 输出增益图
        gains = np.zeros_like(image, dtype=float)

        # 遍历通道 (整数编码)
        for color_id, mesh in [(0, gain_R), (1, gain_Gr), (2, gain_Gb), (3, gain_B)]:
            mask = (bayer_mask == color_id)
            if not np.any(mask):
                continue
            q11 = mesh[i_indices[mask], j_indices[mask]]
            q21 = mesh[i_indices[mask], j_indices[mask]+1]
            q12 = mesh[i_indices[mask]+1, j_indices[mask]]
            q22 = mesh[i_indices[mask]+1, j_indices[mask]+1]
            gains[mask] = ((1 - y_norm[mask]) * (1 - x_norm[mask]) * q11 +
                            (1 - y_norm[mask]) * x_norm[mask] * q21 +
                            y_norm[mask] * (1 - x_norm[mask]) * q12 +
                            y_norm[mask] * x_norm[mask] * q22)

        result = np.clip(image * gains, 0, meta.raw_bit)
        return FrameData(image=result, meta=meta)
    def generate_bayer_mask(self,rows, cols, pattern='RGGB'):
        """生成整数编码 Bayer mask"""
        bayer_mask = np.zeros((rows, cols), dtype=np.uint8)
        pattern_map = {
            'RGGB': np.array([[0, 1], [2, 3]], dtype=np.uint8),
            'BGGR': np.array([[3, 2], [1, 0]], dtype=np.uint8),
            'GBRG': np.array([[2, 3], [0, 1]], dtype=np.uint8),
            'GRBG': np.array([[1, 0], [3, 2]], dtype=np.uint8),
        }
        if pattern not in pattern_map:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")
        tile = pattern_map[pattern]
        bayer_mask[::2, ::2] = tile[0,0]
        bayer_mask[::2, 1::2] = tile[0,1]
        bayer_mask[1::2, ::2] = tile[1,0]
        bayer_mask[1::2, 1::2] = tile[1,1]
        return bayer_mask
class AWBRgbModule(ISPModule):
    """白平衡模块"""
    
    def __init__(self):
        super().__init__("AWBRGB")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image.copy()
        
        # 从 FrameData 获取白平衡参数
        r_gain = meta.wb_gain["r"]
        g_gain = meta.wb_gain["g"]
        b_gain = meta.wb_gain["b"]
        
        image[:, :, 0] *= b_gain
        image[:, :, 1] *= g_gain
        image[:, :, 2] *= r_gain
        
        result = np.clip(image, 0, 1)
        return FrameData(image=result, meta=meta)
class AWBRawModule(ISPModule):
    """白平衡模块"""
    
    def __init__(self):
        super().__init__("AWBRAW")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image.copy()
        
        # 从 FrameData 获取白平衡参数
        r_gain = meta.wb_gain["r"]
        g_gain = meta.wb_gain["g"]
        b_gain = meta.wb_gain["b"]
        
        match meta.bayer_pattern:
            case "BGGR":
                image[1::2, 1::2] *= b_gain
                image[::2, 1::2] *= g_gain
                image[1::2, ::2] *= g_gain
                image[::2, ::2] *= r_gain
            case "RGGB":
                image[::2, ::2] *= r_gain
                image[1::2, ::2] *= g_gain
                image[::2, 1::2] *= g_gain
                image[1::2, 1::2] *= b_gain
            case "GBRG":
                image[::2, ::2] *= g_gain
                image[1::2, ::2] *= b_gain
                image[::2, 1::2] *= r_gain
                image[1::2, 1::2] *= g_gain
            case "GRBG":
                image[1::2, ::2] *= g_gain
                image[::2, ::2] *= r_gain
                image[1::2, 1::2] *= b_gain
                image[::2, 1::2] *= g_gain
            case _:
                raise ValueError(f"Unsupported bayer pattern: {meta.bayer_pattern}")
    
        result = np.clip(image, 0, meta.raw_bit)
        return FrameData(image=result, meta=meta)
class DemosaicModule(ISPModule):
    """去马赛克模块"""
    
    def __init__(self):
        super().__init__("Demosaic")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        match meta.bayer_pattern:
            case "BGGR":
                rgb = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BAYER_BGGR2RGB)
            case "RGGB":
                rgb = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BAYER_RGGB2RGB)
            case "GBRG":
                rgb = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BAYER_GBRG2RGB)
            case "GRBG":
                rgb = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BAYER_GRBG2RGB)
            case _:
                raise ValueError(f"Unsupported bayer pattern: {meta.bayer_pattern}")
        
        result = rgb.astype(np.float32)
        return FrameData(image=result, meta=meta)

class CCMModule(ISPModule):
    """颜色校正矩阵模块"""
    
    def __init__(self):
        super().__init__("CCM")

    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 从 FrameData 获取 CCM 矩阵
        ccm_matrix = meta.ccm_matrix
        if ccm_matrix is None:
            return frame_data
            
        h, w = image.shape[:2]  # H*W*3
        result = image.reshape(-1, 3)  # (h*w, 3)
        np.dot(result, ccm_matrix.T, out=result)  # 等价于 (ccm @ rgb_flat.T).T
        result = result.reshape(h, w, 3)
        result = np.clip(result, 0, 1)
        
        return FrameData(image=result, meta=meta)
    
class GammaModule(ISPModule):
    """Gamma校正模块"""
    
    def __init__(self):
        super().__init__("Gamma")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 从 FrameMeta 直接获取 gamma 参数
        gamma = meta.gamma
        gamma_exp = 1.0 / gamma
        
        mask = image <= 0.0031308
        result = np.where(mask, image * 12.92, 1.055 * (image ** gamma_exp) - 0.055)
        result = np.clip(result, 0, 1)
        
        return FrameData(image=result, meta=meta)

class SimpleISPPipeline:
    """简化的ISP流水线"""
    
    def __init__(self, name: str = "Simple_ISP_Pipeline"):
        self.name = name
        self.modules: List[ISPModule] = []
        self.processing_times = {}
    
    def add_module(self, module: ISPModule):
        """添加模块"""
        self.modules.append(module)
        print(f"add module: {module.name}")
    
    def remove_module(self, module_name: str):
        """移除模块"""
        self.modules = [m for m in self.modules if m.name != module_name]
        print(f"remove module: {module_name}")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        """处理图像"""
        result = frame_data
        
        print(f"开始处理图像: {frame_data.image.shape}")
        
        for module in self.modules:
            start_time = time.time()
            result = module.process(result, **kwargs)
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.processing_times[module.name] = processing_time
            
            print(f"模块 {module.name} 处理完成: {processing_time:.4f}s")
        
        return result
    
    def get_module(self, module_name: str) -> Optional[ISPModule]:
        """获取模块"""
        for module in self.modules:
            if module.name == module_name:
                return module
        return None
    
    def print_pipeline_info(self):
        """打印流水线信息"""
        print(f"\n流水线: {self.name}")
        print("=" * 40)
        for i, module in enumerate(self.modules):
            print(f"{i+1:2d}. {module.name:15s}")
        print("=" * 40)
    
    def print_performance(self):
        """打印性能信息"""
        print(f"\n性能统计:")
        print("=" * 40)
        for module_name, time_taken in self.processing_times.items():
            print(f"{module_name:15s}: {time_taken:.4f}s")
        print("=" * 40)

def create_raw_pipeline() -> SimpleISPPipeline:
    """创建标准流水线"""
    pipeline = SimpleISPPipeline("raw pipeline")
    
    # 添加标准模块
    # pipeline.add_module(ReadRawModule())
    pipeline.add_module(BLCModule())
    pipeline.add_module(LSCRawModule())
    pipeline.add_module(AWBRawModule())
    pipeline.add_module(DemosaicModule())
    pipeline.add_module(CCMModule())
    pipeline.add_module(GammaModule())
    
    return pipeline
    
# 使用示例
if __name__ == "__main__":
    print("简化版模块化ISP流水线")
    print("=" * 50)
    
    # 创建流水线
    pipeline = create_raw_pipeline()
    pipeline.print_pipeline_info()
    
    # 创建测试图像和元数据
    test_image = np.random.randint(0, 1024, (1944, 2592), dtype=np.int16)
    test_image = test_image.astype(np.float32)  
    meta = FrameMeta(
        width=100,
        height=100,
        raw_bit=10,
        bayer_pattern="RGGB",
        wb_gain={"r": 1.2, "g": 1.0, "b": 0.8},
        gamma=2.2,
        blc_value=64,
        lsc_gain=None  # 可以设置为实际的增益图
    )
    frame_data = FrameData(image=test_image, meta=meta)
    
    print(f"\n测试图像: {frame_data.image.shape}")
    
    # 处理图像
    result = pipeline.process(frame_data)
    print(f"处理结果: {result.image.shape}")
    
    # 显示性能
    pipeline.print_performance()
    
    # 演示模块管理
    print("\n模块管理演示:")
    
    # 修改 FrameData 中的参数
    frame_data.meta.wb_gain["r"] = 1.5
    print("修改白平衡参数")
    
    # 重新处理
    result2 = pipeline.process(frame_data)
    pipeline.print_performance()
    
    print("\n简化版ISP流水线演示完成！")
