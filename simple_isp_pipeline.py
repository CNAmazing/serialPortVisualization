#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from typing import Dict, List, Optional, Any
import time
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt

"""
    isppipeline
    =========raw_bit==========||================0-1======================
    BLC   LSC   AWB   Demosaic||            CCM   Gamma   PP
"""
@dataclass
class FrameMeta:
    width: int = 0
    height: int = 0
    raw_bit: int = 10
    bayer_pattern: str = "RGGB"
    blc_value: int = 64
    color_temperature: float = 6500
    lsc_gain: Dict[int, Dict[str, np.ndarray]] = None
    wb_gain:Dict[int,Dict[str,float]]=None
    ccm_matrix: Dict[int, np.ndarray] = None 
    gamma: float = 2.2

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
        imgData = np.fromfile(kwargs.get('image_path', ''), dtype='uint16')
        imgData = imgData.reshape(meta.height, meta.width)
        result = np.clip(imgData, 0, 2**meta.raw_bit-1).astype(np.float32)
        
        return result

class BLCModule(ISPModule):
    """黑电平校正模块"""
    
    def __init__(self):
        super().__init__("BLC")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        # 从 FrameMeta 直接获取黑电平参数
        blc_value = meta.blc_value
        frame_data.image -= blc_value  # 原地修改
        np.clip(frame_data.image, 0, 2**meta.raw_bit-1, out=frame_data.image)
        
        return frame_data  # 原地修改，返回原对象

class LSCRawModule(ISPModule):
    """镜头阴影校正模块"""
    
    def __init__(self):
        super().__init__("LSCRAW")
    
    def interpolate_lsc_gain(self, lsc_gain_config: Dict, target_temp: float):
        """
        根据色温插值计算LSC增益图
        
        Args:
            lsc_gain_config: LSC增益配置字典
            target_temp: 目标色温
        
        Returns:
            插值后的增益图字典
        """
        if lsc_gain_config is None or len(lsc_gain_config) < 2:
            return None
        
        # 获取所有色温点（假设已按升序排列）
        temps = list(lsc_gain_config.keys())
        
        # 边界处理
        if target_temp <= temps[0]:
            return lsc_gain_config[temps[0]]
        elif target_temp >= temps[-1]:
            return lsc_gain_config[temps[-1]]
        
        # 找到目标色温两侧的色温点
        for i in range(len(temps) - 1):
            if temps[i] <= target_temp <= temps[i + 1]:
                # 计算插值权重 lambda
                lambda_weight = (target_temp - temps[i]) / (temps[i + 1] - temps[i])
                
                # 插值计算
                result = {}
                gain1 = lsc_gain_config[temps[i]]
                gain2 = lsc_gain_config[temps[i + 1]]
                
                for channel in ['r', 'gr', 'gb', 'b']:
                    if channel in gain1 and channel in gain2:
                        result[channel] = gain1[channel] + lambda_weight * (gain2[channel] - gain1[channel])
                    else:
                        result[channel] = gain1.get(channel, np.ones((3, 3)))
                
                return result
        
        return lsc_gain_config[temps[0]]
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 使用类内插值方法获取增益图
        gain_map = self.interpolate_lsc_gain(meta.lsc_gain, meta.color_temperature)
        if gain_map is None:
            return frame_data
            
        rows, cols = image.shape[:2]
        gain_R = gain_map['r']
        gain_Gr = gain_map['gr']
        gain_Gb = gain_map['gb']
        gain_B = gain_map['b']
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

        # 预分配增益图，避免重复分配
        gains = np.ones_like(image, dtype=np.float32)

        # 优化的双线性插值 - 向量化操作
        for color_id, mesh in [(0, gain_R), (1, gain_Gr), (2, gain_Gb), (3, gain_B)]:
            mask = (bayer_mask == color_id)
            if not np.any(mask):
                continue
            
            # 预计算所有插值系数
            i_masked = i_indices[mask]
            j_masked = j_indices[mask]
            y_norm_masked = y_norm[mask]
            x_norm_masked = x_norm[mask]
            
            # 向量化双线性插值
            q11 = mesh[i_masked, j_masked]
            q21 = mesh[i_masked, j_masked + 1]
            q12 = mesh[i_masked + 1, j_masked]
            q22 = mesh[i_masked + 1, j_masked + 1]
            
            # 使用更高效的插值公式
            gains[mask] = (q11 + 
                          (q21 - q11) * x_norm_masked + 
                          (q12 - q11) * y_norm_masked + 
                          (q22 - q21 - q12 + q11) * x_norm_masked * y_norm_masked)

        # 原地操作，减少内存分配
        image *= gains
        np.clip(image, 0, 2**meta.raw_bit-1, out=image)
        return frame_data  # 原地修改，返回原对象
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
    
    def interpolate_wb_gain(self, wb_gain_config: Dict, target_temp: float):
        """
        根据色温插值计算白平衡增益
        
        Args:
            wb_gain_config: 白平衡增益配置字典
            target_temp: 目标色温
        
        Returns:
            插值后的白平衡增益字典
        """
        if wb_gain_config is None or len(wb_gain_config) < 2:
            return None
        
        # 获取所有色温点（假设已按升序排列）
        temps = list(wb_gain_config.keys())
        
        # 边界处理
        if target_temp <= temps[0]:
            return wb_gain_config[temps[0]]
        elif target_temp >= temps[-1]:
            return wb_gain_config[temps[-1]]
        
        # 找到目标色温两侧的色温点
        for i in range(len(temps) - 1):
            if temps[i] <= target_temp <= temps[i + 1]:
                # 计算插值权重 lambda
                lambda_weight = (target_temp - temps[i]) / (temps[i + 1] - temps[i])
                
                # 插值计算
                result = {}
                wb1 = wb_gain_config[temps[i]]
                wb2 = wb_gain_config[temps[i + 1]]
                
                for channel in ['r', 'g', 'b']:
                    if channel in wb1 and channel in wb2:
                        result[channel] = wb1[channel] + lambda_weight * (wb2[channel] - wb1[channel])
                    else:
                        result[channel] = wb1.get(channel, 1.0)
                
                return result
        
        return wb_gain_config[temps[0]]
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 使用类内插值方法获取白平衡参数
        wb_params = self.interpolate_wb_gain(meta.wb_gain, meta.color_temperature)
        if wb_params is None:
            return frame_data
            
        r_gain = wb_params["r"]
        g_gain = wb_params["g"]
        b_gain = wb_params["b"]
        
        frame_data.image[:, :, 0] *= b_gain  # 原地修改
        frame_data.image[:, :, 1] *= g_gain
        frame_data.image[:, :, 2] *= r_gain
        
        np.clip(frame_data.image, 0, 1, out=frame_data.image)
        return frame_data  # 原地修改，返回原对象
class AWBRawModule(ISPModule):
    """白平衡模块"""
    
    def __init__(self):
        super().__init__("AWBRAW")
    
    def interpolate_wb_gain(self, wb_gain_config: Dict, target_temp: float):
        """
        根据色温插值计算白平衡增益
        
        Args:
            wb_gain_config: 白平衡增益配置字典
            target_temp: 目标色温
        
        Returns:
            插值后的白平衡增益字典
        """
        if wb_gain_config is None or len(wb_gain_config) < 2:
            return None
        
        # 获取所有色温点（假设已按升序排列）
        temps = list(wb_gain_config.keys())
        
        # 边界处理
        if target_temp <= temps[0]:
            return wb_gain_config[temps[0]]
        elif target_temp >= temps[-1]:
            return wb_gain_config[temps[-1]]
        
        # 找到目标色温两侧的色温点
        for i in range(len(temps) - 1):
            if temps[i] <= target_temp <= temps[i + 1]:
                # 计算插值权重 lambda
                lambda_weight = (target_temp - temps[i]) / (temps[i + 1] - temps[i])
                
                # 插值计算
                result = {}
                wb1 = wb_gain_config[temps[i]]
                wb2 = wb_gain_config[temps[i + 1]]
                
                for channel in ['r', 'g', 'b']:
                    if channel in wb1 and channel in wb2:
                        result[channel] = wb1[channel] + lambda_weight * (wb2[channel] - wb1[channel])
                    else:
                        result[channel] = wb1.get(channel, 1.0)
                
                return result
        
        return wb_gain_config[temps[0]]
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 使用类内插值方法获取白平衡参数
        wb_params = self.interpolate_wb_gain(meta.wb_gain, meta.color_temperature)
        if wb_params is None:
            return frame_data
            
        r_gain = wb_params["r"]
        g_gain = wb_params["g"]
        b_gain = wb_params["b"]
        
        match meta.bayer_pattern:
            case "BGGR":
                frame_data.image[1::2, 1::2] *= b_gain  # 原地修改
                frame_data.image[::2, 1::2] *= g_gain
                frame_data.image[1::2, ::2] *= g_gain
                frame_data.image[::2, ::2] *= r_gain
            case "RGGB":
                frame_data.image[::2, ::2] *= r_gain
                frame_data.image[1::2, ::2] *= g_gain
                frame_data.image[::2, 1::2] *= g_gain
                frame_data.image[1::2, 1::2] *= b_gain
            case "GBRG":
                frame_data.image[::2, ::2] *= g_gain
                frame_data.image[1::2, ::2] *= b_gain
                frame_data.image[::2, 1::2] *= r_gain
                frame_data.image[1::2, 1::2] *= g_gain
            case "GRBG":
                frame_data.image[1::2, ::2] *= g_gain
                frame_data.image[::2, ::2] *= r_gain
                frame_data.image[1::2, 1::2] *= b_gain
                frame_data.image[::2, 1::2] *= g_gain
            case _:
                raise ValueError(f"Unsupported bayer pattern: {meta.bayer_pattern}")
    
        np.clip(frame_data.image, 0, 2**meta.raw_bit-1, out=frame_data.image)
        return frame_data  # 原地修改，返回原对象
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
        
        frame_data.image = rgb.astype(np.float32)  # 原地修改
        frame_data.image /= 2**meta.raw_bit-1
        np.clip(frame_data.image, 0, 1, out=frame_data.image)
        return frame_data  # 原地修改，返回原对象

class CCMModule(ISPModule):
    """颜色校正矩阵模块"""
    
    def __init__(self):
        super().__init__("CCM")
    
    def interpolate_ccm_matrix(self, ccm_config: Dict, target_temp: float):
        """
        根据色温插值计算CCM矩阵
        
        Args:
            ccm_config: CCM矩阵配置字典（键为数字）
            target_temp: 目标色温
        
        Returns:
            插值后的CCM矩阵
        """
        if not ccm_config:
            return None
        
        # 获取所有色温点（假设已按升序排列）
        temps = list(ccm_config.keys())
        
        if len(temps) == 1:
            return ccm_config[temps[0]]
        
        # 边界处理
        if target_temp <= temps[0]:
            return ccm_config[temps[0]]
        if target_temp >= temps[-1]:
            return ccm_config[temps[-1]]
        
        # 线性插值
        for i in range(len(temps) - 1):
            if temps[i] <= target_temp <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                ccm1, ccm2 = ccm_config[t1], ccm_config[t2]
                
                # 计算插值权重
                weight = (target_temp - t1) / (t2 - t1)
                
                # 矩阵插值
                ccm_interp = ccm1 + weight * (ccm2 - ccm1)
                return ccm_interp.astype(np.float32)
        
        return ccm_config[temps[0]]

    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        image = frame_data.image
        
        # 使用类内插值方法获取 CCM 矩阵
        ccm_matrix = self.interpolate_ccm_matrix(meta.ccm_matrix, meta.color_temperature)
        if ccm_matrix is None:
            # 使用单位矩阵作为默认值（不进行颜色校正）
            ccm_matrix = np.eye(3, dtype=np.float32)
        else:
            # 确保矩阵是 float32 类型
            ccm_matrix = ccm_matrix.astype(np.float32)
            
        h, w = frame_data.image.shape[:2]  # H*W*3
        result = frame_data.image.reshape(-1, 3)  # (h*w, 3)
        np.dot(result, ccm_matrix.T, out=result)  # 等价于 (ccm @ rgb_flat.T).T
        frame_data.image = result.reshape(h, w, 3)  # 原地修改
        np.clip(frame_data.image, 0, 1, out=frame_data.image)
        
        return frame_data  # 原地修改，返回原对象
    
class GammaModule(ISPModule):
    """Gamma校正模块"""
    
    def __init__(self):
        super().__init__("Gamma")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        meta = frame_data.meta
        
        # 从 FrameMeta 直接获取 gamma 参数
        gamma = meta.gamma
        gamma_exp = 1.0 / gamma
        
        mask = frame_data.image <= 0.0031308
        frame_data.image = np.where(mask, frame_data.image * 12.92, 1.055 * (frame_data.image ** gamma_exp) - 0.055)  # 原地修改
        np.clip(frame_data.image, 0, 1, out=frame_data.image)
        
        return frame_data  # 原地修改，返回原对象
class PPModule(ISPModule):
    """postProcess校正模块"""
    def __init__(self):
        super().__init__("PP")
    
    def process(self, frame_data: FrameData, **kwargs) -> FrameData:
        frame_data.image[...,:]=frame_data.image[...,::-1]
        frame_data.image*=255
        frame_data.image=frame_data.image.astype(np.uint8)
        
        np.clip(frame_data.image, 0, 255, out=frame_data.image)
        
        return frame_data  # 原地修改，返回原对象

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
        print("=" * 40)
        print(f"start processing: {frame_data.image.shape}")
        
        for module in self.modules:
            start_time = time.time()
            frame_data = module.process(frame_data, **kwargs)
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.processing_times[module.name] = processing_time
            
            print(f"{module.name:15s} : {processing_time:.4f}s")
        print(f"totalTime: {sum(self.processing_times.values()):.4f}s")
        print("=" * 40)
        
        return frame_data
    
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
    pipeline.add_module(BLCModule())
    pipeline.add_module(LSCRawModule())
    pipeline.add_module(AWBRawModule())
    pipeline.add_module(DemosaicModule())
    pipeline.add_module(CCMModule())
    pipeline.add_module(GammaModule())
    pipeline.add_module(PPModule())
    return pipeline
    
# 使用示例
if __name__ == "__main__":
    print("简化版模块化ISP流水线")
    print("=" * 50)
    
    # 创建流水线
    pipeline = create_raw_pipeline()
    pipeline.print_pipeline_info()
    
    
    # 创建色温相关的参数配置（按升序排列）
    # LSC 增益图配置（按色温升序）
    lsc_gain_config = {
        3000: {  # 暖光
            "r": np.ones((3, 3)) * 1.1,
            "gr": np.ones((3, 3)) * 1.0,
            "gb": np.ones((3, 3)) * 1.0,
            "b": np.ones((3, 3)) * 0.9
        },
        6500: {  # 日光
            "r": np.ones((3, 3)) * 1.0,
            "gr": np.ones((3, 3)) * 1.0,
            "gb": np.ones((3, 3)) * 1.0,
            "b": np.ones((3, 3)) * 1.0
        },
        10000: {  # 冷光
            "r": np.ones((3, 3)) * 0.9,
            "gr": np.ones((3, 3)) * 1.0,
            "gb": np.ones((3, 3)) * 1.0,
            "b": np.ones((3, 3)) * 1.1
        }
    }
    
    # 白平衡增益配置（按色温升序）
    wb_gain_config = {
        3000: {"r": 1.3, "g": 1.0, "b": 0.7},
        6500: {"r": 1.0, "g": 1.0, "b": 1.0},
        10000: {"r": 0.7, "g": 1.0, "b": 1.3}
    }
    
    # CCM 矩阵配置（按色温升序）
    ccm_config = {
        3000: np.array([[1.2, -0.1, -0.1], [-0.1, 1.1, 0.0], [-0.1, 0.0, 1.1]], dtype=np.float32),
        6500: np.eye(3, dtype=np.float32),
        10000: np.array([[1.1, 0.0, -0.1], [0.0, 1.0, 0.0], [-0.1, 0.0, 1.2]], dtype=np.float32)
    }
    
    meta = FrameMeta(
        width=2592,
        height=1944,
        raw_bit=10,
        bayer_pattern="RGGB",
        color_temperature=5000,  # 目标色温，会进行插值
        lsc_gain=lsc_gain_config,
        wb_gain=wb_gain_config,
        ccm_matrix=ccm_config,
        gamma=2.2,
        blc_value=64
    )

    readModule=ReadRawModule()
    frame_data = FrameData(image=None, meta=meta)

    # 创建测试图像和元数据
    image_path=r'C:\WorkSpace\serialPortVisualization\data\g07s5ColorChecker\A.raw'
    image= readModule.process(frame_data,image_path=image_path)
    print(image.shape)
    print(image.min(),image.max())
    
    frame_data = FrameData(image=image, meta=meta)
    
    print(f"\n测试图像: {frame_data.image.shape}")
    
    # 处理图像
    result = pipeline.process(frame_data)
    plt.figure()
    plt.imshow(result.image)
    plt.show()
    
    
  