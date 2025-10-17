#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版模块化ISP流水线
实用且易于理解的实现
"""

import numpy as np
from typing import Dict, List, Optional, Any
import time
import cv2
class SimpleISPModule:
    """简化的ISP模块基类"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.params = {}
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """处理图像 - 子类需要重写此方法"""
        return image
    
    def set_param(self, key: str, value: Any):
        """设置参数"""
        self.params[key] = value
    
    def get_param(self, key: str, default: Any = None):
        """获取参数"""
        return self.params.get(key, default)

class BLCModule(SimpleISPModule):
    """黑电平校正模块"""
    
    def __init__(self, blc_value: int = 64):
        super().__init__("BLC")
        self.set_param("blc_value", blc_value)
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        blc_value = self.get_param("blc_value")
        result = image - blc_value
        return np.clip(result, 0, 1023).astype(np.float64)

class LSCModule(SimpleISPModule):
    """镜头阴影校正模块"""
    
    def __init__(self, gain_map: Optional[np.ndarray] = None):
        super().__init__("LSC")
        self.set_param("gain_map", gain_map)
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        gain_map = self.get_param("gain_map")
        if gain_map is None:
            return image
        return np.clip(image * gain_map, 0, 1023).astype(np.float64)

class AWBModule(SimpleISPModule):
    """白平衡模块"""
    
    def __init__(self, r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0):
        super().__init__("AWB")
        self.set_param("r_gain", r_gain)
        self.set_param("g_gain", g_gain)
        self.set_param("b_gain", b_gain)
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        r_gain = self.get_param("r_gain")
        g_gain = self.get_param("g_gain")
        b_gain = self.get_param("b_gain")
        
        result = image.copy()
        result[:, :, 0] *= b_gain
        result[:, :, 1] *= g_gain
        result[:, :, 2] *= r_gain
        return np.clip(result, 0, 1)

class DemosaicModule(SimpleISPModule):
    """去马赛克模块"""
    
    def __init__(self, bayer_pattern: str = "BGGR"):
        super().__init__("Demosaic")
        self.set_param("bayer_pattern", bayer_pattern)
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """简化的去马赛克实现"""
        bayer_pattern = self.get_param("bayer_pattern")
        match bayer_pattern:
            case "BGGR":
                rgb=cv2.cvtColor(image, cv2.COLOR_BAYER_BGGR2RGB)
            case "RGGB":
                rgb=cv2.cvtColor(image, cv2.COLOR_BAYER_RGGB2RGB)
            case "GBRG":
                rgb=cv2.cvtColor(image, cv2.COLOR_BAYER_GBRG2RGB)
            case "GRBG":
                rgb=cv2.cvtColor(image, cv2.COLOR_BAYER_GRBG2RGB)
            case _:
                raise ValueError(f"Unsupported bayer pattern: {bayer_pattern}")
        return rgb.astype(np.float32)

class GammaModule(SimpleISPModule):
    """Gamma校正模块"""
    
    def __init__(self, gamma: float = 2.2):
        super().__init__("Gamma")
        self.set_param("gamma", gamma)
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        gamma = self.get_param("gamma")
        gamma_exp = 1.0 / gamma
        
        mask = image <= 0.0031308
        result = np.where(mask, image * 12.92, 1.055 * (image ** gamma_exp) - 0.055)
        return np.clip(result, 0, 1)

class SimpleISPPipeline:
    """简化的ISP流水线"""
    
    def __init__(self, name: str = "Simple_ISP_Pipeline"):
        self.name = name
        self.modules: List[SimpleISPModule] = []
        self.processing_times = {}
    
    def add_module(self, module: SimpleISPModule):
        """添加模块"""
        self.modules.append(module)
        print(f"添加模块: {module.name}")
    
    def remove_module(self, module_name: str):
        """移除模块"""
        self.modules = [m for m in self.modules if m.name != module_name]
        print(f"移除模块: {module_name}")
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """处理图像"""
        result = image.copy()
        
        print(f"开始处理图像: {image.shape}")
        
        for module in self.modules:
            if not module.enabled:
                print(f"跳过模块: {module.name} (已禁用)")
                continue
            
            start_time = time.time()
            result = module.process(result, **kwargs)
            end_time = time.time()
            
            processing_time = end_time - start_time
            self.processing_times[module.name] = processing_time
            
            print(f"模块 {module.name} 处理完成: {processing_time:.4f}s")
        
        return result
    
    def get_module(self, module_name: str) -> Optional[SimpleISPModule]:
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
            status = "启用" if module.enabled else "禁用"
            print(f"{i+1:2d}. {module.name:15s} - {status}")
        print("=" * 40)
    
    def print_performance(self):
        """打印性能信息"""
        print(f"\n性能统计:")
        print("=" * 40)
        for module_name, time_taken in self.processing_times.items():
            print(f"{module_name:15s}: {time_taken:.4f}s")
        print("=" * 40)

def create_standard_pipeline() -> SimpleISPPipeline:
    """创建标准流水线"""
    pipeline = SimpleISPPipeline("标准ISP流水线")
    
    # 添加标准模块
    pipeline.add_module(BLCModule(blc_value=64))
    pipeline.add_module(LSCModule())
    pipeline.add_module(AWBModule(r_gain=1.2, g_gain=1.0, b_gain=0.8))
    pipeline.add_module(DemosaicModule(bayer_pattern="BGGR"))
    pipeline.add_module(GammaModule(gamma=2.2))
    
    return pipeline

def create_minimal_pipeline() -> SimpleISPPipeline:
    """创建最小流水线"""
    pipeline = SimpleISPPipeline("最小ISP流水线")
    
    # 只添加基本模块
    pipeline.add_module(BLCModule(blc_value=64))
    pipeline.add_module(DemosaicModule(bayer_pattern="BGGR"))
    pipeline.add_module(GammaModule(gamma=2.2))
    
    return pipeline

# 使用示例
if __name__ == "__main__":
    print("简化版模块化ISP流水线")
    print("=" * 50)
    
    # 创建流水线
    pipeline = create_standard_pipeline()
    pipeline.print_pipeline_info()
    
    # 创建测试图像
    test_image = np.random.randint(0, 1024, (100, 100, 3), dtype=np.uint16)
    print(f"\n测试图像: {test_image.shape}")
    
    # 处理图像
    result = pipeline.process(test_image)
    print(f"处理结果: {result.shape}")
    
    # 显示性能
    pipeline.print_performance()
    
    # 演示模块管理
    print("\n模块管理演示:")
    
    # 禁用某个模块
    gamma_module = pipeline.get_module("Gamma")
    if gamma_module:
        gamma_module.enabled = False
        print("禁用Gamma模块")
    
    # 修改模块参数
    awb_module = pipeline.get_module("AWB")
    if awb_module:
        awb_module.set_param("r_gain", 1.5)
        print("修改AWB模块参数")
    
    # 重新处理
    result2 = pipeline.process(test_image)
    pipeline.print_performance()
    
    print("\n简化版ISP流水线演示完成！")
