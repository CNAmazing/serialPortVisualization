#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import Dict

@dataclass
class FrameMeta:
    width: int = 0
    height: int = 0
    raw_bit: int = 10
    bayer_pattern: str = "RGGB"
    blc_value: int = 64
    color_temperature: float = 6500
    lsc_gain: Dict[int, Dict[str, np.ndarray]] = None
    wb_gain: Dict[int, Dict[str, float]] = None
    ccm_matrix: Dict[int, np.ndarray] = None 
    gamma: float = 2.2

@dataclass
class FrameData:
    image: np.ndarray
    meta: FrameMeta

def analyze_frame_data_overhead():
    """分析FrameData传递的开销"""
    print("=== FrameData传递开销分析 ===\n")
    
    # 创建测试数据
    height, width = 1944, 2592
    test_image = np.random.rand(height, width, 3).astype(np.float32)
    
    # 创建元数据
    meta = FrameMeta(
        width=width,
        height=height,
        lsc_gain={3000: {"R": np.ones((10, 10)), "Gr": np.ones((10, 10)), 
                  "Gb": np.ones((10, 10)), "B": np.ones((10, 10))}},
        wb_gain={3000: {"r": 1.0, "g": 1.0, "b": 1.0}},
        ccm_matrix={3000: np.eye(3)}
    )
    
    # 测试1: FrameData对象创建开销
    print("1. FrameData对象创建开销:")
    start_time = time.perf_counter()
    for _ in range(1000):
        frame_data = FrameData(image=test_image, meta=meta)
    creation_time = time.perf_counter() - start_time
    print(f"   创建1000个FrameData对象: {creation_time*1000:.2f}ms")
    print(f"   平均每个对象: {creation_time:.6f}ms")
    
    # 测试2: 内存使用分析
    print("\n2. 内存使用分析:")
    frame_data = FrameData(image=test_image, meta=meta)
    
    # 图像数据内存
    image_memory = frame_data.image.nbytes / 1024 / 1024  # MB
    print(f"   图像数据: {image_memory:.2f} MB")
    
    # 元数据内存估算
    meta_memory = sys.getsizeof(meta) / 1024  # KB
    print(f"   元数据: {meta_memory:.2f} KB")
    
    # 总内存
    total_memory = image_memory + meta_memory / 1024
    print(f"   总内存: {total_memory:.2f} MB")
    
    # 测试3: 对象传递开销
    print("\n3. 对象传递开销:")
    
    def dummy_process(frame_data: FrameData) -> FrameData:
        """模拟处理函数"""
        return FrameData(image=frame_data.image, meta=frame_data.meta)
    
    start_time = time.perf_counter()
    for _ in range(1000):
        result = dummy_process(frame_data)
    pass_time = time.perf_counter() - start_time
    print(f"   传递1000次: {pass_time*1000:.2f}ms")
    print(f"   平均每次传递: {pass_time:.6f}ms")

def analyze_optimization_strategies():
    """分析优化策略"""
    print("\n=== 优化策略分析 ===\n")
    
    # 策略1: 原地修改 vs 创建新对象
    print("1. 原地修改 vs 创建新对象:")
    
    # 创建测试数据
    test_image = np.random.rand(100, 100, 3).astype(np.float32)
    meta = FrameMeta()
    frame_data = FrameData(image=test_image, meta=meta)
    
    # 原地修改
    start_time = time.perf_counter()
    for _ in range(1000):
        frame_data.image *= 1.1  # 原地修改
    inplace_time = time.perf_counter() - start_time
    
    # 创建新对象
    start_time = time.perf_counter()
    for _ in range(1000):
        new_image = frame_data.image * 1.1
        new_frame = FrameData(image=new_image, meta=meta)
    newobj_time = time.perf_counter() - start_time
    
    print(f"   原地修改: {inplace_time*1000:.2f}ms")
    print(f"   创建新对象: {newobj_time*1000:.2f}ms")
    print(f"   性能提升: {newobj_time/inplace_time:.1f}倍")
    
    # 策略2: 元数据共享
    print("\n2. 元数据共享策略:")
    
    # 不共享元数据
    start_time = time.perf_counter()
    for _ in range(1000):
        new_meta = FrameMeta(width=100, height=100)
        new_frame = FrameData(image=test_image, meta=new_meta)
    no_share_time = time.perf_counter() - start_time
    
    # 共享元数据
    shared_meta = FrameMeta(width=100, height=100)
    start_time = time.perf_counter()
    for _ in range(1000):
        new_frame = FrameData(image=test_image, meta=shared_meta)
    share_time = time.perf_counter() - start_time
    
    print(f"   不共享元数据: {no_share_time*1000:.2f}ms")
    print(f"   共享元数据: {share_time*1000:.2f}ms")
    print(f"   性能提升: {no_share_time/share_time:.1f}倍")

def suggest_optimizations():
    """建议优化方案"""
    print("\n=== 优化建议 ===\n")
    
    print("1. 原地修改策略:")
    print("   - 直接修改frame_data.image，避免创建新对象")
    print("   - 减少内存分配和垃圾回收开销")
    print("   - 适用于大多数ISP模块")
    
    print("\n2. 元数据优化:")
    print("   - 使用引用传递，避免元数据复制")
    print("   - 将大型配置数据移到模块外部")
    print("   - 使用缓存机制减少重复计算")
    
    print("\n3. 内存管理:")
    print("   - 预分配图像缓冲区")
    print("   - 使用内存池减少分配开销")
    print("   - 考虑使用Cython或Numba加速")
    
    print("\n4. 架构优化:")
    print("   - 使用生成器模式减少中间对象")
    print("   - 实现流式处理避免全量数据传递")
    print("   - 考虑使用共享内存或零拷贝技术")

if __name__ == "__main__":
    analyze_frame_data_overhead()
    analyze_optimization_strategies()
    suggest_optimizations()
