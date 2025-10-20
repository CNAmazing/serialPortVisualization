import numpy as np
import time

def test_astype_performance():
    """测试astype的性能损耗"""
    print("=== 数据类型转换性能测试 ===\n")
    
    # 创建测试数据
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        print(f"数据大小: {size:,} 个元素")
        
        # 创建uint8数据
        data_uint8 = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # 测试1: 直接astype转换
        start_time = time.time()
        for _ in range(100):  # 重复100次
            result1 = data_uint8.astype(np.float32)
        time1 = time.time() - start_time
        
        # 测试2: 先转换为float64再转换为float32
        start_time = time.time()
        for _ in range(100):
            result2 = data_uint8.astype(np.float64).astype(np.float32)
        time2 = time.time() - start_time
        
        # 测试3: 使用view方法（如果可能）
        start_time = time.time()
        for _ in range(100):
            result3 = data_uint8.astype(np.float32)
        time3 = time.time() - start_time
        
        print(f"  直接astype: {time1:.4f}秒")
        print(f"  双重转换: {time2:.4f}秒")
        print(f"  重复astype: {time3:.4f}秒")
        print(f"  内存使用: {result1.nbytes / 1024:.2f} KB")
        print()

def test_memory_usage():
    """测试内存使用情况"""
    print("=== 内存使用测试 ===\n")
    
    # 创建不同大小的数据
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"数据大小: {size:,} 个元素")
        
        # uint8数据
        data_uint8 = np.random.randint(0, 256, size, dtype=np.uint8)
        print(f"  uint8内存: {data_uint8.nbytes / 1024:.2f} KB")
        
        # float32数据
        data_float32 = data_uint8.astype(np.float32)
        print(f"  float32内存: {data_float32.nbytes / 1024:.2f} KB")
        
        # 内存增长倍数
        growth = data_float32.nbytes / data_uint8.nbytes
        print(f"  内存增长: {growth:.1f}倍")
        print()

def test_alternatives():
    """测试替代方案"""
    print("=== 替代方案测试 ===\n")
    
    size = 100000
    data_uint8 = np.random.randint(0, 256, size, dtype=np.uint8)
    
    # 方案1: 直接astype
    start_time = time.time()
    result1 = data_uint8.astype(np.float32)
    time1 = time.time() - start_time
    
    # 方案2: 使用np.array
    start_time = time.time()
    result2 = np.array(data_uint8, dtype=np.float32)
    time2 = time.time() - start_time
    
    # 方案3: 使用np.asarray
    start_time = time.time()
    result3 = np.asarray(data_uint8, dtype=np.float32)
    time3 = time.time() - start_time
    
    print(f"astype方法: {time1:.6f}秒")
    print(f"np.array方法: {time2:.6f}秒")
    print(f"np.asarray方法: {time3:.6f}秒")
    
    # 验证结果一致性
    print(f"结果一致性: {np.allclose(result1, result2) and np.allclose(result2, result3)}")
    print()

def test_isp_pipeline_impact():
    """测试在ISP流水线中的影响"""
    print("=== ISP流水线性能影响测试 ===\n")
    
    # 模拟ISP流水线中的数据类型转换
    size = 1000000  # 1M像素
    raw_data = np.random.randint(0, 1024, size, dtype=np.uint16)
    
    # 测试多次转换的性能
    start_time = time.time()
    for _ in range(10):  # 模拟10次处理
        # 模拟ISP流水线中的转换
        data1 = raw_data.astype(np.float32)
        data2 = data1 * 1.5  # 模拟处理
        data3 = data2.astype(np.uint8)
    time_total = time.time() - start_time
    
    print(f"1M像素数据，10次处理循环:")
    print(f"  总时间: {time_total:.4f}秒")
    print(f"  平均每次: {time_total/10:.4f}秒")
    print(f"  每秒处理: {size*10/time_total/1000000:.2f}M像素")
    print()

if __name__ == "__main__":
    test_astype_performance()
    test_memory_usage()
    test_alternatives()
    test_isp_pipeline_impact()
    
    print("=== 性能优化建议 ===")
    print("1. 避免不必要的类型转换")
    print("2. 在流水线中保持数据类型一致性")
    print("3. 考虑使用in-place操作")
    print("4. 对于大图像，考虑分块处理")
    print("5. 使用适当的数据类型（避免过度精度）")

