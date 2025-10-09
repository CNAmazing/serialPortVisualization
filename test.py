import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def f(x):
    """目标二次函数"""
    return x**3

def segment_max_error(x_points, i):
    """计算第i个线段的最大误差"""
    x_left, x_right = x_points[i], x_points[i+1]
    y_left, y_right = f(x_left), f(x_right)
    
    # 定义线性插值函数
    def line_func(x):
        slope = (y_right - y_left) / (x_right - x_left)
        return y_left + slope * (x - x_left)
    
    # 定义误差函数
    def error_func(x):
        return np.abs(f(x) - line_func(x))
    
    # 在当前线段区间内寻找最大误差
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(error_func, bounds=(x_left, x_right), method='bounded')
    return -res.fun  # 返回最大误差值

def objective_function(x_internal, x_start=1, x_end=9):
    """
    目标函数：最小化所有线段中的最大误差
    x_internal: 需要优化的内部点（决策变量）
    """
    # 构建完整的分段点（包括固定的起点和终点）
    x_points = np.concatenate([[x_start], x_internal, [x_end]])
    n_segments = len(x_points) - 1
    
    # 计算每个线段的最大误差
    max_errors = [segment_max_error(x_points, i) for i in range(n_segments)]
    
    # 返回所有线段中的最大误差（这是我们要最小化的目标）
    return max(max_errors)

# 使用scipy.optimize.minimize进行优化
def optimize_with_scipy(n_segments, x_start=1, x_end=9):
    """
    使用scipy的minimize函数进行优化
    n_segments: 线段数量
    返回: 优化后的分段点
    """
    n_internal_points = n_segments - 1  # 需要优化的内部点数量
    
    # 初始猜测：均匀分布的内部点
    x0 = np.linspace(x_start, x_end, n_segments + 1)[1:-1]
     
    # 定义约束：内部点必须按顺序排列且在区间内
    bounds = [(x_start, x_end)] * n_internal_points
    
    # 使用优化算法（SLSQP或trust-constr适合有约束的问题）
    result = minimize(
        objective_function, 
        x0, 
        bounds=bounds,
        method='SLSQP',
        options={'disp': True, 'ftol': 1e-8}
    )
    if result.success:
        optimized_internal = result.x
        optimized_points = np.concatenate([[x_start], optimized_internal, [x_end]])
        min_error = result.fun
        return optimized_points, min_error
    else:
        raise ValueError("优化失败: " + result.message)

# 使用示例
n_segments = 4  # 3段线
x_start, x_end = 1, 9

optimized_points, min_error = optimize_with_scipy(n_segments, x_start, x_end)

print(f"优化后的分段点: {optimized_points}")
print(f"最小最大误差: {min_error:.8f}")

# 可视化最终结果
plt.figure(figsize=(10, 6))
x_plot = np.linspace(x_start, x_end, 1000)
y_func = f(x_plot)

plt.plot(x_plot, y_func, 'b-', label='f(x) = x²', linewidth=2)

# 绘制拟合的多段线
for i in range(len(optimized_points) - 1):
    x_seg = [optimized_points[i], optimized_points[i+1]]
    y_seg = [f(optimized_points[i]), f(optimized_points[i+1])]
    plt.plot(x_seg, y_seg, 'r-', linewidth=2, label='拟合线段' if i == 0 else "")
    plt.plot(x_seg, y_seg, 'ro', markersize=6)

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'二次函数的多段线拟合 (最大误差: {min_error:.6f})')
plt.legend()
plt.grid(True)
plt.show()