import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# 设置中文字体和seaborn样式
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Windows 默认有 SimHei
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set(style="whitegrid", font='SimHei')  # 使用seaborn的白色网格样式

# 1. 定义问题
def generate_random_matrices(n=100):
    """生成随机的输入矩阵1和目标矩阵2"""
    np.random.seed(42)
    matrix1 = np.random.rand(n, 3) * 10  # n×3的输入矩阵
    # 定义一个真实的3×3变换矩阵用于生成目标矩阵
    true_transform = np.array([[1.2, 0.5, -0.3],
                               [-0.7, 1.8, 0.2],
                               [0.1, -0.4, 2.1]])
    matrix2 = np.dot(matrix1, true_transform)  # n×3的目标矩阵
    return matrix1, matrix2

# 生成示例数据
matrix1, matrix2 = generate_random_matrices(100)

# 2. 定义评估函数（计算均方根误差之和）
def evaluate_transform(individual):
    """评估3×3变换矩阵的适应度"""
    transform_matrix = np.array(individual).reshape(3, 3)
    predicted = np.dot(matrix1, transform_matrix)
    # 计算每一行的RMSE然后求和
    row_errors = np.sqrt(np.mean((predicted - matrix2)**2, axis=1))
    total_error = np.sum(row_errors)
    return total_error,

# 3. 遗传算法实现
def run_ga(matrix1, matrix2, n_pop=50, n_gen=100, cxpb=0.7, mutpb=0.2):
    # 创建类型（如果尚未创建）
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    # 3×3矩阵有9个元素，每个元素在[-5,5]范围内
    toolbox.register("attr_float", np.random.uniform, -5, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=9)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=n_pop)

    toolbox.register("evaluate", evaluate_transform)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'min', 'avg']
    
    for gen in range(n_gen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = toolbox.map(toolbox.evaluate, pop)
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        pop = toolbox.select(pop, k=len(pop))

    best_matrix = np.array(hof[0]).reshape(3, 3)
    best_error = evaluate_transform(hof[0])[0]
    return best_matrix, best_error, logbook

# 4. 运行遗传算法并可视化结果
def optimize_and_visualize():
    # 运行遗传算法
    best_matrix, best_error, logbook = run_ga(matrix1, matrix2)
    
    print("找到的最佳变换矩阵:")
    print(best_matrix)
    print(f"最小化的总RMSE: {best_error:.6f}")
    
    # 可视化收敛曲线
    plt.figure(figsize=(10, 6))
    gen = logbook.select("gen")
    min_fit = logbook.select("min")
    avg_fit = logbook.select("avg")
    
    plt.plot(gen, min_fit, 'b-', label="最佳适应度")
    plt.plot(gen, avg_fit, 'r-', label="平均适应度")
    plt.xlabel("代数")
    plt.ylabel("总RMSE")
    plt.title("遗传算法优化过程")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 可视化预测结果示例
    predicted = np.dot(matrix1, best_matrix)
    sample_indices = np.random.choice(len(matrix1), 10, replace=False)
    
    plt.figure(figsize=(12, 6))
    for i in range(3):  # 对每个维度
        plt.subplot(1, 3, i+1)
        plt.scatter(matrix1[sample_indices, i], matrix2[sample_indices, i], 
                    c='b', label="真实值", alpha=0.6)
        plt.scatter(matrix1[sample_indices, i], predicted[sample_indices, i], 
                    c='r', marker='x', label="预测值", alpha=0.6)
        plt.xlabel(f"输入矩阵1 维度{i+1}")
        plt.ylabel(f"矩阵2 维度{i+1}")
        plt.title(f"维度{i+1}的预测效果")
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    return best_matrix

if __name__ == "__main__":
    best_transform = optimize_and_visualize()