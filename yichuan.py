import numpy as np
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# 设置中文字体和seaborn样式
mpl.rcParams['font.sans-serif'] = ['SimHei']  # Windows 默认有 SimHei
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set(style="whitegrid", font='SimHei')  # 使用seaborn的白色网格样式

# 1. 定义 Rastrigin 函数（DEAP 和 scipy 共用）
def rastrigin(x):
    x1, x2 = x[0], x[1]
    return 20 + (x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2))

# 2. 遗传算法实现（DEAP）
def run_ga():
    # 检查是否已经定义过，避免重复创建
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox() #注册工具箱
    toolbox.register("attr_float", np.random.uniform, -5.12, 5.12)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=50)

    toolbox.register("evaluate", lambda ind: (rastrigin(ind),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # 创建一个日志记录器来保存每一代的最佳适应度
    logbook = tools.Logbook()
    logbook.header = ['gen', 'min', 'avg']
    
    # 修改后的eaSimple算法，以便记录每一代的数据
    for gen in range(200):
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, pop)
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        pop = toolbox.select(pop, k=len(pop))

    return hof[0], rastrigin(hof[0]), logbook

# 3. 梯度优化实现（scipy.optimize.minimize）
def run_minimize(method='BFGS'):
    # 随机初始点（与 GA 的随机种群对比）
    x0 = np.random.uniform(-5.12, 5.12, 2)

    if method in ['BFGS', 'L-BFGS-B']:
        # 需要梯度（Rastrigin 函数的梯度）
        def gradient(x):
            x1, x2 = x[0], x[1]
            df_dx1 = 2 * x1 + 20 * np.pi * np.sin(2 * np.pi * x1)
            df_dx2 = 2 * x2 + 20 * np.pi * np.sin(2 * np.pi * x2)
            return np.array([df_dx1, df_dx2])
        result = minimize(rastrigin, x0, method=method, jac=gradient)
    else:
        # 无需梯度（如 Nelder-Mead）
        result = minimize(rastrigin, x0, method=method)

    return result.x, result.fun

# 4. 对比实验
def compare_algorithms(n_runs=10):
    ga_results = []
    bfgs_results = []
    nm_results = []
    ga_convergence = []  # 存储GA的收敛曲线数据

    for _ in range(n_runs):
        # 运行 GA
        ga_x, ga_val, logbook = run_ga()
        ga_results.append(ga_val)
        ga_convergence.append(logbook.select("min"))  # 记录每一代的最小值
        
        # 运行 BFGS（依赖梯度）
        bfgs_x, bfgs_val = run_minimize('BFGS')
        bfgs_results.append(bfgs_val)

        # 运行 Nelder-Mead（无需梯度）
        nm_x, nm_val = run_minimize('Nelder-Mead')
        nm_results.append(nm_val)

    # 统计结果
    print(f"\n=== 平均结果（{n_runs} 次运行）===")
    print(f"遗传算法 (GA): 平均 f(x) = {np.mean(ga_results):.6f}, 标准差 = {np.std(ga_results):.6f}")
    print(f"BFGS (梯度法): 平均 f(x) = {np.mean(bfgs_results):.6f}, 标准差 = {np.std(bfgs_results):.6f}")
    print(f"Nelder-Mead:  平均 f(x) = {np.mean(nm_results):.6f}, 标准差 = {np.std(nm_results):.6f}")

    # 准备数据用于seaborn绘图
    data = pd.DataFrame({
        'Algorithm': ['GA']*n_runs + ['BFGS']*n_runs + ['Nelder-Mead']*n_runs,
        'Value': ga_results + bfgs_results + nm_results
    })
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 第一个子图：箱线图比较
    plt.subplot(1, 2, 1)
    #x y 直接对应x轴和y轴坐标轴名称
    sns.boxplot(x='Algorithm', y='Value', data=data,hue='Algorithm', palette="Set2", legend=False)
    plt.title("Rastrigin 函数优化结果对比")
    plt.ylabel("f(x)")
    
   
    
    # 第二个子图：GA收敛曲线
    plt.subplot(1, 2, 2)
   
    for i, conv in enumerate(ga_convergence[:10]):  # 只显示前5次运行以避免过于拥挤
        sns.lineplot(x=range(len(conv)), y=conv, label=f'运行 {i+1}')
    plt.title("遗传算法收敛曲线")
    plt.xlabel("代数")
    plt.ylabel("最佳适应度")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_algorithms(n_runs=10)