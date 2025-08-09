import numpy as np
from scipy.optimize import minimize
def npToString(arr):
    return np.array2string(arr, suppress_small=True, precision=4, floatmode='fixed')
class CCM_3x3:
    def __init__(self,input,output):
        self.input = input
        self.output = output
        self.ccm = np.ones((3, 3))  # 初始化为单位矩阵
        # 仅用shape进行判断
        if input.shape != output.shape:
            raise ValueError("input和output的形状必须相同")
        if input.shape[1] != 3:
            raise ValueError("最后一个维度必须是RGB颜色")
        self.m, self.n = input.shape[:2]  # 获取输入图像的形状
       
    def loss(self, x, input, output):
       
        ccm = x.reshape(3, 3)  # 将扁平化的参数恢复为3x3矩阵
        predicted = np.dot(input, ccm.T)  # 应用颜色校正
        error = np.mean((predicted - output)**2)  # MSE误差
        return error

    def infer_image(self):
        x = self.ccm.flatten()  # 初始猜测值
        C = np.zeros((3, 9))
        for i in range(3):
            C[i, 3*i : 3*i+3] = 1  # 每行对应矩阵M的一行的3个元素
        
        # 约束条件: CCM矩阵的每一行之和为1
        constraints = {
            'type': 'eq', 
            'fun': lambda x: C @ x - np.ones(3),
        } if len(C) > 0 else None
        bounds = [(-3, 3) for _ in range(9)]
        result = minimize(
            self.loss,  # 包装loss函数
            x,  
            args=(self.input, self.output),
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',#trust-constr SLSQP  L-BFGS-B
            options={'disp': True}
        )
        # 打印优化结果的详细信息
        print("\n=== Optimization Result ===")
        print(f"Success: {result.success}")  # 是否成功收敛
        print(f"Message: {result.message}")  # 状态描述
        print(f"Final loss value: {result.fun}")  # 最终损失值
        print(f"Iterations: {result.nit}")  # 迭代次数
        print(f"Function evaluations: {result.nfev}")  # 损失函数调用次数

        # 将优化结果恢复为CCM矩阵
        optimized_ccm = result.x.reshape(3, 3)

        return optimized_ccm
    

def main():
        
    # 示例
    A = np.array([[2, 2.5, 4],[3, 3.5, 5],[4, 5, 6],[5, 6.5, 8],[6, 8, 10],[7, 9, 12],[8, 10, 14],[9, 11, 16],[10, 12, 18]])
    B = np.array([[2, 4, 6],[3, 6, 9],[4, 8, 12],[5, 10, 15],[6, 12, 18],[7, 14, 21],[8, 16, 24],[9, 18, 27],[10, 20, 30]])

    ccmCalib= CCM_3x3(A, B)

    ccm= ccmCalib.infer_image()
    # print(f"颜色校正矩阵:\n{ccm}")
    
    print("===ccm===\n",npToString(ccm))
    print(f"===A@B的颜色校正矩阵===\n{npToString(ccm@A.T)}")

main()