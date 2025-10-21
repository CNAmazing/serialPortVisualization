
import numpy as np
import pwlf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

class piecewise_Line:
    def __init__(self,data,k) -> None:
        self.data=data
        self.k=k
        self.res=[]
    def curvefit(self):
        my_pwlf = pwlf.PiecewiseLinFit(self.data[:,0], self.data[:,1])
        my_pwlf.fit(self.k)
        break_x=my_pwlf.fit_breaks
        break_y=my_pwlf.predict(break_x)

        for x,y in zip(break_x,break_y):
            self.res.append([x,y])
        self.res=np.array(self.res)
        print(self.res)
        return self.res
    def show(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[:,0], self.data[:,1], color='red', s=50, alpha=0.7, label='原始数据')
        plt.plot(self.res[:,0], self.res[:,1], 'b-', linewidth=2, marker='o', markersize=6, label='分段线性拟合')
        plt.xlabel('X值')
        plt.ylabel('Y值')
        plt.title('分段线性拟合可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()




# def piecewise_Line(data,k):
#     my_pwlf = pwlf.PiecewiseLinFit(data[:,0], data[:,1])
#     my_pwlf.fit(k)
#     break_x=my_pwlf.fit_breaks
#     break_y=my_pwlf.predict(break_x)
#     res=[]
#     for x,y in zip(break_x,break_y):
#         res.append([x,y])
#     res=np.array(res)
#     return res

# data=np.array([[4393,68],
#                    [2173,63],
#                    [4668,68],
#                    [1781,61],
#                    [929,62],
#                    [3363,70],
#                    [4020,68],
#                    [1400,60],
#                    [10737,82],
#                    [500,56],
#                    [0.1,52],

#     ])

# pl=piecewise_Line(data,5)
# pl.curvefit()
# pl.show()

# 可视化
