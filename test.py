import numpy as np
import matplotlib.pyplot as plt
from colour.temperature.mccamy1992 import xy_to_CCT_McCamy1992
from colour.temperature.hernandez1999 import xy_to_CCT_Hernandez1999
from colour.temperature.ohno2013 import XYZ_to_CCT_Ohno2013
from colour.models import xy_to_XYZ

# -----------------------------
# 1. 随机生成 xy 数据
# -----------------------------
np.random.seed(42)  # 为了可重复
num_points = 100

# xy 坐标合理范围：x约[0.25,0.45], y约[0.25,0.45]
x_rand = np.random.uniform(0.25, 0.45, num_points)
y_rand = np.random.uniform(0.25, 0.45, num_points)
xy_list = np.stack([x_rand, y_rand], axis=1)

# -----------------------------
# 2. 用不同方法计算 CCT
# -----------------------------
CCT_mccamy = np.array([xy_to_CCT_McCamy1992(xy) for xy in xy_list])
CCT_hernandez = np.array([xy_to_CCT_Hernandez1999(xy) for xy in xy_list])
# xy -> XYZ -> Ohno2013 CCT
CCT_ohno = np.array([XYZ_to_CCT_Ohno2013(xy_to_XYZ(xy))[0] for xy in xy_list])


# -----------------------------
# 3. 打印部分对比
# -----------------------------
print("x       y       McCamy   Hernandez   Ohno")
for i in range(0, num_points, 5):
    print(f"{xy_list[i,0]:.3f}  {xy_list[i,1]:.3f}  "
          f"{CCT_mccamy[i]:8.2f}  {CCT_hernandez[i]:10.2f}  {CCT_ohno[i]:8.2f}")

# -----------------------------
# 4. 绘图比较
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(num_points), CCT_mccamy, 'o-', label="McCamy1992")
plt.plot(range(num_points), CCT_hernandez, 's-', label="Hernandez1999")
plt.plot(range(num_points), CCT_ohno, 'x-', label="Ohno2013")
plt.xlabel("Sample index")
plt.ylabel("Calculated CCT (K)")
plt.title("CCT Calculation Comparison on Random xy")
plt.legend()
plt.grid(True)
plt.show()


