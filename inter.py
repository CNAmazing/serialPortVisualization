import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 已知光源数据
prior_cct = np.array([2300, 2850, 3800, 4100, 5000, 6500, 7500])
prior_r = np.array([1/0.774472, 1/0.938632, 1/1.289859, 1/1.396722, 1/1.367716, 1/1.630072, 1/1.669945])
prior_b = np.array([1/2.190720, 1/2.002354, 1/1.718581, 1/1.799861, 1/1.444114, 1/1.283499, 1/1.227189])

# 生成插值函数（线性或更平滑的样条插值都可以）
interp_r = interp1d(prior_cct, prior_r, kind='linear')
interp_b = interp1d(prior_cct, prior_b, kind='linear')

# 生成每隔 50 K 的采样点
cct_new = np.arange(2300, 7500 + 50, 50)
r_new = interp_r(cct_new)
b_new = interp_b(cct_new)
print("double prior_cct[] = {", ", ".join(f"{v:.0f}" for v in cct_new), "};")
print("double prior_r[]   = {", ", ".join(f"{v:.6f}" for v in r_new), "};")
print("double prior_b[]   = {", ", ".join(f"{v:.6f}" for v in b_new), "};")
# 输出成表格或数组形式
data = pd.DataFrame({
    'CCT': cct_new,
    'R_over_G': r_new,
    'B_over_G': b_new
})

# 打印前 10 行预览
print(data)

# 如果你想导出到文件
# data.to_csv('awb_prior_interp.csv', index=False, float_format='%.6f')
