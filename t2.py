import numpy as np

# 原始 7 组 CCM
ccm_list = [
    np.array([[164710760, 846800, -65557563],
              [-50787070, 217759418, -66972360],
              [-90459144, -88924750, 279383900]], dtype=np.float64),

    np.array([[195636082, -65563446, -30072638],
              [-42741406, 188462245, -45720840],
              [-55076260, -148995435, 304071700]], dtype=np.float64),

    np.array([[200623965, -100617993, -5970],
              [-31185531, 173692667, -42507132],
              [-16869554, -131106353, 247975900]], dtype=np.float64),

    np.array([[273247337, -184326279, 11078944],
              [-43417355, 182759368, -39342013],
              [-19811375, -128423400, 248234773]], dtype=np.float64),

    np.array([[193467712, -69837100, -23630613],
              [-36941240, 192514515, -55573270],
              [-18077955, -84569180, 202647138]], dtype=np.float64),

    np.array([[195305026, -72459620, -22845410],
              [-31694360, 189678454, -57984090],
              [-14665438, -78874474, 193539917]], dtype=np.float64),

    np.array([[210706758, -89659095, -21047656],
              [-31355175, 187258625, -55903447],
              [-12858337, -77057980, 189916313]], dtype=np.float64),
]

# 饱和度增强函数
def add_saturation_to_ccm(ccm: np.ndarray, saturation: float) -> np.ndarray:
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    W1 = np.tile(w, (3,1))
    sat_matrix = (1 - saturation) * W1 + saturation * np.eye(3)
    return np.dot(ccm, sat_matrix)

# 增加饱和度 1.8
new_ccm_list = [np.round(add_saturation_to_ccm(ccm, 1.8)).astype(int) for ccm in ccm_list]

# 输出
for i, ccm in enumerate(new_ccm_list):
    print(f"CCM {i+1}:")
    print(ccm)
    print()
