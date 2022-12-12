# 基于粒子滤波的锂离子电池RUL预测
# 电气工程及其自动化专业综合设计
# From: SWUST IPC14 DaiXingRong
# Email:1594508421@qq.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# 除去数据异常的较大值和较小值点
def drop_outlier(array, count, bins):
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

# 剩余容量双指数方程，状态方程
def hfun(X, k):
    Q = X[0] * np.exp(X[1] * k) + X[2] * np.exp(X[3] * k)
    return Q

# 重采样步骤：
# 预测：抽取新粒子
# 更新：更新粒子权值
# 状态估计
# 多项式重采样

# 重采样
def randomR(inIndex, q):
    outIndex = np.zeros(np.shape(inIndex))
    num = np.shape(q)
    u = np.random.rand(num[0], 1)
    u = np.sort(u, axis = 0)
    u = np.array(u)
    l = np.cumsum(q, axis=0)
    i = 0
    for j in np.arange(0, num[0]):
        while (i <= (num[0]-1)) and (u[i] <= l[j]):
            outIndex[i] = j
            i = i + 1
    return outIndex

# capacity：阶段放电容量
# resistance：放电阶段电池的的内阻平均值
# CCCT：恒定电流充电时间
# CVCT: 恒定电压充电时间
if __name__ == "__main__":

    # ========================================
    #                加载数据
    # ========================================
    Battery_name = 'CS2_38'
    Battery = np.load('dataset/' + Battery_name + '.npy', allow_pickle=True)
    Battery = Battery.item()
    battery = Battery[Battery_name]

    names = ['capacity', 'resistance', 'CCCT', 'CVCT']
    # battery[1] = battery[1] /(battery[1] + 1.1)


    # 初始化
    # 更新粒子状态
    # 权值计算和归一化
    # 重采样
    # 判断程序是否结束，迭代

    # ========================================
    #           预估数据为电池容量
    # ========================================
    N = len(battery['cycle'])
    pf_Number = 200                # 粒子数
    Prediction_Interval = 300      # 未来趋势值,预测区间的大小
    cita = 1e-4
    wa = 0.000001                  # 设定状态初始值
    wb = 0.01
    wc = 0.1
    wd = 0.0001
    Q = cita * np.diag([wa, wb, wc, wd])  # Q为过程噪声方差矩阵，diag()创建指定对角矩阵

    F = np.eye(4)   # F为驱动计算矩阵，eye()单位对角矩阵
    R = 0.001       # 观测噪声的协方差
    # ========= 状态方程赋初值 ==============
    a = -0.0000083499
    b = 0.055237
    c = 0.90097
    d = -0.00088543
    X0 = np.mat([a, b, c, d]).T  # 矩阵转置

    # ========= 滤波器状态初始化 ==============
    Xpf = np.zeros([4, N])
    Xpf[:, 0:1] = X0  # 对齐数组

    # ========= 粒子集初始化 ==============
    Xm = np.zeros([4, pf_Number, N])
    for i in np.arange(0, pf_Number - 1):
        sqr1 = np.array(sqrtm(Q))
        sqr2 = sqr1.dot(np.random.randn(4, 1))  # 矩阵乘法，直接用*是矩阵点乘
        Xm[:, i:i + 1, 0] = X0 + sqr2  # 对齐数组,需要将矩阵对齐后才能相加

    # ========= 从数据集读取观测量 =============
    capacity = battery[names[0]]
    Z = np.array(capacity)
    # ========= Zm为滤波器预测观测值，Zm与Xm对应 =============
    Zm = np.zeros([1, pf_Number, N])
    # ========= Zpf与Xpf对应 =============
    Zpf = np.zeros([1, N])         # 计算中得到的Zpf为滤波器更新得到的容量值
    # ========= 权值初始化 =============
    Weight = np.zeros([N, pf_Number])    # 计算中得到的W为更新的粒子权重

    # 粒子滤波算法
    for k in np.arange(1, N - 1):
        # 重要性采样
        for i in np.arange(0, pf_Number - 1):
            sqr1 = np.array(sqrtm(Q))       # 观测噪声
            sqr2 = sqr1.dot(np.random.randn(4, 1))  # 矩阵乘法，直接用*是矩阵点乘
            Xm[:, i:i + 1, k] = F.dot(Xm[:, i:i + 1, k - 1]) + sqr2
        # 权值重要性计算
        for i in np.arange(0, pf_Number - 1):
            Zm[0, i:i + 1, k] = hfun(Xm[:, i:i + 1, k], k)      # 观测预测
            Weight[k, i] = np.exp(-(Z[k] - Zm[0, i:i + 1, k:k + 1]) ** 2 / 2 / R) + 1e-99  # 重要性权值计算，乘方用 **
        Weight[k, :] = Weight[k, :] / sum(Weight[k, :])    # 权值归一化
        # 重采样
        # 这里的重采样以权值为传入值，返回值为采样后的索引
        outlndex = randomR(np.arange(0, pf_Number), Weight[k, :])
        # 得到新的样本
        for i in np.arange(0, len(outlndex)):
            Xm[:, i, k] = Xm[:, int(outlndex[i]), k]
        # 滤波后的状态更新，更新参数[a,b,c,d]
        Xpf[:, k] = [np.mean(Xm[0, :, k]),
                     np.mean(Xm[1, :, k]),
                     np.mean(Xm[2, :, k]),
                     np.mean(Xm[3, :, k])]
        # 更新后的状态计算预测的容量值
        Zpf[0, k] = hfun(Xpf[:, k], k)

    # ========================================
    #         计算自然条件下的预测值
    # ========================================
    start = N - Prediction_Interval    # 预测的区间
    Zf = np.zeros(Prediction_Interval)  # 自然预测值
    Xf = np.zeros(Prediction_Interval)
    for k in np.arange(start-1, N-1):
        Zf[k-start] = hfun(Xpf[:, start], k)
        Xf[k-start] = k

    # 画线
    nax = [start, start]
    nay = [0, 1]
    plt.figure(figsize=(12, 9))
    plt.title('Particle filter  '+ Battery_name)  # 折线图标题
    plt.xlabel('Number of Cycles', fontsize=14)
    plt.ylim((0, 1.1))
    plt.ylabel(names[0], fontsize=14)
    plt.plot(battery['cycle'],  Z,          markersize=3)
    plt.plot(battery['cycle'],  Zpf[0, :],  markersize=3)
    plt.plot(Xf,                Zf,         markersize=3)
    plt.plot(nax,               nay,        linewidth=4)
    plt.legend(['Measured value', 'pf Predictive value', 'Natural predicted value'])
    plt.show()
