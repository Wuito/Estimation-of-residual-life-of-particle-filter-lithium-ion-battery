# 程序功能： 读取一种电池的数据解析计算
#           SOH,CCCT,CVCT,resistance,capacity数据并存储为npy文件
#       使用马里兰大学数据集，电池型号为CS2电池
# From: SWUST IPC14 DaiXingRong

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

Battery_name = 'CS2_38'   # 加载数据
dir_path = 'dataset/'

# 除去数据异常的较大值和较小值点
def drop_outlier(array, count, bins):
    index = []
    range_ = np.arange(1, count, bins)  # 返回一个有终点和起点的固定步长的排列，起始为1，终值为count，步长为bins
    for i in range_[:-1]:
        array_lim = array[i:i+bins] # 从array中截取一个从i到i+bins的数组段
        sigma = np.std(array_lim)   # 计算该数组段的全局标准差
        mean = np.mean(array_lim)   # 计算数组段的平均值
        th_max,th_min = mean + sigma*2, mean - sigma*2  # 以方差和平均值作为上下限取值
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i    # 将当前的idx值+i后赋值到实际数组的角标
        index.extend(list(idx))     # 在列尾增加一个list
    return np.array(index)

Battery = {}

print('Load the data directory structure ' + Battery_name + ' ...')
path = glob.glob(dir_path + Battery_name + '/*.xlsx')
dates = []
for p in path:
    df = pd.read_excel(p, sheet_name=1)
    print('Load file sequence ' + str(p) + ' ...')
    dates.append(df['Date_Time'][0])
idx = np.argsort(dates)
path_sorted = np.array(path)[idx]
print("The file structure was read successfully. There are {} files in total".format(len(path_sorted)))

count = 0
discharge_capacities = []
health_indicator = []
internal_resistance = []
CCCT = []
CVCT = []

for p in path_sorted:
    df = pd.read_excel(p, sheet_name=1)
    print('Load and Analytical data' + str(p) + ' ...')
    cycles = list(set(df['Cycle_Index']))             # 按照每个循环次数读取
    # 在单个表中按照Cycle_Index（实验次数）逐次取出每次的数据进行处理
    for c in cycles:
        df_lim = df[df['Cycle_Index'] == c]
        # Charging
        df_c = df_lim[(df_lim['Step_Index'] == 2) | (df_lim['Step_Index'] == 4)]    # 第2步和第4步充电
        c_v = df_c['Voltage(V)']        # 电压
        c_c = df_c['Current(A)']        # 实时电流值
        c_t = df_c['Test_Time(s)']      # 记录的测试时间
        # CC or CV
        df_cc = df_lim[df_lim['Step_Index'] == 2]   # 第2步恒流充电
        df_cv = df_lim[df_lim['Step_Index'] == 4]   # 第4步恒压充电
        CCCT.append(np.max(df_cc['Test_Time(s)']) - np.min(df_cc['Test_Time(s)']))      # 恒定电流充电时间
        CVCT.append(np.max(df_cv['Test_Time(s)']) - np.min(df_cv['Test_Time(s)']))      # 恒定电压充电时间

        # Discharging
        df_d = df_lim[df_lim['Step_Index'] == 7]    # 第7步时放电
        d_v = df_d['Voltage(V)']
        d_c = df_d['Current(A)']
        d_t = df_d['Test_Time(s)']               # 步长时间
        d_im = df_d['Internal_Resistance(Ohm)']  # 内阻

        if len(list(d_c)) != 0:
            time_diff = np.diff(list(d_t))  # 求时间差，np.diff：计算数组中n[a]-n[a-1]
            d_c = np.array(list(d_c))[1:]   # 读出电流值
            discharge_capacity = time_diff * d_c / 3600  # 计算安培时(Ah)Q = A*h
            # print("discharge_capacity shape[0] is:{}".format(discharge_capacity.shape[0]))
            # 将所有第7步中的电池放电容量求和并正定
            # discharge_capacity_sum = np.sum(discharge_capacity)
            # discharge_capacities.append(-1 * discharge_capacity_sum)
            discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
            discharge_capacities.append(-1 * discharge_capacity[-1])

            dec = np.abs(np.array(d_v) - 3.8)[1:]       # np.abs:求绝对值
            # np.argmin:将数组展平返回最小值的下标
            start = np.array(discharge_capacity)[np.argmin(dec)]    # 取出当电压最接近3.8V时电池的放电量
            dec = np.abs(np.array(d_v) - 3.4)[1:]
            end = np.array(discharge_capacity)[np.argmin(dec)]  # 取出电压最接近3.4V时的电池放电量
            health_indicator.append((-1 * (end - start)))     # 这里定义的SOH是电池从3.8V放电到3.4V的电池容量

            internal_resistance.append(np.mean(np.array(d_im)))     # 求放电阶段电池的的内阻平均值
            count += 1
health_indicator = health_indicator/np.max(health_indicator)    # 计算SOH

discharge_capacities = np.array(discharge_capacities)
SOC = discharge_capacities/1.1                  # CS2电池的标准容量为1.1Ah
health_indicator = np.array(health_indicator)
internal_resistance = np.array(internal_resistance)
CCCT = np.array(CCCT)
CVCT = np.array(CVCT)

idx = drop_outlier(discharge_capacities, count, 40)     # 以所有轮次中放电步骤的放电容量值为原始数据，以40为步长，对数据进行清洗处理
df_result = pd.DataFrame({'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),  # 步数
                          'capacity': SOC[idx],            # 容量
                          'SoH': health_indicator[idx],     # SOH
                          'resistance': internal_resistance[idx],   # 电池内阻
                          'CCCT': CCCT[idx],
                          'CVCT': CVCT[idx]})
Battery[Battery_name] = df_result
np.save(dir_path + Battery_name, Battery)
print("Data parsing succeeded. The .npy file was saved to {}".format(dir_path + Battery_name + '.npy'))

