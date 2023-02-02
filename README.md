# 基于粒子滤波算法的锂离子电池剩余使用寿命估计



## 文件内容

```
└─Src_Data
    my_processing_onedata.py
    pf_main.py
    └─dataset
    	CS2_35.npy
    	CS2_36.npy
    	CS2_37.npy
    	CS2_38.npy
```

#####  my_processing_onedata.py

主要使用了两个代码，第一个代码`my_processing_onedata.py`是进行数据处理并且将处理好的数据保存为`.npy`格式的文件。这个脚本无法直接运行，需要先下载马里兰大学的数据，下载地址：[Battery Research Data | Center for Advanced Life Cycle Engineering (umd.edu)](https://calce.umd.edu/data#CS2)   下载后的数据保存到`/dataset.`路径下，代码里使用的是CS2电池组，将四个不同温度的数据分别用4文件夹存放，然后运行脚本就可以计算得到有效数据并保存了。

这里因为原始数据太大了，就不保存了，有需要复现的话再单独下载。

##### pf_main.py

这个代码将读取处理好的4个数据之一，并进行粒子滤波预测。

##### my_calce.py

读取处理好的4个数据文件之一，显示出主要的各项参数



## requirement

```
python				3.8
matplotlib           3.6.0
numpy                1.23.3
pandas               1.5.0
```

## matlab 文件夹下是matlab相关的程序和数据包
推荐使用matlab18及以上版本
### 在matlab环境中运行mian.m文件时注意文件夹下的工程路径应该包含以下文件
.mat文件是根据代码需要可以在data目录中找到
```
+--- Battery_Capacity.mat
+--- hfun.m
+--- main.m
+--- pf.m
+--- randomR.m
+--- residualR.m

```

## 详细原理解释可以查看连接 [CSDN博客](https://blog.csdn.net/weixin_47407066/article/details/127424785?spm=1001.2014.3001.5501)
