# model

## 常用库

```Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

test_dict = {'A': [9, 3, 2, 4], 'B': [3, 5, 7, 8], 'C': [10, 4, 8, 9], }
df = pd.DataFrame(test_dict)

```


## 指标评价

&ensp;&ensp;&ensp;&ensp;#### 灰色预测

```Python
from Suluoya.model import grey_relation
# rhp --> 关联度一般为0.5
# func --> 0：回归；1：评价
# mother --> 母序列，默认为空
grey_relation(df, rho=0.5, func=0, mother='')

```


&ensp;&ensp;&ensp;&ensp;#### 熵权法

```Python
from Suluoya.model import entropy_weight
entropy_weight(df)

```


&ensp;&ensp;&ensp;&ensp;#### TOPSIS法

```Python
from Suluoya.model import topsis
# weights --> 按顺序传入指标权重
topsis(df, weights=[])
```


&ensp;&ensp;&ensp;&ensp;#### 基于熵权法修正的TOPSIS法

```Python
from Suluoya.model import entropy_weight_topsis
entropy_weight_topsis(df)

```


&ensp;&ensp;&ensp;&ensp;#### 模糊综合评级

```Python
from Suluoya.model import ladder_distribution,fuzzy_synthesis
# 梯型分布时的模糊综合评价矩阵R
R = ladder_distribution(index_dict={'A': [1, 2, 3], 
                                    'B': [2, 3, 4], 
                                    'C': [3, 4, 5]}, 
                        score_dict={'A': 2.9, 
                                    'B': 5, 
                                    'C': 4.1}
                        )
fuzzy_synthesis(weights, R) # weights为dict，需自行构建
```


## 机器学习

&ensp;&ensp;&ensp;&ensp;#### 机器学习探索性实验

```Python
from Suluoya.model import exploratory_experiment
# target --> 因变量
# target_type --> R连续变量；C分类变量
exploratory_experiment(df, target, target_type='R')

```


## 回归

&ensp;&ensp;&ensp;&ensp;#### 多重线性回归

```Python
from Suluoya.model import linear_regression as lr
# df --> 原始数据
# y --> 因变量
lr(df,y)
```


&ensp;&ensp;&ensp;&ensp;#### 二分类logistic回归

```Python
from Suluoya.model import logistic_regression as lr
# df --> 原始数据
# y --> 因变量
lr(df,y)

```


&ensp;&ensp;&ensp;&ensp;#### 分位数回归

```Python
from Suluoya.model import quantile_regression
# y --> 因变量
# step --> 分位数步长
qr = quantile_regression(df, y='y', step=0.05) # 初始化
qr.quant_summary(q=0.5) # 分位数回归概要，q --> 分位数
qr.ols_summary() # 最小二乘法概要
qr.quantile(q=0.5) # 不准用！！！
qr.quantiles() # 包含从0.05到0.95的分位数回归结果概要 --> dataframe
qr.quantiles_dict() # 包含从0.05到0.95的分位数回归结果概要 --> dict，键为自变量
qr.plot() # 作图
###############################
###############################
# 一个例子
qr = quantile_regression(data, 'y')
print(qr.quantiles())
qr.plot()
```


## 数据预处理

```Python
from Suluoya.model import DataPreprocessor
dp = DataPreprocessor(df) # 初始化
dp.normalize(columns=[]) # 归一化选定变量
dp.normalize_all() # 归一化全部变量
'''
direction用法：
type=0 --> 极大型，越大越好
type=1 --> 极小型，越小越好
type=2 --> 中间型，接近某个值越好，需传入best_value
type=3 --> 区域型，落在某个区间越好，需传入range
'''
dp.direction(column, type=1, best_value=0, range=(0, 1))
dp.dummy(columns=[]) # 独热编码选定变量
dp.dummy_all() # 独热编码全部分类变量
dp.ks_test(columns=[]) # 检验变量是否符合正态分布
dp.correlation(columns=[], method='pearson') # 计算相关系数 --> pearson or spearman or kendall
dp.missing_summary()
dp.cubic(x, y, kind='cubic') # 插值法
dp.cubic_all() # 画图+补缺失值

```


