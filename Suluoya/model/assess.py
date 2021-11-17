import math
import numpy as np
import pandas as pd


def grey_relation(df, rho=0.5, func=0, mother=''):
    '''   
    Func:灰色关联\n
    IN:dataframe原始数据\n
    OUT:series每个样本的评分\n
    rhp --> 关联度一般为0.5\n
    func --> 0：回归；1：评价\n
    mother --> 母序列，默认为空
    '''
    df = df/df.mean()
    df_temp = df.copy(deep=True)
    if func == 1:
        # 评价
        mother_series = df.max(axis=1)
    else:
        # 回归
        mother_series = df[mother]
        df = df.drop(mother, axis=1)
    df = np.abs(df.sub(mother_series, axis=0))
    min = df.min().min()
    max = df.max().max()
    grey = (min+rho*max)/(df+rho*max)
    weights = grey.sum()/df.index.size
    if func == 0:
        # 回归
        return weights
    else:
        # 评价
        weights = weights/weights.sum()
        return df_temp.dot(weights)


def entropy_weight(df):
    '''
    Func:熵权法计算权重\n
    In:dataframe各指标数据\n
    Out:series各指标权重
    '''
    # 归一化
    df = df.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    # 求k
    k = 1 / math.log(df.index.size)
    # 信息熵
    df_p = df/df.sum()
    df_inf = -np.log(df_p).replace([np.inf, -np.inf], 0)*df_p*k
    # 计算冗余度
    redundancy = 1 - df_inf.sum()
    # 计算各指标的权重
    weights = redundancy/redundancy.sum()
    return weights


def topsis(df, weights=[]):
    '''
    Func:TOPSIS法\n
    IN:dataframe各指标数据\n
    OUT:series每个样本的分数\n
    weights --> 按顺序传入指标权重
    '''
    df = df/((df**2).sum())
    series_max = df.max()
    series_min = df.min()
    if weights == []:
        weights = np.ones(df.columns.size)/df.columns.size
    D_max = ((weights*((df-series_max)**2)).sum(axis=1))**(1/2)
    D_min = ((weights*((df-series_min)**2)).sum(axis=1))**(1/2)
    D = D_min/(D_min+D_max)
    return D/D.sum()


def entropy_weight_topsis(df):
    '''
    Func:基于熵权法修正的TOPSIS法\n
    IN:dataframe各指标数据\n
    OUT:series每个样本的分数
    '''
    return topsis(df, list((entropy_weight(df))))


def ladder_distribution(index_dict={'A': [1, 2, 3], 'B': [2, 3, 4], 'C': [3, 4, 5]}, score_dict={'A': 2.9, 'B': 5, 'C': 4.1}):
    '''
    Func:梯型分布模糊综合评价\n
    OUT:模糊综合评价矩阵R
    '''
    def calculate(index_list, score):
        result = [0]*len(index_list)
        index_list.append(score)
        index_list.sort()
        index = index_list.index(score)
        if index > 0 and index < len(index_list)-1:
            left = index_list[index-1]
            right = index_list[index+1]
            result[index] = (score-left)/(right-left)
            result[index-1] = (right-score)/(right-left)
        elif index == 0:
            result[0] = 1
        else:
            result[-1] = 1
        return result
    keys = index_dict.keys()
    retult = [calculate(i[1][0], i[1][1]) for i in enumerate(
        zip(index_dict.values(), score_dict.values()))]
    return pd.DataFrame(dict(zip(keys, retult))).T


def fuzzy_synthesis(weights, R):
    '''
    Func:模糊综合评级\n
    OUT:总隶属度
    weights --> dict,指标权重\n
    R --> 模糊综合评价矩阵R
    '''
    return pd.Series(weights).dot(R)


if __name__ == '__main__':
    test_dict = {'A': [9, 3, 2, 4], 'B': [
        3, 5, 7, 8], 'C': [10, 4, 8, 9], }
    test_df = pd.DataFrame(test_dict)
    t = entropy_weight_topsis(test_df)
    print(t)
