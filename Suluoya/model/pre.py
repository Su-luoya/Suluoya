import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.interpolate import interp1d
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号


class DataPreprocessor(object):
    '''
    Func:数据预处理\n
    df --> 原始数据
    '''

    def __init__(self, df):
        self.df = df
        self.cubic_dict = {}

    def normalize(self, columns=[]):
        '''
        Func:归一化选定变量
        '''
        for column in columns:
            max = self.df[column].max()
            min = self.df[column].min()
            self.df[column] = self.df[column].map(
                lambda x: (x - min) / (max - min))

    def normalize_all(self):
        '''
        Func:归一化全部变量
        '''
        self.df = self.df.apply(lambda x: (
            (x - np.min(x)) / (np.max(x) - np.min(x))))

    def direction(self, column, type=1, best_value=0, range=(0, 1)):
        '''
        type=0 --> 极大型，越大越好\n
        type=1 --> 极小型，越小越好\n
        type=2 --> 中间型，接近某个值越好，需传入best_value\n
        type=3 --> 区域型，落在某个区间越好，需传入range
        '''
        def rgnl(x):
            min_value = range[0]
            max_value = range[1]
            M = max(min_value-np.min(self.df[column]),
                    np.max(self.df[column])-max_value)
            if x < min_value:
                return 1-(min_value-x)/M
            elif x > max_value:
                return 1-(x-max_value)/M
            else:
                return 1
        if type == 1:
            self.df[column] = np.max(self.df[column])-self.df[column]
        elif type == 2:
            self.df[column] = 1-abs(self.df[column]-best_value) / \
                np.max(abs(self.df[column]-best_value))
        elif type == 3:
            self.df[column] = self.df[column].map(lambda x: rgnl(x))

    def dummy(self, columns=[]):
        '''
        Func:独热编码选定变量
        '''
        for column in columns:
            self.df = self.df.join(pd.get_dummies(
                self.df[column], prefix=column))
            del self.df[column]

    def dummy_all(self):
        '''
        Func:独热编码全部分类变量
        '''
        self.df = pd.get_dummies(self.df)

    def ks_test(self, columns=[]):
        '''
        Func:检验变量是否符合正态分布
        '''
        df_ks = self.df[columns]
        df_ks.dropna(inplace=True)
        return {column: kstest(df_ks.astype(float), 'norm') for column in columns}

    def correlation(self, columns=[], method='pearson'):
        '''
        Func:计算相关系数\n
        method --> pearson or spearman or kendall
        '''
        df_cor = self.df[columns]
        df_cor.dropna(inplace=True)
        lists = []
        for i in combinations(columns, 2):
            if method == 'pearson':
                result = stats.pearsonr(df_cor[i[0]].astype(
                    float), df_cor[i[1]].astype(float))
            elif method == 'spearman':
                result = stats.spearmanr(df_cor[i[0]].astype(
                    float), df_cor[i[1]].astype(float))
            elif method == 'kendall':
                result = stats.kendalltau(df_cor[i[0]].astype(
                    float), df_cor[i[1]].astype(float))
            else:
                raise ValueError('method is wrong!')
            lists.append([i, result[0], result[1], result[1] < 0.05])
        df = pd.DataFrame(
            lists, columns=['combinations', 'correlation', 'pvalue', 'rejection'])
        return df.set_index('combinations')

    def cubic(self, x, y, kind='cubic'):
        '''
        Func: 插值法(cubic/linear)
        x --> 插值依据序列
        y --> 待插值序列
        '''
        df_null = self.df[[x, y]].isnull()
        df_x = self.df[x][df_null[x] == False]
        df_y = self.df[y][df_null[x] == False]
        df_null['is_null'] = df_null[y]
        null_index = df_null[df_null['is_null'] == True].index
        x_news = df_x.iloc[null_index]
        x_cubic = df_x.drop(null_index, axis=0)
        y_cubic = df_y.drop(null_index, axis=0)
        f = interp1d(x_cubic, y_cubic, kind=kind)
        self.cubic_dict[(x, y)] = {
            'f': f, 'null_index': null_index, 'x_news': x_news}

    def plot(self, x, y, f, null_index):
        # 画图
        fig, ax = plt.subplots(len(self.cubic_dict.keys()),
                               1, sharex=True, figsize=(6, 6))
        num = 0
        for i, j in self.cubic_dict.items():
            x = i[0]
            y = i[1]
            f = j['f']
            null_index = j['null_index']
            x_plot = np.linspace(self.df[x].min(), self.df[x].max(), 500)
            ax[num].plot(x_plot, f(x_plot))
            ax[num].spines['top'].set_visible(False)
            ax[num].spines['right'].set_visible(False)
            ax[num].set_xlabel(f"{x}")
            ax[num].set_ylabel(f"{y}")
            for i, j in self.df[[x, y]].iloc[null_index].iterrows():
                ax[num].text(j[0], j[1], '%.2f' % j[1])
                ax[num].plot(j[0], j[1], marker=".", markersize=8)
            num += 1
        plt.savefig(f"{x}-{y}.png", dpi=500)

    def cubic_all(self):
        '''
        Func:填充缺失值
        '''
        for i, j in self.cubic_dict.items():
            self.df[i[1]].iloc[j['null_index']] = j['f'](j['x_news'])
        self.plot(x=i[0], y=i[1], f=j['f'], null_index=j['null_index'])

    def missing_summary(self):
        '''
        Func:缺失值所在列
        '''
        return self.df.isnull().any()


if __name__ == '__main__':
    test_dict = {'A': [9, 3, 2, 4, 5, None, 7, 8],
                 'B': [2, 4, 3, None, 1, 13, 14, 12],
                 'C': [3, 5, 7, 8, 9, None, 11, 12],
                 'D': [2, 6, 7, 9, 1, 11, 13, 12]
                 }
    df = pd.DataFrame(test_dict)
    dp = DataPreprocessor(df)
    #print(dp.correlation(columns=['A', 'B', 'C', 'D']))
    dp.cubic(x='B', y='A')
    dp.cubic(x='A', y='B')
    dp.cubic(x='D', y='C')
    dp.cubic_all()
    print(dp.df)
