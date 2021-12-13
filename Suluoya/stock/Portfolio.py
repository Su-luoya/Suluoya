import json
import math
import os
import random
import sys
import time
import warnings
from functools import reduce
from itertools import combinations, product
from operator import add
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_errors
import scipy.optimize as sco
import seaborn as sns
import statsmodels.api as sm
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Kline, Line
from pyecharts.commons.utils import JsCode
from statsmodels import regression
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Stock import StockData
    from ..log.log import hide, makedir, progress_bar, show, slog, sprint
except:
    from data.Stock import StockData
    from log.log import hide, makedir, progress_bar, show, slog, sprint

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 13


class Markovitz(object):
    '''
    组合投资权重\n
    names=['贵州茅台', '隆基股份', '五粮液']\n
    start_date='2021-05-01'\n
    end_date='2021-11-01'\n
    frequency='d' --> d/w/m\n
    rfr=0.023467/365\n
    funds=10000000\n
    path --> 默认缓存路径为：".\\Suluoya cache\\"，可传入False不缓存
    '''

    def __init__(self, names=['比亚迪', '阳光电源', '璞泰来', '紫光国微', '盛新锂能'],
                 start_date='2021-05-01',
                 end_date='2021-11-01',
                 frequency='d',
                 rfr=0.023467,
                 funds=10000000,
                 path='.\\Markovitz cache\\'):
        self.names = names
        self.lens = len(names)
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.rfr = (rfr*100) / \
            {'d': 365, 'w': 52, 'm': 30}[frequency]
        self.funds = funds
        self.path = path
        if self.path:
            makedir(self.path, '')
        sprint('Initializing...')
        if not self.path:
            sd = StockData(names=self.names, start_date=self.start_date,
                           end_date=self.end_date, frequency=self.frequency)
            self.datas = sd.stocks_data()
        else:
            try:
                self.datas = pd.read_csv(
                    f'{self.path}\\stock data\\stocks_data.csv')
            except:
                sd = StockData(names=self.names, start_date=self.start_date,
                               end_date=self.end_date, frequency=self.frequency, path=self.path)
                self.datas = sd.stocks_data()
        self.datas.index = self.datas['name']
        self.data = self.datas.reset_index(drop=True)
        self.date = list(map(lambda x: str(x)[:10], self.data.date.unique()))
        self.first_date = self.date[0]
        self.last_date = self.date[-1]
        # 第一天开盘价
        self.first_price = self.data[self.data.date == self.data.date.unique(
        )[0]][['open', 'name']].set_index('name').to_dict()['open']
        # 最后一天收盘价
        self.last_price = self.data[self.data.date == self.data.date.unique(
        )[-1]][['close', 'name']].set_index('name').to_dict()['close']
        # 每只股票最大手数
        self.max_shares_dict = {name: math.floor(
            self.funds/(shares*100)) for name, shares in self.last_price.items()}

    def weights(self, number=5000):
        '''
        生成股票随机权重
        '''
        return np.random.dirichlet(np.ones(self.lens), size=number)

    def calculate(self):
        '''
        计算收益率均值、协方差矩阵、相关系数矩阵
        '''
        data = self.data[['date', 'name', 'pctChg']]
        # 收益率均值
        data_mean = data.groupby('name').mean().T[self.names]
        # 协方差矩阵 & 相关系数矩阵
        df = pd.DataFrame()
        for name in self.names:
            df[name] = list(data[data['name'] == name]['pctChg'])
        data_cov = df.cov()
        data_corr = df.corr()
        if self.path:
            makedir(self.path, 'mean,cov,corr')
            data_mean.T.to_csv(
                f'{self.path}\\mean,cov,corr\\data_mean.csv')
            data_cov.to_csv(f'{self.path}\\mean,cov,corr\\data_cov.csv')
            data_corr.to_csv(f'{self.path}\\mean,cov,corr\\data_corr.csv')
        return {'mean': data_mean, 'cov': data_cov, 'correlation': data_corr}

    def heatmap(self, show=True):
        '''
        收益率相关系数热力图
        '''
        if self.path:
            try:
                data_corr = pd.read_csv(f'{self.path}\\mean,cov,corr\\data_corr.csv').rename(
                    {'Unnamed: 0', 'correlation'}).set_index('correlation')
            except:
                data_corr = self.calculate()['correlation']
        else:
            data = self.data[['name', 'pctChg']]
            df = pd.DataFrame()
            for name in self.names:
                df[name] = list(data[data['name'] == name]['pctChg'])
            data_corr = df.corr()
        # 画图
        # plt.subplots(figsize=(9, 9))
        sns.heatmap(data_corr, annot=True, vmax=1,
                    square=True, cmap='Purples', cbar=False)
        if show:
            plt.show()
        else:
            plt.savefig(f'{self.path}\\heatmap.svg', format='svg')

    def sharpe(self, weights):
        '''
        按照names顺序传入权重（权重和为1）
        '''
        data_dict = self.calculate()
        data_mean = data_dict['mean']
        data_cov = data_dict['cov']
        weights = np.array(weights)
        rate = data_mean.dot(weights.T)['pctChg']
        risk = np.sqrt(weights.dot(data_cov).dot(weights.T))
        return (self.rfr-rate)/risk  # 相反数

    def optimization(self):
        '''
        非线性规划求解最大夏普比率和对应权重
        '''
        opts = sco.minimize(fun=self.sharpe,
                            # 传入股票权重，即shapre函数的参数weights
                            x0=np.ones(self.lens)/self.lens,
                            bounds=tuple((0, 1)for x in range(self.lens)),
                            constraints={'type': 'eq',
                                         'fun': lambda x: np.sum(x) - 1}
                            )
        opt_dict = {'weights': dict(
            zip(self.names, list(opts.x))), 'sharpe': -opts.fun}
        if self.path:
            df_opt = pd.DataFrame(opt_dict)
            df_opt['sharpe'] = None
            df_opt['sharpe'].iloc[0] = opt_dict['sharpe']
            df_opt.to_csv(f'{self.path}\\max sharpe and weights.csv')
        return opt_dict

    def scatter_data(self, number=5000):
        '''
        散点数据，默认生成5000个
        '''
        data_dict = self.calculate()
        data_mean = data_dict['mean']
        data_cov = data_dict['cov']
        weights = self.weights(number=number)
        # 散点DataFrame
        df_scatter = pd.DataFrame()
        # 随机权重
        df_scatter['weights'] = pd.Series(map(lambda x: str(x), weights))
        # 风险
        df_scatter['risk'] = np.sqrt(np.diagonal(
            weights.dot(data_cov).dot(weights.T)))
        # 收益率
        df_scatter['rate'] = data_mean.dot(weights.T).T['pctChg']
        # 夏普比率
        df_scatter['sharpe'] = (
            df_scatter.rate-self.rfr)/df_scatter.risk
        df_scatter = df_scatter.sort_values(by='sharpe', ascending=False)
        if self.path:
            makedir(self.path, 'scatter data')
            df_scatter.to_csv(f'{self.path}\\scatter data\\scatter_data.csv')
        return df_scatter

    def boundary_scatter_data(self, number=500):
        '''
        边界散点数据，默认生成500个
        '''
        if self.path:
            try:
                df_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\scatter_data.csv', index=False)
            except:
                df_scatter = self.scatter_data()
        else:
            df_scatter = self.scatter_data()
        data_dict = self.calculate()
        data_mean = data_dict['mean']
        data_cov = data_dict['cov']
        scatter_list = []
        sprint('Searching for boundary scatter...')
        for i in trange(number):
            random_rate = random.uniform(
                df_scatter.rate.min(), df_scatter.rate.max())
            constraints = ({'type': 'eq', 'fun': lambda weights: weights.sum()-1},
                           {'type': 'eq', 'fun': lambda weights: data_mean.dot(weights.T)['pctChg']-random_rate})
            opts = sco.minimize(fun=lambda weights: weights.dot(data_cov).dot(weights.T),
                                x0=np.ones(self.lens)/self.lens,
                                bounds=tuple((0, 1)for x in range(self.lens)),
                                constraints=constraints
                                )
            scatter_list.append([opts.x,  np.sqrt(opts.fun), random_rate])
        df_boundary_scatter = pd.DataFrame(scatter_list, columns=[
            'weights', 'risk', 'rate'])
        df_boundary_scatter['sharpe'] = (
            df_boundary_scatter.rate-self.rfr)/df_boundary_scatter.risk
        df_boundary_scatter = df_boundary_scatter.sort_values(
            by='sharpe', ascending=False)
        if self.path:
            makedir(self.path, 'scatter data')
            df_boundary_scatter.to_csv(
                f'{self.path}\\scatter data\\boundary_scatter_data.csv')
        return df_boundary_scatter

    def get_return(self, rate=0.05):
        '''
        计算给定收益率对应组合的风险、权重和夏普比率 --> 返回一个包含收益率、标准差、权重和夏普比率的字典
        '''
        if self.path:
            try:
                df_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\scatter_data.csv', index=False)
            except:
                df_scatter = self.scatter_data()
        else:
            df_scatter = self.scatter_data()
        data_dict = self.calculate()
        data_mean = data_dict['mean']
        data_cov = data_dict['cov']
        constraints = ({'type': 'eq', 'fun': lambda weights: weights.sum()-1},
                       {'type': 'eq', 'fun': lambda weights: data_mean.dot(weights.T)['pctChg']-rate})
        opts = sco.minimize(fun=lambda weights: weights.dot(data_cov).dot(weights.T),
                            x0=np.ones(self.lens)/self.lens,
                            bounds=tuple((0, 1)for x in range(self.lens)),
                            constraints=constraints
                            )
        return {'weights': dict(zip(self.names, opts.x)),  'risk': np.sqrt(opts.fun), 'rate': rate, 'sharpe': (rate-self.rfr)/np.sqrt(opts.fun)}

    def cml(self, show=True):
        '''
        资本市场线 & 有效边界
        '''
        if self.path:
            try:
                df_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\scatter_data.csv')
                df_boundary_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\boundary_scatter_data.csv')
            except:
                df_scatter = self.scatter_data()
                df_boundary_scatter = self.boundary_scatter_data()
            df_scatter['boundary'] = False
            df_boundary_scatter['boundary'] = True
            pd.concat([df_scatter, df_boundary_scatter]).to_csv(
                f'{self.path}\\scatter data\\all_scatter_data.csv')
        else:
            df_scatter = self.scatter_data()
            df_boundary_scatter = self.boundary_scatter_data()

        max_sharpe = self.optimization()['sharpe']
        sprint(f'max sharpe: {max_sharpe}')
        plt.cla()
        plt.style.use('seaborn-paper')
        plt.scatter(df_scatter.risk, df_scatter.rate,
                    s=10, marker=".", c='b')
        plt.scatter(df_boundary_scatter.risk, df_boundary_scatter.rate,
                    s=10, marker=".", c='r')
        plt.axline(xy1=(0, self.rfr), slope=max_sharpe, c='m')
        plt.xlim(df_scatter.risk.min()*0.8, df_scatter.risk.max()*1.2)
        plt.ylim(df_scatter.rate.min()*0.8, df_scatter.rate.max()*1.2)
        plt.xlabel('Risk')
        plt.ylabel('Yield')
        if show:
            plt.show()
        else:
            plt.savefig(f'{self.path}\\cml.svg', format='svg')
        return pd.concat([df_scatter, df_boundary_scatter])

    def exam(self, shares_dict={'迎驾贡酒': 1, '明微电子': 2, '健民集团': 3}):
        '''
        检验：
        0<=手数<=最大手数
        总成本<=fund
        返回sharpe或0
        '''
        for name, number in shares_dict.items():
            if number > self.max_shares_dict[name]:
                return 0  # 检验不符合
        # if 0 in shares_dict.values():
        #     return 0  # 检验不符合
        weights = np.array([self.last_price[name]*100*shares_dict[name]
                           for name in self.names])
        if weights.sum() > self.funds:
            return 0  # 检验不符合
        weights = weights/weights.sum()
        return -self.sharpe(weights=weights)

    def init_port(self):
        '''
        初始临近整数组合 --> DataFrame
        '''
        if self.path:
            try:
                opt_dict = pd.read_csv(f'{self.path}\\max sharpe and weights.csv').set_index(
                    'Unnamed: 0')['weights'].to_dict()
            except:
                opt_dict = self.optimization()['weights']
        # 股票初始手数（不取整）
        init_shares_dict = {}
        for name, weight in opt_dict.items():
            init_shares_dict[name] = (
                self.funds*weight)/(self.last_price[name]*100)
        ceils = [math.ceil(i) for i in init_shares_dict.values()]
        floors = [math.floor(i) for i in init_shares_dict.values()]
        result = np.array(list(zip(ceils, floors)))
        df_shares = pd.DataFrame(
            list(product(*result)), columns=init_shares_dict.keys())
        return df_shares

    def init_scatter_data(self):
        '''
        初始临近整数组合散点
        '''
        # 初始临近整数组合权重
        df_init_shares = self.init_port()
        for name in self.names:
            df_init_shares[name] = df_init_shares[name]*self.last_price[name]
        df_init_shares['sum'] = df_init_shares.sum(axis=1)
        for name in self.names:
            df_init_shares[name] = df_init_shares[name]/df_init_shares['sum']
        df_init_shares = df_init_shares[df_init_shares['sum']
                                        <= self.funds/100]
        weights = np.array([list(i.values()) for i in list(
            df_init_shares[self.names].T.to_dict().values())])
        # 收益率、风险和夏普比率
        data_dict = self.calculate()
        data_mean = data_dict['mean']
        data_cov = data_dict['cov']
        # 散点DataFrame
        df_init_scatter = pd.DataFrame()
        # 随机权重
        df_init_scatter['weights'] = pd.Series(map(lambda x: str(x), weights))
        # 风险
        df_init_scatter['risk'] = np.sqrt(np.diagonal(
            weights.dot(data_cov).dot(weights.T)))
        # 收益率
        df_init_scatter['rate'] = data_mean.dot(weights.T).T['pctChg']
        # 夏普比率
        df_init_scatter['sharpe'] = (
            df_init_scatter.rate-self.rfr)/df_init_scatter.risk
        df_init_scatter = df_init_scatter.sort_values(
            by='sharpe', ascending=False)
        if self.path:
            makedir(self.path, 'scatter data')
            df_init_scatter.to_csv(
                f'{self.path}\\scatter data\\df_init_scatter.csv')
        return df_init_scatter

    def init_cml(self, show=True):
        '''
        初始临近组合散点图
        '''
        if self.path:
            try:
                df_init_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\df_init_scatter.csv')
                df_boundary_scatter = pd.read_csv(
                    f'{self.path}\\scatter data\\boundary_scatter_data.csv')
            except:
                df_init_scatter = self.init_scatter_data()
                df_boundary_scatter = self.boundary_scatter_data()
        else:
            df_init_scatter = self.init_scatter_data()
            df_boundary_scatter = self.boundary_scatter_data()

        max_sharpe = self.optimization()['sharpe']
        plt.style.use('seaborn-paper')
        plt.cla()
        plt.scatter(df_init_scatter.risk, df_init_scatter.rate,
                    s=100, marker=".", c='r')
        plt.scatter(df_boundary_scatter.risk, df_boundary_scatter.rate,
                    s=10, marker=".", c='b')
        plt.axline(xy1=(0, self.rfr), slope=max_sharpe, c='m')
        plt.xlim(df_init_scatter.risk.min()*0.8,
                 df_init_scatter.risk.max()*1.2)
        plt.ylim(df_init_scatter.rate.min()*0.8,
                 df_init_scatter.rate.max()*1.2)
        plt.xlabel('Risk')
        plt.ylabel('Yield')
        if show:
            plt.show()
        else:
            plt.savefig(f'{self.path}\\init_cml.svg', format='svg')

    def init_tree(self):
        '''
        第一次分枝定界结果 --> DataFrame
        '''
        init_tree_list = []
        for index, shares_series in self.init_port().iterrows():
            shares_dict = shares_series.to_dict()
            examed_sharpe = self.exam(shares_dict)
            # if examed_sharpe > 0:
            #     init_tree_list.append([shares_dict, examed_sharpe])
            init_tree_list.append([shares_dict, examed_sharpe])
        # if len(init_tree_list) == 0:
        #     sprint('fund数值过小或组合选择存在问题，存在股票权重为0！！！')
            # raise ValueError('fund数值过小或组合选择存在问题，存在股票权重为0！！！')
        return pd.DataFrame(init_tree_list, columns=['shares', 'sharpe']).sort_values(by='sharpe', ascending=False)

    def near_port_constructor(self, shares_dict={'迎驾贡酒': 1, '明微电子': 2, '健民集团': 3}, near=1):
        '''
        构建临近整数组合 --> DataFrame
        '''
        lists = [shares_dict]
        for name, number in shares_dict.items():
            shares_dict_copy_1 = shares_dict.copy()
            shares_dict_copy_2 = shares_dict.copy()
            if number == 0:
                shares_dict_copy_1[name] = number+near
                lists.append(shares_dict_copy_1)
                continue
            elif number == 1:
                near = 1
            lists.append(shares_dict_copy_1)
            lists.append(shares_dict_copy_2)
            shares_dict_copy_1[name] = number-near
            shares_dict_copy_2[name] = number+near
        return pd.DataFrame(lists)

    def port(self, df_shares, near=1):
        '''
        临近整数组合 --> DataFrame
        '''
        port_list = []
        # 寻找每一个检验剩下的组合的临近组合
        for i in df_shares.itertuples():
            port_list.append(self.near_port_constructor(i.shares, near=near))
        # 拼接所有临近整数解
        return pd.Series(pd.concat(port_list).drop_duplicates().reset_index(drop=True).T.to_dict())

    def tree(self):
        '''
        分枝定界
        返回最优整数解和sharpe
        '''
        # 初始整数组合
        exam_tree = pd.DataFrame()
        exam_tree['weights'] = self.port(self.init_tree(), near=1)
        max_sharpe = -9999999
        sprint('Searching for the integer shares')
        n = 0
        flag = False
        near = 1
        while True:
            n += 1
            tree_list = []
            print(f'第{n}次迭代：')
            for i in tqdm(list(exam_tree.itertuples())):
                examed_sharpe = self.exam(i.weights)
                if examed_sharpe != 0:
                    tree_list.append([i.weights, examed_sharpe])
            df_exam = pd.DataFrame(tree_list, columns=['shares', 'sharpe']).sort_values(
                by='sharpe', ascending=False)
            # 引入过滤条件减少计算量
            df_exam = df_exam[df_exam['sharpe'] >= max_sharpe]
            if len(df_exam) == 1:
                return df_exam.iloc[0].to_dict()
            # 本次迭代最大sharpe
            max_sharpe = df_exam['sharpe'].iloc[0]
            print(
                f'max_sharpe:{max_sharpe}\nshares:{df_exam["shares"].iloc[0]}\n'+'-'*100)
            # 寻找下一个临近点
            if flag:
                near = 1
                flag = False
            elif n > 1:
                near = 2
                flag = True
            exam_tree = pd.DataFrame()
            exam_tree['weights'] = self.port(df_exam, near=near)

    def buy(self):
        '''
        最优整数解
        return a dict contained all above:
        sharpe --> 夏普比率
        weights --> 各股票权重
        cost --> 各股票购买成本
        shares --> 各股票购买手数
        cost --> 总成本
        '''
        result_dict = self.tree()
        result_dict['weights'] = {}
        cost_dict = {}
        for name in self.names:
            cost_dict[name] = result_dict['shares'][name] * \
                self.last_price[name]*100
        sum_cost = np.array(list(cost_dict.values())).sum()
        result_dict['weights'] = {name: cost /
                                  sum_cost for name, cost in cost_dict.items()}
        result_dict['cost'] = cost_dict
        result_dict['sum_cost'] = sum_cost
        if self.path:
            df_opt = pd.DataFrame(result_dict)
            df_opt['sharpe'] = None
            df_opt['sharpe'].iloc[0] = result_dict['sharpe']
            df_opt['sum_cost'] = None
            df_opt['sum_cost'].iloc[0] = result_dict['sum_cost']
            df_opt.columns = ['shares', 'sharpe',
                              'weights', 'cost', 'sum_cost']
            df_opt.to_csv(f'{self.path}\\max sharpe and weights (integer).csv')
        return result_dict

    def weight_tests(self, number=5):
        '''
        构建所有股票个数为number的组合
        '''
        lists = []
        for port in tqdm(list(combinations(self.names, number))):
            self.names = list(port)
            self.data = self.datas.loc[self.names].reset_index(drop=True)
            self.lens = len(self.names)
            test_dict = self.optimization()
            weight_array = np.array(list(test_dict['weights'].values()))
            test_dict['std'] = np.std(weight_array+1)
            test_dict['min'] = np.min(weight_array)
            test_dict['max'] = np.max(weight_array)
            lists.append(test_dict)
            if test_dict['min'] > 0.02:
                sprint(test_dict)
        df_test = pd.DataFrame(
            lists, columns=['weights', 'sharpe', 'std', 'min', 'max'])
        if self.path:
            df_test.to_csv(f'{self.path}\\weight_test.csv', index=False)
        return df_test


class CAPM(Markovitz):
    '''
    资产定价模型\n
    names=['贵州茅台', '隆基股份', '五粮液']\n
    start_date='2021-05-01'\n
    end_date='2021-11-01'\n
    rfr=0.023467/365\n
    market_index='沪深300指数'\n
    path --> 默认缓存路径为：".\\CAPM cache\\"，可传入False不缓存
    '''

    def __init__(self, names=['比亚迪', '阳光电源', '璞泰来', '紫光国微', '盛新锂能'],
                 start_date='2021-05-01',
                 end_date='2021-11-01',
                 frequency='w',
                 rfr=0.023467,
                 market_index='沪深300指数',
                 path='.\\CAPM cache\\'):
        self.names = names
        self.lens = len(names)
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.rfr = (rfr*100) / \
            {'d': 365, 'w': 52, 'm': 30}[frequency]
        self.market_index = market_index
        self.path = path
        if self.path:
            makedir(self.path, '')
        sprint('Initializing...')
        if not self.path:
            sd = StockData(names=self.names+[market_index], start_date=self.start_date,
                           end_date=self.end_date, frequency=self.frequency)
            self.datas = sd.stocks_data()
        else:
            try:
                self.datas = pd.read_csv(
                    f'{self.path}\\stock data\\stocks_data.csv')
            except:
                sd = StockData(names=self.names+[market_index], start_date=self.start_date,
                               end_date=self.end_date, frequency=self.frequency, path=self.path)
                self.datas = sd.stocks_data()

        self.datas.index = self.datas['name']
        self.data = self.datas.loc[self.names].reset_index(drop=True)
        self.Rm_data = self.datas.loc[self.market_index].reset_index(drop=True)

    def beta(self, Ri):
        '''
        按照定义计算Beta系数 --> 返回beta值
        '''
        Rm = self.Rm_data['pctChg']
        assert len(Ri) != len(Rm), '传入的数据序列的长度不等于市场指数序列的长度!'
        return (np.cov(Ri, Rm))[0][1]/np.var(Rm)

    def ls_beta(self, Ri):
        '''
        回归估计beta系数 --> 返回包含alpha和beta的字典
        '''
        y = Ri.reset_index(drop=True)
        x = sm.add_constant(self.Rm_data['pctChg'])
        model = regression.linear_model.OLS(y, x).fit()
        return {'alpha_ols': model.params['const'], 'beta_ols': model.params['pctChg']}

    def scl(self, name='', show=True):
        '''
        给定资产的证券特征线
        '''
        if name not in self.names:
            sprint(
                f'name参数值未给定，或参数值{name}不在给定风险资产范围内！已重新随机选择一种给定风险资产！', color='red')
            name = random.choice(self.names)
        Ri = self.data[self.data.name == name]['pctChg']
        Rm = self.Rm_data['pctChg']
        ls_dict = self.ls_beta(Ri)
        plt.cla()
        plt.axline(xy1=(0, ls_dict['alpha_ols']),
                   slope=ls_dict['beta_ols'], c='m')
        plt.scatter(Ri, Rm, s=10, marker=".", c='b')
        plt.xlabel(f'{name}收益率（%）')
        plt.ylabel(f'{self.market_index}收益率（%）')
        if show:
            plt.show()
        else:
            makedir(self.path, 'scl')
            plt.savefig(f'{self.path}\\scl\\{name}.svg', format='svg')

    def all_beta(self):
        '''
        回归估计beta系数、回归估计截距项alpha的值和按照定义计算beta系数汇总
        '''
        lists = []
        Rm_mean = self.Rm_data['pctChg'].mean()
        for name in self.names:
            pctChg_series = self.data[self.data.name == name]['pctChg']
            dic = self.ls_beta(pctChg_series)
            dic['beta'] = self.beta(pctChg_series)
            dic['Ri'] = pctChg_series.mean()
            dic['Rm'] = Rm_mean
            dic['name'] = name
            lists.append(dic)
        df_all_beta = pd.DataFrame(lists).set_index('name')
        if self.path:
            df_all_beta.to_csv(f'{self.path}\\beta.csv')
        return df_all_beta

    def sml(self, show=True):
        '''
        估计收益率在证券市场线上的分布
        '''
        plt.style.use('seaborn-paper')
        Rm_mean = self.Rm_data['pctChg'].mean()
        k = Rm_mean-self.rfr
        all_beta = self.all_beta()
        plt.cla()
        plt.axline(xy1=(0, self.rfr), slope=k, c='m')
        plt.scatter(all_beta.beta, all_beta.beta*k +
                    self.rfr, s=100, marker=".", c='b')
        plt.scatter(all_beta.beta, all_beta.Ri, s=100, marker=".", c='r')
        for i in all_beta.itertuples():
            plt.annotate(text=i.Index, xy=(
                i.beta, i.Ri), xytext=(i.beta, i.Ri))
            plt.annotate(text=i.Index, xy=(i.beta, i.beta*k +
                         self.rfr), xytext=(i.beta, i.beta*k + self.rfr))
        if all_beta.beta.min() < 0:
            plt.axvline(0, linestyle='--')
        else:
            plt.xlim(0, all_beta.beta.max()*1.2)
        plt.xlabel('Beta')
        plt.ylabel('E(Ri)')
        if show:
            plt.show()
        else:
            plt.savefig(f'{self.path}\\sml.svg', format='svg')

    def all(self):
        '''
        证券市场线 + 所有风险资产的特征线
        '''
        for name in self.names:
            self.scl(name, show=False)
        self.sml(show=False)


class Port(Markovitz):
    '''
    资产组合管理与评价
    names=['贵州茅台', '隆基股份', '五粮液']\n
    weights=False --> 不传入则根据历史数据自动计算，也可传入字典\n
    start_date='2021-05-01'\n
    end_date='2021-11-01'\n
    frequency='d' --> d/w/m\n
    rfr=0.023467/365\n
    market_index='沪深300指数'\n
    path --> 默认缓存路径为：".\\Port cache\\"，可传入False不缓存
    '''

    def __init__(self, names=['贵州茅台', '隆基股份', '五粮液'],
                 weights=False,
                 start_date='2021-05-01',
                 end_date='2021-11-01',
                 frequency='d',
                 rfr=0.023467,
                 market_index='沪深300指数',
                 path='.\\Port cache\\'
                 ):
        self.names = names
        self.lens = len(names)
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.rfr = rfr
        self.market_index = market_index
        self.path = path
        sprint('Initializing...')
        if not self.path:
            sd = StockData(names=self.names+[market_index], start_date=self.start_date,
                           end_date=self.end_date, frequency=self.frequency, path=self.path)
            self.datas = sd.stocks_data()
        else:
            try:
                self.datas = pd.read_csv(
                    f'{self.path}\\stock data\\stocks_data.csv')
            except:
                sd = StockData(names=self.names+[market_index], start_date=self.start_date,
                               end_date=self.end_date, frequency=self.frequency, path=self.path)
                self.datas = sd.stocks_data()

        self.datas.index = self.datas['name']
        self.data = self.datas.loc[self.names].reset_index(drop=True)
        self.Rm_data = self.datas.loc[self.market_index].reset_index(drop=True)
        self.date = list(map(lambda x: str(x)[:10], self.datas.date.unique()))
        if not weights:
            self.weights_dict = self.optimization()['weights']
        elif isinstance(weights, dict):
            if list(weights.keys()) != self.names:
                raise ValueError('参数weights的keys必须与names相同！')
            elif np.array(weights.values()).sum() != 1:
                weights = dict(
                    zip(self.names, [i/np.sum(list(weights.values())) for i in weights.values()]))
            self.weights_dict = weights
        else:
            raise ValueError('参数weights必须为dict！')

    def port_rate(self):
        '''
        按照各资产权重计算的组合收益率序列 --> 返回一个Series
        '''
        df_pctChg = self.data[['name', 'date', 'pctChg']]
        df_pctChg['weights'] = df_pctChg.name.map(self.weights_dict)
        df_pctChg['weighted_pctChg'] = df_pctChg['weights']*df_pctChg['pctChg']
        df = pd.DataFrame()
        for name in self.names:
            df[name] = list(df_pctChg[df_pctChg.name == name]
                            ['weighted_pctChg'])
        df.index = self.date
        if self.path:
            makedir(self.path, 'rate')
            df.sum(axis=1).to_csv(f'{self.path}\\rate\\port_rate_series.csv')
        return df.sum(axis=1)

    def mean(self):
        '''
        各风险资产的收益率期望
        '''
        return self.calculate()['mean']

    def port_mean(self):
        '''
        组合的收益率期望
        '''
        series = pd.Series({'port mean': self.port_rate().mean()})
        if self.path:
            makedir(self.path, 'rate')
            series.to_csv(f'{self.path}\\rate\\port_mean.csv')
        return self.port_rate().mean()

    def cov(self):
        '''
        各风险资产协方差矩阵
        '''
        df = self.calculate()['cov']
        if self.path:
            makedir(self.path, 'rate')
            df.to_csv(f'{self.path}\\rate\\cov.csv')
        return df

    def sharpe_ratio(self):
        '''
        组合的夏普比率
        '''
        return -self.sharpe(weights=list(self.weights_dict.values()))

    def ask_returns(self, rate=0.05):
        '''
        计算给定收益率对应组合的风险、权重和夏普比率 --> 返回一个包含收益率、标准差、权重和夏普比率的字典
        '''
        return self.get_return(rate=rate)

    def max_loss(self, level=0.05):
        '''
        在5%显著性水平下，组合中每种风险资产的最大跌幅
        '''
        df_pctChg = self.data[['name', 'date', 'pctChg']]
        loss_dict = dict(zip(self.names, [abs(np.percentile(
            df_pctChg[df_pctChg.name == name]['pctChg'], level*100)) for name in self.names]))
        loss_series = pd.Series(loss_dict)
        if self.path:
            makedir(self.path, 'ask')
            loss_series.to_csv(
                f'{self.path}\\ask\\max_loss(level={level}).csv')
        return loss_series

    def port_max_loss(self, level=0.05):
        '''
        在5%显著性水平下，基金的最大损失跌幅
        '''
        return abs(np.percentile(self.port_rate(), level*100))

    def drawdown(self, show=True):
        '''
        组合和各风险资产的最大回撤
        '''
        df_rate = pd.DataFrame()
        df_rate['port'] = self.port_rate()
        df_rate[self.market_index] = list(self.Rm_data['pctChg'])
        data = self.data[['name', 'date', 'pctChg']]
        for name in self.names:
            df_rate[name] = list(data[data.name == name]['pctChg'])
        del data
        # 构建财富指数
        wealth = (1+df_rate/100).cumprod()
        # 找出上一个最高点
        previous_max = wealth.cummax()
        # 回撤率
        draw_down = (wealth-previous_max)/previous_max
        # 折线图
        if show:
            wealth.plot()
            previous_max.plot()
            draw_down.plot()
            plt.show()
        else:
            makedir(self.path, 'drawdown')
            wealth.plot()
            plt.savefig(f'{self.path}\\drawdown\\wealth.svg', format='svg')
            previous_max.plot()
            plt.savefig(
                f'{self.path}\\drawdown\\previous_max.svg', format='svg')
            draw_down.plot()
            plt.savefig(f'{self.path}\\drawdown\\draw_down.svg', format='svg')
        if self.path:
            makedir(self.path, 'drawdown')
            draw_down.min().to_csv(f'{self.path}\\drawdown\\max drawdown.csv')
        return draw_down.min()

    def kline(self):
        pl = PortKline(weights=self.weights_dict, path=self.path+'\\kline')
        pl.kline()


class PortKline(object):

    def __init__(self, weights={'贵州茅台': 0.5,
                                '隆基股份': 0.2,
                                '五粮液': 0.3},
                 path='.\\Kline cache\\'
                 ):
        self.names = list(weights.keys())
        self.weights = list(weights.values())
        self.lens = len(self.names)
        self.path = path

    def stock_weights(self):
        return dict(zip(self.names, [i/sum(self.weights) for i in self.weights]))

    def stock_data(self):
        if not self.path:
            sd = StockData(names=self.names, path=self.path)
            sd.start_date = sorted(
                [i['ipoDate'] for i in sd.stocks_info().values()])[-1]
            sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
            return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]
        else:
            try:
                stocks_info = pd.read_csv(
                    f'{self.path}\\kline\\stock data\\stocks_info.csv').set_index('name').T.to_dict()
                sd = StockData(names=self.names, path=self.path)
                sd.start_date = sorted([i['ipoDate']
                                       for i in stocks_info.values()])[-1]
                sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
                return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]
            except:
                sd = StockData(names=self.names, path=self.path)
                sd.start_date = sorted(
                    [i['ipoDate'] for i in sd.stocks_info().values()])[-1]
                sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
                return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]

    def get_data(self):
        data = self.stock_data()
        data['weight'] = data['name'].map(self.stock_weights())
        data['real_weight'] = data.weight/data.open
        date = data.date.unique()
        x = [list(data[data.date == i].real_weight /
                  data[data.date == i].real_weight.sum()) for i in date]
        data = data.sort_values(by=['date', 'name'])
        data['z_weight'] = reduce(add, tuple(x))
        data.date = data.date.astype(str)
        data.open = (data.open*data.z_weight).map(lambda x: round(x, 2))
        data.close = (data.close*data.z_weight).map(lambda x: round(x, 2))
        data.low = (data.low*data.z_weight).map(lambda x: round(x, 2))
        data.high = (data.high*data.z_weight).map(lambda x: round(x, 2))
        data.volume = (data.volume*data.z_weight).map(lambda x: round(x, 0))
        data = data.groupby('date').sum()
        data['date'] = data.index
        return data[['date', 'open', 'close', 'low', 'high', 'volume']]

    def calculate(self):
        data = self.get_data()
        # 计算EMA(12)和EMA(16)
        data['EMA12'] = data['close'].ewm(alpha=2 / 13, adjust=False).mean()
        data['EMA26'] = data['close'].ewm(alpha=2 / 27, adjust=False).mean()
        # 计算DIFF、DEA、MACD
        data['DIFF'] = data['EMA12'] - data['EMA26']
        data['DEA'] = data['DIFF'].ewm(alpha=2 / 10, adjust=False).mean()
        data['MACD'] = 2 * (data['DIFF'] - data['DEA'])
        data['DIFF'] = data['DIFF'].map(lambda x: round(x, 2))
        data['DEA'] = data['DEA'].map(lambda x: round(x, 2))
        data['MACD'] = data['MACD'].map(lambda x: round(x, 2))
        # 上市首日，DIFF、DEA、MACD均为0
        data['DIFF'].iloc[0] = 0
        data['DEA'].iloc[0] = 0
        data['MACD'].iloc[0] = 0
        data['datas'] = data[['open', 'close', 'low', 'high', 'volume',
                              'EMA12', 'EMA26', 'DIFF', 'DEA', 'MACD']].values.tolist()
        return data

    def calculate_ma(self, day_count, data: int):
        result: List[Union[float, str]] = []
        for i in range(len(data["date"])):
            if i < day_count:
                result.append("-")
                continue
            sum_total = 0.0
            for j in range(day_count):
                sum_total += float(data["datas"][i - j][1])
            result.append(abs(float("%.2f" % (sum_total / day_count))))
        return result

    def kline(self):
        data = self.calculate().to_dict('list')
        kline = (
            Kline()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="portfolio index",
                y_axis=data["datas"],
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#ef232a",
                    color0="#14b143",
                    border_color="#ef232a",
                    border_color0="#14b143",
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False, type_="inside", xaxis_index=[0, 0], range_end=100
                    ),
                    opts.DataZoomOpts(
                        is_show=True, xaxis_index=[0, 1], pos_top="97%", range_end=100
                    ),
                    opts.DataZoomOpts(is_show=False, xaxis_index=[
                                      0, 2], range_end=100),
                ],
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=1,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
                brush_opts=opts.BrushOpts(
                    x_axis_index="all",
                    brush_link="all",
                    out_of_brush={"colorAlpha": 0.1},
                    brush_type="lineX",
                ),
                # 三个图的 axis 连在一块
                axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )
        )

        kline_line = (
            Line()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="MA5",
                y_axis=self.calculate_ma(day_count=5, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA10",
                y_axis=self.calculate_ma(day_count=10, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA20",
                y_axis=self.calculate_ma(day_count=20, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA30",
                y_axis=self.calculate_ma(day_count=30, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=1,
                    split_number=3,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )
        # Overlap Kline + Line
        overlap_kline_line = kline.overlap(kline_line)

        # Bar-1
        bar_1 = (
            Bar()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="Volumn",
                y_axis=data["volume"],
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                    function(params) {
                        var colorList;
                        if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # Bar-2 (Overlap Bar + Line)
        bar_2 = (
            Bar()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="MACD",
                y_axis=data["MACD"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                            function(params) {
                                var colorList;
                                if (params.data >= 0) {
                                colorList = '#ef232a';
                                } else {
                                colorList = '#14b143';
                                }
                                return colorList;
                            }
                            """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=2,
                    split_number=4,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        line_2 = (
            Line()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="DIFF",
                y_axis=data["DIFF"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="DEA",
                y_axis=data["DEA"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )
        # 最下面的柱状图和折线图
        overlap_bar_line = bar_2.overlap(line_2)

        # 最后的 Grid
        grid_chart = Grid(init_opts=opts.InitOpts(
            width="1500px", height="750px"))

        # 这个是为了把 data.datas 这个数据写入到 html 中,还没想到怎么跨 series 传值
        # demo 中的代码也是用全局变量传的
        grid_chart.add_js_funcs("var barData = {}".format(data["datas"]))

        # K线图和 MA5 的折线图
        grid_chart.add(
            overlap_kline_line,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", height="60%"),
        )
        # Volumn 柱状图
        grid_chart.add(
            bar_1,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
            ),
        )
        # MACD DIFS DEAS
        grid_chart.add(
            overlap_bar_line,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="82%", height="14%"
            ),
        )
        makedir(self.path, 'kline')
        grid_chart.render(
            f"{self.path}\\kline\\{str(self.names)}_{time.strftime('%Y-%m-%d', time.localtime())}.html")


if __name__ == '__main__':
    from pprint import pprint
    # fund = Port(weights={'贵州茅台': 0.3, '隆基股份': 0.2, '五粮液': 0.5})
    # pprint(fund.port_mean())

    # capm = CAPM(names=['安科瑞', '紫光国微', '航发控制'],  # 股票组合
    #             start_date='2020-11-01',  # 开始日期
    #             end_date='2021-11-01',  # 结束日期
    #             frequency='w',  # 'd' or 'w' or 'm' → 日、周、月
    #             rfr=0.023467,  # 无风险利率
    #
    mk = Markovitz(names=['比亚迪', '阳光电源', '璞泰来', '紫光国微', '盛新锂能'],  # 股票组合
                   start_date='2021-05-01',  # 开始日期
                   end_date='2021-11-01',  # 结束日期
                   frequency='w',  # 'd' or 'w' or 'm' → 日、周、月
                   rfr=0.023467,  # 无风险利率
                   funds=10000000,  # 最大资金限制
                   )
    # mk.heatmap(show=False)
    # mk.cml()
    pprint(mk.init_port())
