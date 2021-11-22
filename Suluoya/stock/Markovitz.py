import warnings
import math
import os
import random
import sys
import time
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_errors
import scipy.optimize as sco
import seaborn as sns
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Stock import StockData
    from ..log.log import hide, makedir, progress_bar, show, slog, sprint
except:
    from data.Stock import StockData
    from log.log import hide, makedir, progress_bar, show, slog, sprint

warnings.filterwarnings('ignore')


class Markovitz(object):
    '''
    组合投资权重
    names=['贵州茅台', '隆基股份', '五粮液']
    start_date='2021-05-01'
    end_date='2021-11-01'
    no_risk_rate=0.023467/365
    funds=10000000
    path --> 默认缓存路径为：".\\Suluoya cache\\"，可传入False不缓存
    '''

    def __init__(self, names=['比亚迪', '阳光电源', '璞泰来', '紫光国微', '盛新锂能'],
                 start_date='2021-05-01',
                 end_date='2021-11-01',
                 frequency='d',
                 no_risk_rate=0.023467/365,
                 funds=10000000,
                 path='.\\Suluoya cache\\'):
        self.names = names
        self.lens = len(names)
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.no_risk_rate = no_risk_rate*100
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
        # 收益率 均值
        data_mean = data.groupby('name').mean().T[self.names]
        # 收益率 协方差矩阵和相关系数矩阵
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

    def heatmap(self):
        '''
        收益率相关系数热力图
        '''
        if self.path:
            try:
                data_corr = pd.read_csv(f'{self.path}\\mean,cov,corr\\data_corr.csv').rename(
                    {'Unnamed: 0', 'correlation'}).set_index('correlation')
            except:
                data = self.data[['name', 'pctChg']]
                df = pd.DataFrame()
                for name in self.names:
                    df[name] = list(data[data['name'] == name]['pctChg'])
                data_corr = df.corr()
                data_corr.to_csv(f'{self.path}\\mean,cov,corr\\data_corr.csv')
        else:
            data = self.data[['name', 'pctChg']]
            df = pd.DataFrame()
            for name in self.names:
                df[name] = list(data[data['name'] == name]['pctChg'])
            data_corr = df.corr()
        plt.rcParams['font.sans-serif'] = ['FangSong']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        plt.rcParams['font.size'] = 13
        plt.subplots(figsize=(9, 9))
        sns.heatmap(data_corr, annot=True, vmax=1, square=True, cmap='Purples')
        plt.show()

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
        return (self.no_risk_rate-rate)/risk  # 相反数

    def optimization(self):
        '''
        非线性规划求解最大夏普比率
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
        df_scatter['weights'] = pd.Series(
            map(lambda x: str(x), self.weights(number=number)))
        # 风险
        df_scatter['risk'] = np.sqrt(np.diagonal(
            weights.dot(data_cov).dot(weights.T)))
        # 收益率
        df_scatter['rate'] = data_mean.dot(weights.T).T['pctChg']
        # 夏普比率
        df_scatter['sharpe'] = (
            df_scatter.rate-self.no_risk_rate)/df_scatter.risk
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
            df_boundary_scatter.rate-self.no_risk_rate)/df_boundary_scatter.risk
        df_boundary_scatter = df_boundary_scatter.sort_values(
            by='sharpe', ascending=False)
        if self.path:
            makedir(self.path, 'scatter data')
            df_boundary_scatter.to_csv(
                f'{self.path}\\scatter data\\boundary_scatter_data.csv')
        return df_boundary_scatter

    def drawing(self):
        '''
        有效边界散点图
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
        plt.style.use('seaborn-paper')
        plt.scatter(df_scatter.risk, df_scatter.rate,
                    s=10, marker=".", c='b')
        plt.scatter(df_boundary_scatter.risk, df_boundary_scatter.rate,
                    s=10, marker=".", c='r')
        plt.axline(xy1=(0, self.no_risk_rate), slope=max_sharpe, c='m')
        plt.xlim(df_scatter.risk.min()*0.8, df_scatter.risk.max()*1.2)
        plt.ylim(df_scatter.rate.min()*0.8, df_scatter.rate.max()*1.2)
        plt.xlabel('Risk')
        plt.ylabel('Yield')
        plt.show()

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

    def near_port_constructor(self, shares_dict={'迎驾贡酒': 1, '明微电子': 2, '健民集团': 3}):
        '''
        构建临近整数组合 --> DataFrame
        '''
        lists = [shares_dict]
        for name, number in shares_dict.items():
            shares_dict_copy_1 = shares_dict.copy()
            shares_dict_copy_2 = shares_dict.copy()
            if number == 0:
                shares_dict_copy_1[name] = number+1
                lists.append(shares_dict_copy_1)
                continue
            lists.append(shares_dict_copy_1)
            lists.append(shares_dict_copy_2)
            shares_dict_copy_1[name] = number-1
            shares_dict_copy_2[name] = number+1
        return pd.DataFrame(lists)

    def port(self, df_shares):
        '''
        临近整数组合 --> DataFrame
        '''
        port_list = []
        # 寻找每一个检验剩下的组合的临近组合
        for i in df_shares.itertuples():
            port_list.append(self.near_port_constructor(i.shares))
        # 拼接所有临近整数解
        return pd.Series(pd.concat(port_list).drop_duplicates().reset_index(drop=True).T.to_dict())

    def tree(self):
        '''
        分枝定界
        返回最优整数解和sharpe
        '''
        # 初始整数组合
        exam_tree = pd.DataFrame()
        exam_tree['weights'] = self.port(self.init_tree())
        max_sharpe = -9999999
        sprint('Searching for the integer shares')
        n = 0
        while True:
            n += 1
            tree_list = []
            for i in exam_tree.itertuples():
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
                f'第{n}次迭代\nmax_sharpe:{max_sharpe}\nshares:{df_exam["shares"].iloc[0]}\n'+'-'*100)
            exam_tree = pd.DataFrame()
            exam_tree['weights'] = self.port(df_exam)

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


if __name__ == '__main__':
    from pprint import pprint
    mk = Markovitz(names=['口子窖', '安科瑞', '紫光国微', '航发控制', '隆基股份'],  # 股票组合
                   start_date='2020-11-01',  # 开始日期
                   end_date='2021-11-01',  # 结束日期
                   frequency='d',
                   no_risk_rate=0.023467/365,  # 无风险利率
                   funds=10000000,  # 最大资金限制
                   path='cache'
                   )
    pprint(mk.drawing())
