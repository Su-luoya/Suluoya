from scipy import stats
from matplotlib.dates import MONDAY, DateFormatter, WeekdayLocator
import pylab as mpl  # 导入中文字体，避免显示乱码
import pretty_errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import itertools
import os
import sys
import warnings
warnings.filterwarnings("ignore")


plt.rcParams['font.sans-serif'] = ['Simhei']  # 解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Sample import TimeData
    from ..data.Stock import StockData
    from ..log.log import hide, makedir, progress_bar, show, slog, sprint
except:
    from data.Sample import TimeData
    from data.Stock import StockData
    from log.log import hide, makedir, progress_bar, show, slog, sprint


def data_check(df=None, date=None):
    if df is None:
        global TimeData
        TimeData = TimeData()
        df = TimeData.time_frame()
        date = df.index
    elif isinstance(df, pd.DataFrame):
        if date is None:
            date = pd.Series(list(range(len(df))))
            df = df.set_index(date)
        elif isinstance(date, (list, pd.Series)) or date in df.columns:
            df = df.set_index(date)
        else:
            TypeError('date must be a list, series or a column name!')
    else:
        raise TypeError('df must be a pandas DataFrame!')
    return df, date


class Chart(object):

    def __init__(self, df=None, date=None):
        self.df, self.date = data_check(df, date)
        self.columns = self.df.columns
        self.dtypes = self.df.dtypes
        self.category_var = self.dtypes[self.dtypes == 'category']
        self.size = len(self.df)
        self.if_setting = False

    def setting(self, date_type='%Y-%m-%d', rotation=False, stacked=False, subplots=False, sharex=True):
        self.rotation = rotation
        self.stacked = stacked
        self.subplots = subplots
        self.sharex = sharex
        self.date_type = date_type
        self.if_setting = True

    def combinations(self, columns):
        return list(itertools.combinations(columns, 2))

    def plot(self, columns=['A', 'B'], kind='line', label=None):
        '''line, hist, box, area, bar, scatter, kde(Kernel Density Est) chat'''
        if not self.if_setting:
            self.setting()
        if set(columns) <= set(self.columns):
            pass
        else:
            columns = self.columns
        if kind in ['line', 'scatter', 'area']:
            fig, ax = plt.subplots()
            self.df[columns].plot(
                kind=kind, ax=ax, x_compat=True, subplots=self.subplots, sharex=self.sharex)
        elif kind in ['scatter', 'hexbin']:
            if label is None or kind == 'hexbin':
                for c in self.combinations(columns):
                    self.df[columns].plot(
                        kind=kind, x=c[0], y=c[1], gridsize=int(self.size/10))
            elif label not in self.category_var:
                raise ValueError(
                    'The data frame must contain the label variable!')
            elif label in self.category_var:
                for c in self.combinations(columns):
                    self.df.plot.scatter(x=c[0], y=c[1], c=label)
        else:
            fig, ax = plt.subplots()
            try:
                ax.set_xticklabels([x.strftime(self.date_type)
                                    for x in self.df.index], rotation=self.rotation)
            except:
                pass
            self.df[columns].plot(
                kind=kind, stacked=self.stacked, ax=ax, subplots=self.subplots, sharex=self.sharex)
        plt.show()

    def series(self, column='A', kind='QQ'):
        '''QQ, series'''
        if kind == 'QQ':
            stats.probplot(self.df[column], dist="norm", plot=plt)
        elif kind == 'series':
            self.df[column].plot(x_compat=True, legend=True)
        plt.show()


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randn(100, 2), columns=['A', 'B']).cumsum()
    cs = Chart()
    # print(df)
    # cs.setting(subplots=False, stacked=True)
    # cs.plot(columns=['A', 'B'], kind='hist', )
    cs.series(kind='series')
