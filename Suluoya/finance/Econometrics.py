from scipy import stats
import statsmodels.api as sm
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
from sklearn import datasets
warnings.filterwarnings("ignore")


plt.rcParams['font.sans-serif'] = ['Simhei']  # 解决中文显示问题，目前只知道黑体可行
plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Sample import TimeData, RegressionData
    from ..data.Stock import StockData
    from ..log.log import hide, makedir, progress_bar, show, slog, sprint
    from .Chart import Chart
except:
    from data.Sample import TimeData, RegressionData
    from data.Stock import StockData
    from log.log import hide, makedir, progress_bar, show, slog, sprint
    from Chart import Chart





class Regression(object):

    def __init__(self, df):
        self.df = df
    
    def cov(self):
        return self.df.cov()
    

class TimeSeries(object):
    
    def __init__(self,series=None):
        self.series = series


if __name__ == '__main__':
    td = TimeData()
    series = td.time_frame(['A'])
    ts = TimeSeries(series=series)
    print(ts.series)
