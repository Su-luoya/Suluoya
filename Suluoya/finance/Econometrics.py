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


class Regression(object):
    
    def __init__(self,df):
        self.df = df
        