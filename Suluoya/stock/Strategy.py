import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pretty_errors
import scipy.optimize as sco
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..log.log import slog, sprint, hide, show
    from .Stock import StockData, ConstituentStocks
except:
    from log.log import slog, sprint, hide, show
    from Stock import StockData, ConstituentStocks

# Stock Selection


class LowValuation(object):
    def __init__(self, industry='银行', compare_stocks=['中证银行','沪深300指数', ],
                 start_date='2019-01-01',
                 end_date='2020-03-01',):
        sprint('Please make sure your industry is present in the market!')
        stock_industry = ConstituentStocks().stock_industry()
        self.start_date = start_date
        self.end_date = end_date
        self.names = stock_industry[stock_industry['industry']
                                    == industry]['code_name'][0:2]
        self.compare_stocks = compare_stocks
        sprint('Initializing...')
        global StockData
        # stock_data = StockData(names=self.names, start_date=self.start_date,
        #                        end_date=self.end_date)
        # self.stocks_valuation = stock_data.stocks_valuation()[['name', 'date', 'close', 'peTTM']]
        # self.dates = self.stocks_valuation.date.unique()
        compare_stocks_data = StockData(names=self.compare_stocks, start_date=self.start_date,
                                        end_date=self.end_date)
        self.compare_stocks_data = compare_stocks_data.stocks_data()


if __name__ == '__main__':
    lv = LowValuation()
    test = lv.compare_stocks_data
    print(test)
