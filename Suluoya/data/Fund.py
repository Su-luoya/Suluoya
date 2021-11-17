import os
import sys

import pandas as pd
import pretty_errors
from tqdm import tqdm, trange


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..log.log import slog, sprint, hide, show
except:
    from log.log import slog, sprint, hide, show


class FundData(object):
    def __init__(self):
        self.main_url = 'http://quotes.money.163.com/old/#FN'
    
    def shares_holding(self, code='011146'):
        '''
        基金持仓数据
        '''
        df = pd.read_html(f'http://quotes.money.163.com/fund/cgmx_{code}.html')
        return {
            '重仓持股': df[0],
            '本期新进': df[1],
            '本期退出': df[2],
            '本期增持': df[3],
            '本期减持': df[4]
        }


if __name__ == '__main__':
    from pprint import pprint
    fd = FundData()
    test = fd.shares_holding()
    pprint('http://quotes.money.163.com/old/#query=stock')
