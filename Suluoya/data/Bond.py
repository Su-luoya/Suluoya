import requests
import json
from time import time
import pandas as pd


class BondData(object):
    def __init__(self):
        self.url = 'https://xueqiu.com/service/v5/stock/screener/quote/list'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.56',
            'X-Requested-With': 'XMLHttpRequest',
        }

    def response(self, mode='可转债'):
        '''
        凡人，休得僭越！
        '''
        dic = {'可转债': 'convert', '国债': 'national', '企债': 'corp'}
        params = {
            "page": "1",
            "size": "1000",
            "order": "desc",
            "order_by": "percent",
            "industry": mode,
            "type": dic[mode],
            "_": str(time())
        }
        resp = requests.request(
            "GET", self.url, headers=self.headers, params=params)
        data = json.loads(resp.text)['data']
        df = pd.DataFrame(data['list'])
        return df

    def convertible_bond(self):
        """可转债
        """
        df = self.response('可转债')
        columns = ['代码', '正股价', 'type', '涨跌幅', 'has_follow', 'tick_size', '正股涨跌幅', '税前收益', '当前价', '正股简称', 'lot_size', '转股价值', '涨跌幅1',
                   '正股净资产', 'maturity_time', 'interest_memo', '转股价', '正股代码', '正股市净率', '剩余年限', '转债规模(万)', '税后收益', '名称', '转债占比', '溢价率', 'putback_price']
        df.columns = columns
        return df

    def corporate_bond(self):
        """企业债
        """
        df = self.response('企债')
        columns = ['债券代码', '涨跌额', '剩余年限', '债项评级', '到期时间', 'type', '涨跌幅',
                   'has_foliow', 'tick_size', '成交量', '当前价', '票面利息', '债券名称', 'lot_size']
        df.columns = columns
        return df

    def national_bond(self):
        """国债
        """
        df = self.response('国债')
        columns = ['债券代码', '涨跌额', '当前价', 'has_follow', 'lot_size', '到期时间',
                   '债券名称', '涨跌幅', '债项评级', '债券代码', '剩余年限', 'tick_size', 'type', '成交量']
        df.columns = columns
        return df





if __name__ == '__main__':
    b = BondData()
    print(b.convertible_bond())
