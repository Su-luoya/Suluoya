import json
import os
import re
import sys
import time
from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Stock import StockData, ConstituentStocks
    from ..log.log import hide, show, slog, sprint
    from ..data.Code import stock_pair
except:
    from log.log import hide, show, slog, sprint
    from data.Stock import StockData, ConstituentStocks
    from data.Code import stock_pair


class IndustryAnalysis(object):
    '''
    同业比较
    '''

    def __init__(self, names=['贵州茅台', '隆基股份']):
        self.names = names
        self.codes = {i: j for i, j in stock_pair(names=names).items()}.values()
        self.url = 'http://f10.eastmoney.com/IndustryAnalysis/IndustryAnalysisAjax?'
        self.headers = {
            "Host": "f10.eastmoney.com",
            "Referer": "http://f10.eastmoney.com/IndustryAnalysis/Index?type=web&code=SZ300059",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.128 Safari/537.36 Edg/89.0.774.77",
            "X-Requested-With": "XMLHttpRequest"
        }

    def params_list(self, codes):
        params_list = []
        for code in codes:
            params_list.append({"code": code[0:2].upper()+code[3:],
                                "icode": "447"})
        return params_list

    def request(self, params):
        response = requests.get(
            self.url, headers=self.headers, params=params, timeout=100)
        response.encoding = response.apparent_encoding
        return json.loads(response.text)

    def get_data(self):
        data_list = []
        for params in tqdm(self.params_list(codes=self.codes)):
            data_list.append(self.request(params=params))
        return data_list

    @property
    def info(self):
        '''
        凡人，休得僭越！
        '''
        sprint('Getting industry analysis data...')
        industry_info = []
        growth_info = []
        valuation_info = []
        dupont_info = []
        market_size = []
        i = 0
        for data in self.get_data():
            industry_info.append({self.names[i]: data['hyzx']})  # 行业资讯
            growth_info.append(data['czxbj']['data'])  # 成长性比较
            valuation_info.append({self.names[i]: data['gzbj']['data']})  # 估值
            dupont_info.append({self.names[i]: data['dbfxbj']['data']})  # 杜邦
            market_size.append(
                {self.names[i]+'——'+'按总市值排名': data['gsgmzsz']})  # 总市值
            market_size.append(
                {self.names[i]+'——'+'按流通市值排名': data['gsgmltsz']})  # 流通市值
            market_size.append(
                {self.names[i]+'——'+'按营业收入排名': data['gsgmyysr']})  # 营业收入
            market_size.append(
                {self.names[i]+'——'+'按净利润排名': data['gsgmjlr']})  # 净利润
            i += 1
        return {
            'industry_info': industry_info,
            'growth_info': growth_info,
            'valuation_info': valuation_info,
            'dupont_info': dupont_info,
            'market_size': market_size,
        }

    def industry_info(self):
        '''
        企业资讯
        '''
        lists = []
        for i in self.info['industry_info']:
            for j, k in i.items():
                for l in k:
                    lists.append([j, l['date'], l['title']])
        return pd.DataFrame(lists, columns=['stock', 'date', 'advisory'])

    def growth_info(self):
        '''
        成长性比较
        '''
        lists = []
        for i in self.info['growth_info']:
            lists.append([i[0]['jc']])
            for j in i:
                lists.append(j.values())
        T1 = ['基本每股收益增长率(%)', '营业收入增长率(%)', '净利润增长率(%)']
        T2 = ['3年复合', '19A', 'TTM', '20E', '21E', '22E']
        columns = ['排名', '代码', '简称']
        for t1 in T1:
            for t2 in T2:
                columns.append(t2+'--'+t1)
        return pd.DataFrame(lists, columns=columns)

    def valuation_info(self):
        '''
        估值比较\n
        (1)MRQ市净率=上一交易日收盘价/最新每股净资产\n
        (2)市现率①=总市值/现金及现金等价物净增加额\n
        (3)市现率②=总市值/经营活动产生的现金流量净额
        '''
        columns = ['排名', '代码', '简称', 'PEG']
        T = {'市盈率': ['19A', 'TTM', '20E', '21E', '22E'],
             '市销率': ['19A', 'TTM', '20E', '21E', '22E'],
             '市净率': ['19A', 'MRQ'],
             '市现率①': ['19A', 'TTM'],
             '市现率②': ['19A', 'TTM'],
             'EV/EBITDA': ['19A', 'TTM']}
        for i, j in T.items():
            for k in j:
                columns.append(k+'--'+i)
        lists = []
        for i in self.info['valuation_info']:
            lists.append(i.keys())
            for j in i.values():
                for k in j:
                    lists.append(k.values())
        return pd.DataFrame(lists, columns=columns)

    def dupont_info(self):
        '''
        杜邦分析比较
        '''
        T1 = ['ROE(%)', '净利率(%)', '总资产周转率(%)', '权益乘数(%)']
        T2 = ['3年平均', '17A', '18A', '19A']
        columns = ['排名', '代码', '简称']
        for t1 in T1:
            for t2 in T2:
                columns.append(t2+'--'+t1)
        lists = []
        for i in self.info['dupont_info']:
            lists.append(i.keys())
            for j in i.values():
                for k in j:
                    lists.append(k.values())
        return pd.DataFrame(lists, columns=columns)

    def market_size(self):
        '''
        公司规模
        '''
        columns = ['排名', '代码', '简称',
                   '总市值(元)', '流通市值(元)', '营业收入(元)', '净利润(元)', '报告期']
        lists = []
        for i in self.info['market_size']:
            lists.append(i.keys())
            for j in i.values():
                for k in j:
                    lists.append(k.values())
        return pd.DataFrame(lists, columns=columns)


class FinancialStatements(object):
    '''
    会计报表和主要财务指标
    '''

    def __init__(self, names=['贵州茅台', '隆基股份']):
        self.names = names
        self.code_dict = {i: j[3:] for i, j in stock_pair(names=names).items()}
        self.main_urls = [
            f'http://quotes.money.163.com/service/zycwzb_{i}.html?type=report' for i in self.code_dict.values()]
        self.statement_urls = [
            f'http://quotes.money.163.com/service/cwbbzy_{i}.html' for i in self.code_dict.values()]

    def request(self, urls):
        '''
        大胆！不准看！
        '''
        sprint('Getting data...')
        result_list = []
        n = 0
        for url in tqdm(urls):
            response = requests.get(url, timeout=100)
            response.encoding = response.apparent_encoding
            data = response.text.replace(
                '\r', '').replace('\t', '').split('\n')
            df = pd.DataFrame(
                [i.split(',') for i in data if i != '']).set_index(0).T
            df['名称'] = self.names[n]
            n += 1
            result_list.append(df)
        df = pd.concat(result_list)
        return df

    def summary(self):
        '''财务报表摘要'''
        return self.request(self.statement_urls)

    def balance(self):
        '''资产负债表'''
        return self.request([i.replace('cwbbzy', 'zcfzb') for i in self.statement_urls])

    def income(self):
        '''利润表'''
        return self.request([i.replace('cwbbzy', 'lrb') for i in self.statement_urls])

    def cash_flow(self):
        '''现金流量表'''
        return self.request([i.replace('cwbbzy', 'xjllb') for i in self.statement_urls])

    def main_info(self):
        '''主要财务指标'''
        return self.request(self.main_urls)

    def profit_info(self):
        '''盈利能力指标'''
        return self.request([i+'&part=ylnl' for i in self.main_urls])

    def debt_info(self):
        '''偿债能力指标'''
        return self.request([i+'&part=chnl' for i in self.main_urls])

    def growth_info(self):
        '''成长能力指标'''
        return self.request([i+'&part=cznl' for i in self.main_urls])

    def operation_info(self):
        '''营运能力指标'''
        return self.request([i+'&part=yynl' for i in self.main_urls])

    def save(self, data, file_name='data', path='./'):
        '''
        按照名称分类保存数据在excel不同的sheet中\n
        data --> 传入本类中任意方法的返回数据\n
        file_name --> 保存的excel名称\n
        path --> 保存路径，默认在当前文件夹
        '''
        writer = pd.ExcelWriter(path+f'{file_name}.xlsx')
        for name in self.names:
            df = data[data['名称'] == name]
            del df['名称']
            df.to_excel(writer, name, index=False)
        writer.save()




if __name__ == '__main__':
    from pprint import pprint
    ia = IndustryAnalysis(names=['贵州茅台', '隆基股份'])
    print(ia.dupont_info())
