import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Stock import ConstituentStocks
    from ..log.log import hide, show, slog, sprint
except:
    from log.log import hide, show, slog, sprint
    from data.Stock import ConstituentStocks


def stock_pair(names=['贵州茅台','隆基股份']):
    cs = ConstituentStocks()
    stock_industry = cs.stock_industry()
    df = stock_industry[['code','code_name']].set_index('code_name')
    return df.loc[names].to_dict()['code']
    
if __name__ == '__main__':
    sp = stock_pair()
    print(sp)
    
