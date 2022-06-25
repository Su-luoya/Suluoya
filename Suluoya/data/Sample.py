import pandas as pd
import numpy as np
import datetime
import random
import pretty_errors
from sklearn import datasets


class RegressionData(object):
    
    def __init__(self):
        pass

    def iris(self):
        load = datasets.load_iris()
        data = load.data
        target = load.target
        feature = load.feature_names
        df = pd.DataFrame(data, columns = feature)
        df['target'] = target
        return df

    def boston(self):
        load = datasets.load_boston()
        data = load.data
        target = load.target
        feature = load.feature_names
        df = pd.DataFrame(data, columns=feature)
        df['target'] = target
        return df


class TimeData(object):

    def __init__(self, size=100):
        self.size = size

    def date_series(self, freq='D'):
        return pd.date_range(datetime.date.today(), periods=self.size, freq=freq)

    def data_series(self):
        data_list = [random.random() for i in range(self.size)]
        return pd.Series(data_list)

    def time_frame(self, columns=['A', 'B', 'D'], categories=['c1', 'c2']):
        df = pd.DataFrame()
        for column in columns:
            df[column] = self.data_series()
        if len(categories) > 0:
            for category in categories:
                df[category] = pd.Series([random.choice(
                    ['a', 'b', 'c']) for i in range(self.size)], dtype = "category")
        df['date']=self.date_series()
        return df.set_index('date')


if __name__ == '__main__':
    ts=TimeData()
    print(ts.time_frame())
