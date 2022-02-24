import pandas as pd
import numpy as np
import datetime
import random
import pretty_errors


class TimeData(object):

    def __init__(self, size=100):
        self.size = size

    def date_series(self, freq='D'):
        return pd.date_range(datetime.date.today(), periods=self.size, freq=freq)

    def data_series(self):
        data_list = [random.random() for i in range(self.size)]
        return pd.Series(data_list)

    def time_frame(self, columns=['A', 'B', 'D']):
        df = pd.DataFrame()
        for column in columns:
            df[column] = self.data_series()
        df['c1'] = pd.Series([random.choice(
            ['a', 'b', 'c']) for i in range(self.size)], dtype = "category")
        df['c2'] = pd.Series([random.choice(
            ['a', 'b', 'c']) for i in range(self.size)], dtype = "category")
        df['date']=self.date_series()
        return df.set_index('date')


if __name__ == '__main__':
    ts=TimeData()
    print(ts.time_frame())
