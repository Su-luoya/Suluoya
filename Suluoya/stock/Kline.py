import json
import os
import sys
import time
from functools import reduce
from operator import add
from typing import List, Sequence, Union


import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Kline, Line, Bar, Grid

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
try:
    from ..data.Stock import StockData
    from ..log.log import hide, show, slog, sprint, makedir
except:
    from data.Stock import StockData
    from log.log import hide, show, slog, sprint, makedir


class SyntheticIndex(object):

    def __init__(self, weights={'贵州茅台': 0.5,
                                '隆基股份': 0.2,
                                '五粮液': 0.3},
                 path='.\\Suluoya cache\\'
                 ):
        self.names = list(weights.keys())
        self.weights = list(weights.values())
        self.lens = len(self.names)
        self.path = path
        makedir(self.path, '')

    def stock_weights(self):
        return dict(zip(self.names, [i/sum(self.weights) for i in self.weights]))

    def stock_data(self):
        if not self.path:
            # global StockData
            sd = StockData(names=self.names, path=self.path)
            sd.start_date = sorted(
                [i['ipoDate'] for i in sd.stocks_info().values()])[-1]
            sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
            return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]
        else:
            try:
                stocks_info = pd.read_csv(
                    f'{self.path}\\stock data\\stocks_info.csv').set_index('name').T.to_dict()
                sd = StockData(names=self.names, path=self.path)
                sd.start_date = sorted([i['ipoDate']
                                       for i in stocks_info.values()])[-1]
                sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
                return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]
            except:
                sd = StockData(names=self.names, path=self.path)
                sd.start_date = sorted(
                    [i['ipoDate'] for i in sd.stocks_info().values()])[-1]
                sd.end_date = time.strftime("%Y-%m-%d", time.localtime())
                return sd.stocks_data()[['name', 'date', 'open', 'close', 'low', 'high', 'volume']]

    def get_data(self):
        data = self.stock_data()
        data['weight'] = data['name'].map(self.stock_weights())
        data['real_weight'] = data.weight/data.open
        date = data.date.unique()
        x = [list(data[data.date == i].real_weight /
                  data[data.date == i].real_weight.sum()) for i in date]
        data = data.sort_values(by=['date', 'name'])
        data['z_weight'] = reduce(add, tuple(x))
        data.date = data.date.astype(str)
        data.open = (data.open*data.z_weight).map(lambda x: round(x, 2))
        data.close = (data.close*data.z_weight).map(lambda x: round(x, 2))
        data.low = (data.low*data.z_weight).map(lambda x: round(x, 2))
        data.high = (data.high*data.z_weight).map(lambda x: round(x, 2))
        data.volume = (data.volume*data.z_weight).map(lambda x: round(x, 0))
        data = data.groupby('date').sum()
        data['date'] = data.index
        return data[['date', 'open', 'close', 'low', 'high', 'volume']]

    def calculate(self):
        data = self.get_data()
        # 计算EMA(12)和EMA(16)
        data['EMA12'] = data['close'].ewm(alpha=2 / 13, adjust=False).mean()
        data['EMA26'] = data['close'].ewm(alpha=2 / 27, adjust=False).mean()
        # 计算DIFF、DEA、MACD
        data['DIFF'] = data['EMA12'] - data['EMA26']
        data['DEA'] = data['DIFF'].ewm(alpha=2 / 10, adjust=False).mean()
        data['MACD'] = 2 * (data['DIFF'] - data['DEA'])
        data['DIFF'] = data['DIFF'].map(lambda x: round(x, 2))
        data['DEA'] = data['DEA'].map(lambda x: round(x, 2))
        data['MACD'] = data['MACD'].map(lambda x: round(x, 2))
        # 上市首日，DIFF、DEA、MACD均为0
        data['DIFF'].iloc[0] = 0
        data['DEA'].iloc[0] = 0
        data['MACD'].iloc[0] = 0
        data['datas'] = data[['open', 'close', 'low', 'high', 'volume',
                              'EMA12', 'EMA26', 'DIFF', 'DEA', 'MACD']].values.tolist()
        return data

    def calculate_ma(self, day_count, data: int):
        result: List[Union[float, str]] = []
        for i in range(len(data["date"])):
            if i < day_count:
                result.append("-")
                continue
            sum_total = 0.0
            for j in range(day_count):
                sum_total += float(data["datas"][i - j][1])
            result.append(abs(float("%.2f" % (sum_total / day_count))))
        return result

    def draw_chart(self):
        data = self.calculate().to_dict('list')
        kline = (
            Kline()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="portfolio index",
                y_axis=data["datas"],
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#ef232a",
                    color0="#14b143",
                    border_color="#ef232a",
                    border_color0="#14b143",
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False, type_="inside", xaxis_index=[0, 0], range_end=100
                    ),
                    opts.DataZoomOpts(
                        is_show=True, xaxis_index=[0, 1], pos_top="97%", range_end=100
                    ),
                    opts.DataZoomOpts(is_show=False, xaxis_index=[
                                      0, 2], range_end=100),
                ],
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross",
                    background_color="rgba(245, 245, 245, 0.8)",
                    border_width=1,
                    border_color="#ccc",
                    textstyle_opts=opts.TextStyleOpts(color="#000"),
                ),
                brush_opts=opts.BrushOpts(
                    x_axis_index="all",
                    brush_link="all",
                    out_of_brush={"colorAlpha": 0.1},
                    brush_type="lineX",
                ),
                # 三个图的 axis 连在一块
                axispointer_opts=opts.AxisPointerOpts(
                    is_show=True,
                    link=[{"xAxisIndex": "all"}],
                    label=opts.LabelOpts(background_color="#777"),
                ),
            )
        )

        kline_line = (
            Line()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="MA5",
                y_axis=self.calculate_ma(day_count=5, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA10",
                y_axis=self.calculate_ma(day_count=10, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA20",
                y_axis=self.calculate_ma(day_count=20, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA30",
                y_axis=self.calculate_ma(day_count=30, data=data),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=1,
                    split_number=3,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )
        # Overlap Kline + Line
        overlap_kline_line = kline.overlap(kline_line)

        # Bar-1
        bar_1 = (
            Bar()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="Volumn",
                y_axis=data["volume"],
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                    function(params) {
                        var colorList;
                        if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # Bar-2 (Overlap Bar + Line)
        bar_2 = (
            Bar()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="MACD",
                y_axis=data["MACD"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                            function(params) {
                                var colorList;
                                if (params.data >= 0) {
                                colorList = '#ef232a';
                                } else {
                                colorList = '#14b143';
                                }
                                return colorList;
                            }
                            """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=2,
                    split_number=4,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        line_2 = (
            Line()
            .add_xaxis(xaxis_data=data["date"])
            .add_yaxis(
                series_name="DIFF",
                y_axis=data["DIFF"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="DEA",
                y_axis=data["DEA"],
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )
        # 最下面的柱状图和折线图
        overlap_bar_line = bar_2.overlap(line_2)

        # 最后的 Grid
        grid_chart = Grid(init_opts=opts.InitOpts(
            width="1500px", height="750px"))

        # 这个是为了把 data.datas 这个数据写入到 html 中,还没想到怎么跨 series 传值
        # demo 中的代码也是用全局变量传的
        grid_chart.add_js_funcs("var barData = {}".format(data["datas"]))

        # K线图和 MA5 的折线图
        grid_chart.add(
            overlap_kline_line,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", height="60%"),
        )
        # Volumn 柱状图
        grid_chart.add(
            bar_1,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
            ),
        )
        # MACD DIFS DEAS
        grid_chart.add(
            overlap_bar_line,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="82%", height="14%"
            ),
        )
        makedir(self.path, 'kline')
        grid_chart.render(
            f"{self.path}\\kline\\{str(self.names)}_{time.strftime('%Y-%m-%d', time.localtime())}.html")


if __name__ == "__main__":
    si = SyntheticIndex()
    si.draw_chart()
