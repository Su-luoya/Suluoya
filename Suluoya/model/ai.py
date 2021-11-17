import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def exploratory_experiment(df, target, target_type='R'):
    '''
    Func:机器学习探索性实验\n
    In:dataframe各指标数据\n
    target --> 因变量\n
    target_type --> R连续变量；C分类变量
    '''
    if target_type == 'R' or target_type == 'r':
        from pycaret.regression import compare_models, setup
    elif target_type == 'C' or target_type == 'c':
        from pycaret.classification import compare_models, setup
    setup(df, target)
    compare_models()


def formula(df, y='y'):
    x_list = list(df.columns)
    x_list.remove(y)
    fm = f'y ~ '
    for x in x_list:
        fm += f'{x}+'
    return fm[:-1]


def linear_regression(df, y='y'):
    '''
    Func:最小二乘法\n
    df --> 各指标数据\n
    y --> 因变量（连续变量）
    '''
    fm = formula(df, y='y')
    mod = smf.ols(formula=fm, data=df)
    res = mod.fit()
    return res.summary()


def logistic_regression(df, y='y'):
    '''
    Func:二分类logistic回归\n
    df --> 各指标数据\n
    y --> 因变量（二分类变量）
    '''
    df_y = df[y]
    df_x = df.drop(y, axis=1)
    mod = sm.Logit(df_y, df_x)
    res = mod.fit()
    return res.summary()

#import statsmodels.api as sm
# data = sm.datasets.engel.load_pandas().data


class quantile_regression(object):
    '''
    Func:分位数回归\n
    In:dataframe各指标数据\n
    df --> 各指标数据\n
    y --> 因变量\n
    step --> 分位数步长
    '''

    def __init__(self, df, y='y', step=0.05):
        self.df = df
        self.y = y
        self.step = step
        x_list = list(df.columns)
        x_list.remove(y)
        self.x = x_list
        self.formula = formula(df, y=y)

    def quant_summary(self, q=0.5):
        '''
        Func:分位数回归概要\n
        q --> 分位数
        '''
        mod = smf.quantreg(formula=self.formula, data=self.df)
        res = mod.fit(q=q)
        return res.summary()

    def ols_summary(self):
        '''
        Func:最小二乘法概要\n
        '''
        mod = smf.ols(formula=self.formula, data=self.df)
        res = mod.fit()
        return res.summary()

    def quantile(self, q=0.5):
        '''
        Func:不准用！！!
        '''
        mod = smf.quantreg(formula=self.formula, data=self.df)
        res = mod.fit(q=q)
        df_lu = res.conf_int()
        df_lu.columns = ['lb', 'ub']
        result_dict = df_lu.to_dict()
        result_dict['params'] = dict(res.params)
        result_dict['pvalue'] = dict(res.pvalues)
        result_dict['q'] = str(res.q)[:4]
        result_dict['Pseudo R-squared'] = res.prsquared
        return pd.DataFrame(result_dict)

    def quantiles(self):
        '''
        Func:分位数回归结果\n
        OUT:包含从0.05到0.95的分位数回归结果概要 --> dataframe 
        '''
        qs = np.arange(0.05, 0.96, self.step)
        q_results = [self.quantile(q) for q in qs]
        for i in range(int(1/self.step)-1):
            q_results[i]['q'] = str(qs[i])[:4]
        return pd.concat(q_results)

    def quantiles_dict(self):
        '''
        Func:分位数回归结果\n
        OUT:包含从0.05到0.95的分位数回归结果概要 --> dict，键为自变量 
        '''
        qs = np.arange(0.05, 0.96, self.step)
        q_results = [self.quantile(q) for q in qs]
        q_dict = {}
        for x in self.x:
            lists = []
            for q_result in q_results:
                lists.append(q_result.T[[x]].T)
            q_dict[x] = pd.concat(lists)
        return q_dict

    def plot(self):
        '''
        Func:作图
        '''
        fig, axs = plt.subplots(len(self.x), 1, sharex=True, figsize=(6, 6))
        i = 0
        for x, data in self.quantiles_dict().items():
            q = list(data.q)
            lb = list(data.lb)
            ub = list(data.ub)
            params = list(data.params)
            axs[i].fill_between(q, lb, ub, alpha=0.3)
            axs[i].plot(q, params, marker=".", markersize=8)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            for m, n in zip(q, params):
                axs[i].text(m, n, '%.2f' % n)
            for m, n in zip(q, lb):
                axs[i].text(m, n, '%.2f' % n)
            for m, n in zip(q, ub):
                axs[i].text(m, n, '%.2f' % n)
            axs[i].set_ylabel(x)
            i += 1
        axs[i-1]
        plt.show()


if __name__ == '__main__':
    df = pd.read_excel('./data.xlsx')
    lr = logistic_regression(df, y='yl')
    print(lr)
