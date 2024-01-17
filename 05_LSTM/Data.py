import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self, path, code, cr, md_set: np.ndarray) -> None:
        df = pd.read_excel(path, sheet_name=code)

        self.close = df['Close'].values  # 取出收盘价

        # 计算均线
        self.sma = np.zeros(shape=(df.shape[0], md_set.shape[0]))
        for i in range(md_set.shape[0]):
            self.sma[:, i] = df['Close'].rolling(md_set[i]).mean()

        # 将数据拼接成矩阵，其中去除了第一列的日期
        self.m = np.column_stack((df.iloc[:, 1:].values, self.sma))
        # 标准化
        self.m = StandardScaler().fit_transform(self.m)

        # 将日期转为datetime格式
        df['Date'] = pd.to_datetime(df['Date'])
        self.trade_date = df['Date'].dt.date.values  # 取出日期
        # 一次性将每月最后一天的索引计算出来
        self.ldi = df.groupby(pd.Grouper(
            key='Date', freq='M')).apply(lambda x: x.index[-1]).values
        # last day of each month
        self.ld = self.trade_date[self.ldi]

        self.stock_code = code
        self.cr = cr

    def get_input(self,  _scope, fd, ud):
        begin = _scope[0]-(fd+ud-1)
        end = _scope[1]-fd+1  # 使其取不到
        matrix = self.m[begin:end]
        m = _scope[1]-_scope[0]+1
        input_array = np.zeros(shape=(m, matrix.shape[1], ud))

        for i in range(m):
            input_array[i] = matrix[i:i + ud].T
        return input_array

    def scope2str(self, _scope):
        return f"{self.trade_date[_scope[0]]} - {self.trade_date[_scope[1]]}"
