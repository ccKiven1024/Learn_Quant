import numpy as np
import pandas as pd


class Data:
    def __init__(self, path, code, cr, md_set: np.ndarray) -> None:
        df = pd.read_excel(path, sheet_name=code)

        # 读取数据并存入ndarray
        self.open = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        self.close = df['Close'].values
        volume = df['Vol'].values
        # 计算均线
        self.sma = np.zeros(shape=(df.shape[0], md_set.shape[0]))
        for i in range(md_set.shape[0]):
            self.sma[:, i] = df['Close'].rolling(md_set[i]).mean()
        # 将数据拼接成矩阵
        self.m = np.column_stack((self.open, high, low, self.close, volume))
        self.m = np.hstack((self.m, self.sma))

        df['Date'] = pd.to_datetime(df['Date'])  # 将日期转为datetime格式
        self.trade_date = df['Date'].dt.date.values  # 取出日期
        # 一次性将每月最后一天的索引计算出来
        self.ldi = df.groupby(pd.Grouper(
            key='Date', freq='M')).apply(lambda x: x.index[-1]).values
        # last day of each month
        self.ld = self.trade_date[self.ldi]

        self.stock_code = code
        self.cr = cr

    def get_input(self, fd, _scope):
        begin, end = _scope[0]-fd, _scope[1]+1-fd
        matrix = self.m[begin:end]
        m = end - begin  # 待划分的组数
        input_array = np.zeros(shape=(m, fd*matrix.shape[1]))

        indices = np.arange(fd)  # 初始化索引
        for i in range(m-fd):
            input_array[i] = np.concatenate(matrix[indices])
            indices = indices+1
        return input_array

    def scope2str(self, _range):
        return f"{self.trade_date[_range[0]].isoformat()} - {self.trade_date[_range[1]].isoformat()}"
