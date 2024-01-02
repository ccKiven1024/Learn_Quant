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
        self.last_day_indices = df.groupby(pd.Grouper(
            key='Date', freq='M')).apply(lambda x: x.index[-1]).values
        # last day of each month
        self.ld = self.trade_date[self.last_day_indices]

        self.stock_code = code
        self.cr = cr

    def get_last_index(self, date):
        """
        返回date所在月份的最后一天的索引
        """
        return self.last_day_indices[np.searchsorted(self.ld, date)]
