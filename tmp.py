import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from time import time
from datetime import date
from matplotlib.dates import YearLocator, DateFormatter


class DataNode:
    def __init__(self, path, name):
        df = pd.read_excel(path, sheet_name=name)
        self.name = name
        self.trade_date = pd.to_datetime(df['Date'].values).date
        self.open = df['Open'].values
        self.close = df['Close'].values
        self.high = df['High'].values
        self.low = df['Low'].values


def rolling_mean(arr, window):
    conv_kernel = np.ones(window)
    return np.convolve(arr, conv_kernel, 'valid')/window


def calculate_ma(data: DataNode, boundary, ma_set):
    """
    计算必要的均线：
    由于numpy没有series的rolling方法，如果用for循环，效率不高
    这里用全1数组卷积，等价于求和，除以窗口大小，得到均值
    """
    ibegin, iend = boundary
    ma = np.zeros(shape=(data.open.shape[0], len(ma_set)), dtype=np.float64)
    for i in range(len(ma_set)):
        ma[ibegin-1:iend+1,  i] = rolling_mean(
            data.close[ibegin-ma_set[i]:iend+1], ma_set[i])
    return ma


def calculate_trade_signals(data: DataNode, boundary, ma, _shares):
    dif = ma[:, 0] - ma[:, 1]
    gold_cross = np.where((dif[:-1] < 0) & (dif[1:] > 0))[0]+1
    dead_cross = np.where((dif[:-1] > 0) & (dif[1:] < 0))[0]+1

    # 剪枝
    ibegin, iend = boundary
    buy_indices = gold_cross[(gold_cross > ibegin) & (gold_cross <= iend)]
    sold_indices = dead_cross[(dead_cross > ibegin) & (dead_cross <= iend)]

    # 对首日作单独判断
    if dif[ibegin-1] > 0:
        gold_cross = np.insert(gold_cross, 0, ibegin-1)


def trade(data: DataNode, boundary, ma, _capital, _shares, cr):
    ibegin, iend = boundary
    capital = _capital
    shares = _shares


def main():
    file_path = r"./data/StockData.xlsx"
    date_interval = (date(2006, 1, 4), date(2023, 8, 31))
    ma_set = (5, 20)
    stock_name = '399300'

    data = DataNode(file_path, stock_name)
    irange = list(map(lambda date: np.where(
        data.trade_date == date)[0][0], date_interval))


if __name__ == '__main__':
    main()
