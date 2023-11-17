import numpy as np
import pandas as pd
from time import time
from datetime import date


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


def calculate_ma(data: DataNode, boundary, day_set):
    """
    计算必要的均线：
    由于numpy没有series的rolling方法，如果用for循环，效率不高
    这里用全1数组卷积，等价于求和，除以窗口大小，得到均值
    """
    ibegin, iend = boundary
    ma = np.zeros(shape=(data.open.shape[0], len(day_set)), dtype=np.float64)
    for i in range(len(day_set)):
        ma[ibegin-1:iend+1,  i] = rolling_mean(
            data.close[ibegin-day_set[i]:iend+1], day_set[i])
    return ma


def calculate_trade_signals(boundary, ma, _shares):
    dif = ma[:, 0] - ma[:, 1]
    buy_indices = np.where((dif[:-1] < 0) & (dif[1:] > 0))[0]+2
    sell_indices = np.where((dif[:-1] > 0) & (dif[1:] < 0))[0]+2

    # 剪枝
    ibegin, iend = boundary
    buy_indices = buy_indices[(buy_indices > ibegin) & (buy_indices <= iend)]
    sell_indices = sell_indices[(
        sell_indices > ibegin) & (sell_indices <= iend)]

    # 对首日作单独判断
    # 如果dif[ibegin-1] > 0，那么第一个信号一定为买入信号
    # 如果无持仓，则可以添加该信号；否则，舍弃
    if dif[ibegin-1] > 0 and _shares == 0:
        buy_indices = np.concatenate(([ibegin], buy_indices))
    # 如果dif[ibegin-1] < 0，那么第一个信号一定为卖出信号
    # 如果无持仓，则舍弃该信号；否则，可以添加
    elif dif[ibegin-1] < 0 and _shares != 0:
        sell_indices = np.concatenate(([ibegin], sell_indices))

    return buy_indices, sell_indices


def trade(data: DataNode, boundary, day_set, _capital, _shares, cr):
    # 计算双均线
    ma = calculate_ma(data, boundary, day_set)
    # 计算交易信号
    buy_indices, sell_indices = calculate_trade_signals(boundary, ma, _shares)

    # 模拟交易
    # 既无买入信号，也无卖出信号
    if buy_indices.shape[0] == 0 and sell_indices.shape[0] == 0:
        return (_capital, _shares)
    # 有买入信号，无卖出信号
    elif buy_indices.shape[0] != 0 and sell_indices.shape[0] == 0:
        if _shares == 0:
            var_price = data.open[buy_indices[0]]
            shares = _capital(1-cr)//var_price
            capital = _capital - shares*var_price(1+cr)
            return (capital, shares)
        else:
            return (_capital, _shares)
    # 无买入信号，有卖出信号
    elif buy_indices.shape[0] == 0 and sell_indices.shape[0] != 0:
        if _shares != 0:
            var_price = data.open[sell_indices[0]]
            capital = _capital + _shares*var_price(1-cr)
            return (capital, 0)
        else:
            return (_capital, _shares)

    # 既有买入信号，也有卖出信号
    capital = _capital
    shares = _shares
    var_price = 0.0

    if sell_indices[0] < buy_indices[0]:  # 如果第一个是卖出信号
        if shares != 0:  # 如果有持仓，那么卖出
            var_price = data.open[sell_indices[0]]
            capital = capital + shares*var_price(1-cr)
            shares = 0
        sell_indices = sell_indices[1:]

        if not sell_indices:  # 如果卖出信号已经用完，由于存在买入信号，那么买入
            var_price = data.open[buy_indices[0]]
            shares = capital*(1-cr)//var_price
            capital -= shares*var_price*(1+cr)
            return (capital, shares)

    for bi, si in zip(buy_indices, sell_indices):
        var_price = data.open[bi]
        shares = capital*(1-cr)//var_price
        capital -= shares*var_price*(1+cr)

        var_price = data.open[si]
        capital += shares*var_price*(1-cr)
        shares = 0

    # 可能还存在一个买入信号未处理
    if buy_indices[-1] > sell_indices[-1]:
        var_price = data.open[buy_indices[-1]]
        shares = capital*(1-cr)//var_price
        capital -= shares*var_price*(1+cr)

    return (capital, shares)


def calculate_max_drawdown(data: DataNode, boundary):
    ibegin, iend = boundary
    peek_index = ibegin
    md = (data.high[ibegin]-data.low[ibegin])/data.low[ibegin]
    for i in range(ibegin, iend+1):
        if data.high[i] > data.high[peek_index]:
            peek_index = i
        md = max(md, (data.high[peek_index]-data.low[i])/data.low[i])
    return md


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./data/StockData.xlsx"
    stock_name = r'399300'
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    date_interval = (date(2006, 1, 4), date(2023, 8, 31))
    day_set = (5, 20)

    # 1 - 处理数据
    data = DataNode(file_path, stock_name)
    boundary = list(map(lambda date: np.where(
        data.trade_date == date)[0][0], date_interval))

    # 2 - 模拟交易
    capital, shares = trade(data, boundary, day_set,
                            init_capital, init_shares, cr)
    final_net_asset = capital+shares*data.close[-1]
    print(
        f"net asset = {final_net_asset:.3f}, cost time = {time()-s_clk:.3f} s")
    # net asset = 4946561.254, cost time = 1.247 s


if __name__ == '__main__':
    main()
