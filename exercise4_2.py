import numpy as np
import pandas as pd
import talib
import multiprocessing as mp
from datetime import date
from time import time


class DataNode:
    def __init__(self, file_path):
        df = pd.read_excel(file_path, sheet_name="399300")
        self.trade_date = pd.to_datetime(df['Date'].values).date
        self.open_price = df['Open'].values
        self.close_price = df['Close'].values
        self.dif, self.dea, self.hist = talib.MACD(
            self.close_price, fastperiod=12, slowperiod=26, signalperiod=9)
        self.gold_cross = np.where(
            (self.hist[:-1] < 0) & (self.hist[1:] > 0))[0] + 1
        self.dead_cross = np.where(
            (self.hist[:-1] > 0) & (self.hist[1:] < 0))[0] + 1


def calculate_trade_signals(data: DataNode, index_range, dea1, dea2, _shares):
    ibegin, iend = index_range
    # 剪枝金死叉
    gc = data.gold_cross[(data.gold_cross > ibegin-1)
                         & (data.gold_cross < iend)]
    dc = data.dead_cross[(data.dead_cross > ibegin-1)
                         & (data.dead_cross < iend)]

    buy_set = gc[data.dea[gc] < dea1]+1
    sell_set = dc[data.dea[dc] > dea2]+1

    # 对首日作单独判断
    if data.dea[ibegin-1] < dea1 and data.hist[ibegin-1] > 0:
        buy_set = np.insert(buy_set, 0, ibegin)
    elif data.dea[ibegin-1] > dea2 and data.hist[ibegin-1] < 0:
        sell_set = np.insert(sell_set, 0, ibegin)

    # 检查买入/卖出集合
    if buy_set.size == 0 and sell_set.size == 0:
        return ([], [])
    if buy_set.size != 0 and sell_set.size == 0:
        return ([buy_set[0]], [])
    if buy_set.size == 0 and sell_set.size != 0:
        return ([], [sell_set[0]])

    # 买入/卖出集合都不为空
    # 由于两集合元素是混乱的，现在需要挑选使其交替出现
    buy_indices = []
    sell_indices = []
    m, n = buy_set.shape[0], sell_set.shape[0]
    i, j = 0, 0
    flag_bs = None

    # 根据_shares的值确定首个买入/卖出信号
    if _shares == 0:
        buy_indices.append(buy_set[0])
        flag_bs = True  # 找到买入信号
        i = 1
    else:
        sell_indices.append(sell_set[0])
        flag_bs = False  # 找到卖出信号
        j = 1

    while i < m and j < n:
        if flag_bs:
            while j < n and sell_set[j] <= buy_indices[-1]:
                j += 1
            if j != n:
                sell_indices.append(sell_set[j])
                flag_bs = False
                j += 1
        else:
            while i < m and buy_set[i] <= sell_indices[-1]:
                i += 1
            if i != m:
                buy_indices.append(buy_set[i])
                flag_bs = True
                i += 1

    if i == m and flag_bs:  # buy_set用完，试图从sell_set中添加一个元素
        while j < n and sell_set[j] <= buy_indices[-1]:
            j += 1
        if j != n:
            sell_indices.append(sell_set[j])
    if j == n and (not flag_bs):  # sell_set用完，试图从buy_set中添加一个元素
        while i < m and buy_set[i] <= sell_indices[-1]:
            i += 1
        if i != m:
            buy_indices.append(buy_set[i])

    return (buy_indices, sell_indices)


def trade(data: DataNode, index_range, dea1, dea2, _shares, _capital, cr):
    # 计算交易信号
    buy_indices, sell_indices = calculate_trade_signals(
        data, index_range, dea1, dea2, _shares)

    # 模拟交易

    # 存在交易信号为空时
    # 既没有买入信号，也没有卖出信号
    if (not buy_indices) and (not sell_indices):
        return (_capital, _shares)  # 不交易，直接返回
    # 只有买入信号
    if buy_indices and (not sell_indices):
        if _shares == 0:  # 如果没有持股，直接买入
            var_price = data.open_price[buy_indices[0]]
            shares = int(_capital * (1 - cr) / var_price)
            capital = _capital - shares * var_price * (1 + cr)
            return (capital, shares)
        else:  # 如果有持股
            return (_capital, _shares)  # 不交易，直接返回
    # 只有卖出信号
    if (not buy_indices) and sell_indices:
        if _shares == 0:  # 无持股，直接返回
            return (_capital, 0)
        else:  # 有持股，卖出一次直接返回
            var_price = data.open_price[sell_indices[0]]
            capital = _capital + _shares * var_price * (1 - cr)
            return (capital, 0)

    # 既有买入信号，也有卖出信号
    capital = _capital
    shares = _shares
    var_price = 0.0

    if sell_indices[0] < buy_indices[0]:  # 如果第一个是卖出信号
        # 若持仓，执行一次卖出
        if shares > 0:
            var_price = data.open_price[sell_indices[0]]
            capital += shares * var_price * (1 - cr)
            shares = 0
        # 若不持仓，可以跳过无意义的0持仓卖出
        sell_indices = sell_indices[1:]  # 去掉第一个卖出信号

        if not sell_indices:  # 如果没有卖出信号了，由于存在买入信号，那么买入
            var_price = data.open_price[buy_indices[0]]
            shares = int(capital * (1 - cr) / var_price)
            capital -= shares * var_price * (1 + cr)
            return (capital, shares)

    for bi, si in zip(buy_indices, sell_indices):
        # 买入
        var_price = data.open_price[bi]
        shares = int(capital * (1 - cr) / var_price)
        capital -= shares * var_price * (1 + cr)
        # 卖出
        var_price = data.open_price[si]
        capital += shares * var_price * (1 - cr)
        shares = 0

    # 可能还有一个买入信号未处理
    if buy_indices[-1] > sell_indices[-1]:
        var_price = data.open_price[buy_indices[-1]]
        shares = int(capital * (1 - cr) / var_price)
        capital -= shares * var_price * (1 + cr)

    return (capital, shares)


def func(data: DataNode, index_range, dea1, dea2, _shares, _capital, cr):
    c, s = trade(data, index_range, dea1, dea2, _shares, _capital, cr)
    na = c + s * data.close_price[index_range[1]]
    return (na, dea1, dea2)


def get_optimal_dea(data: DataNode, index_range, dea1_range, dea2_range, _capital, cr):
    with mp.Pool() as pool:
        res = pool.starmap(func, [(data, index_range, dea1, dea2, 0, _capital, cr)
                           for dea1 in dea1_range for dea2 in dea2_range])
    return max(res, key=lambda x: x[0])


def main():
    s_clk = time()
    # 0 - 题目数据
    file_path = './data/StockData.xlsx'
    init_capital = 1e6
    cr = 5e-4
    date_interval = (date(2006, 1, 4), date(2023, 8, 31))
    dea1_range = dea2_range = range(-100, 101)

    # 1 - 处理数据
    data = DataNode(file_path)
    irange = list(map(lambda date: np.where(
        data.trade_date == date)[0][0], date_interval))

    # 2 - 计算最优dea1,dea2
    na, dea1, dea2 = get_optimal_dea(
        data, irange, dea1_range, dea2_range, init_capital, cr)
    print(
        f"na = {na:.3f}, dea1 = {dea1}, dea2 = {dea2}, time cost = {time()-s_clk:.3f} s")

    # na = 7123024.124, dea1 = 71, dea2 = 0, time cost = 5.090 s


if __name__ == '__main__':
    main()


"""
# 版本1： 采用循环日期的方式，速度慢

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from datetime import date
from time import time
import multiprocessing as mp


def trade(_open, _dea, _hist, dea1, dea2, _capital, _shares, cr, irange):
    flag_bs = 0
    position = 0  # 0 - 空仓，1 - 多仓，-1 - 空仓
    var_price = 0.0
    capital = _capital
    shares = _shares
    ibegin, iend = irange
    # 处理首日
    if _dea[ibegin-1] < dea1 and _hist[ibegin-1] > 0:
        flag_bs = 1
    elif _dea[ibegin-1] > dea2 and _hist[ibegin-1] < 0:
        flag_bs = -1

    if flag_bs == 1 and position == 0:  # 空仓时买入
        var_price = _open[ibegin]
        shares = int(capital * (1 - cr) / var_price)  # 买入份额
        if shares > 0:
            position = 1
            capital -= shares * var_price * (1 + cr)  # 更新资金
        flag_bs = 0
    elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
        var_price = _open[ibegin]
        capital += shares * var_price * (1 - cr)  # 更新资金
        shares = 0
        position = 0
        flag_bs = 0

    # 处理(ibegin,iend]之间的日期
    for i in range(ibegin+1, iend+1):
        if _dea[i-1] < dea1 and (_hist[i-2] < 0 and _hist[i-1] > 0):
            flag_bs = 1
        elif _dea[i-1] > dea2 and (_hist[i-2] > 0 and _hist[i-1] < 0):
            flag_bs = -1

        if flag_bs == 1 and position == 0:
            var_price = _open[i]
            shares = int(capital * (1 - cr) / var_price)
            if shares > 0:
                position = 1
                capital -= shares * var_price * (1 + cr)
            flag_bs = 0
        elif flag_bs == -1 and position == 1:
            var_price = _open[i]
            capital += shares * var_price * (1 - cr)
            shares = 0
            position = 0
            flag_bs = 0

    return (capital, shares)


def trade1(_trade_date, _open, _close, _dea, _hist, dea1, dea2, _capital, _shares, cr, irange, net_asset, records):
    flag_bs = 0
    position = 0  # 0 - 空仓，1 - 多仓，-1 - 空仓
    var_price = 0.0
    capital = _capital
    shares = _shares
    ibegin, iend = irange
    # 处理首日
    if _dea[ibegin-1] < dea1 and _hist[ibegin-1] > 0:
        flag_bs = 1
    elif _dea[ibegin-1] > dea2 and _hist[ibegin-1] < 0:
        flag_bs = -1

    if flag_bs == 1 and position == 0:  # 空仓时买入
        var_price = _open[ibegin]
        shares = int(capital * (1 - cr) / var_price)  # 买入份额
        if shares > 0:
            position = 1
            capital -= shares * var_price * (1 + cr)  # 更新资金
            records.append(
                (_trade_date[ibegin], flag_bs, capital, shares, var_price))
        flag_bs = 0
    elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
        var_price = _open[ibegin]
        capital += shares * var_price * (1 - cr)  # 更新资金
        shares = 0
        position = 0
        records.append(
            (_trade_date[ibegin], flag_bs, capital, shares, var_price))
        flag_bs = 0
    net_asset[ibegin] = capital + shares * _close[ibegin]

    # 处理(ibegin,iend]之间的日期
    for i in range(ibegin+1, iend+1):
        if _dea[i-1] < dea1 and (_hist[i-2] < 0 and _hist[i-1] > 0):
            flag_bs = 1
        elif _dea[i-1] > dea2 and (_hist[i-2] > 0 and _hist[i-1] < 0):
            flag_bs = -1

        if flag_bs == 1 and position == 0:
            var_price = _open[i]
            shares = int(capital * (1 - cr) / var_price)
            if shares > 0:
                position = 1
                capital -= shares * var_price * (1 + cr)
                records.append(
                    (_trade_date[i], flag_bs, capital, shares, var_price))
            flag_bs = 0
        elif flag_bs == -1 and position == 1:
            var_price = _open[i]
            capital += shares * var_price * (1 - cr)
            shares = 0
            position = 0
            records.append(
                (_trade_date[i], flag_bs, capital, shares, var_price))
            flag_bs = 0
        net_asset[i] = capital + shares * _close[i]

    return (capital, shares)


def func1(_open, _close, _dea, _hist, dea1, dea2, init_capital, shares, cr, irange):
    _capital, _shares = trade(_open, _dea, _hist, dea1,
                              dea2, init_capital, shares, cr, irange)
    _net_asset = _capital + _shares * _close[irange[1]]
    return (_net_asset, dea1, dea2)


def get_optimal_dea(_open, _close, _dea, _hist, dea1_range, dea2_range, init_capital,  cr, irange):
    with mp.Pool() as pool:
        res = pool.starmap(func1, [
            (_open, _close, _dea, _hist, dea1,
             dea2, init_capital, 0, cr, irange)
            for dea1 in dea1_range
            for dea2 in dea2_range
        ])
    return max(res, key=lambda x: x[0])

# 单线程
# def get_optimal_dea(_open, _close, _dea, _hist, dea1_range, dea2_range, init_capital,  cr, irange):
#     max_ns = 0.0
#     optimal_dea1 = optimal_dea2 = 0
#     for dea1 in dea1_range:
#         for dea2 in dea2_range:
#             _capital, _shares = trade(_open, _dea, _hist, dea1,
#                                       dea2, init_capital, 0, cr, irange)
#             _net_asset = _capital + _shares * _close[irange[1]]
#             if _net_asset > max_ns:
#                 max_ns = _net_asset
#                 optimal_dea1, optimal_dea2 = dea1, dea2
#     return (max_ns, optimal_dea1, optimal_dea2)


def main():
    s_clk = time()

    # 0 - 题目数据
    init_capital = 1e6
    cr = 5e-4
    date_interval = (date(2006, 1, 4), date(2023, 8, 31))
    dea1_range = dea2_range = range(-100, 101)

    # 1 - 读取数据
    excel_path = r"./StockData.xlsx"
    df = pd.read_excel(excel_path)
    trade_date = pd.to_datetime(df['Date'].values).date
    _open = df['Open'].values
    _close = df['Close'].values
    _dif, _dea, _hist = talib.MACD(
        _close, fastperiod=12, slowperiod=26, signalperiod=9)

    irange = (np.where(trade_date == date_interval[0])[0][0],
              np.where(trade_date == date_interval[1])[0][0])

    # 2 - 计算最优dea1, dea2
    c, dea1, dea2 = get_optimal_dea(
        _open, _close, _dea, _hist, dea1_range, dea2_range, init_capital, cr, irange)

    print(f"dea1={dea1}, dea2={dea2}, c={c}\nTime Cost = {time()-s_clk} s")
    # dea1=71, dea2=0, c=7123024.123745045

if __name__ == "__main__":
    main()
"""
