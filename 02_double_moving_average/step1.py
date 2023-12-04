import numpy as np
import pandas as pd
from time import time
from datetime import date


class DataNode:
    def __init__(self, path, code):
        df = pd.read_excel(path, sheet_name=code)
        self.name = code
        self.trade_date = pd.to_datetime(df['Date'].values).date
        self.open = df['Open'].values
        self.close = df['Close'].values
        self.high = df['High'].values
        self.low = df['Low'].values
        self.net_asset = np.zeros_like(self.open, dtype=np.float32)


def rolling_mean(arr, window):
    conv_kernel = np.ones(window)
    return np.convolve(arr, conv_kernel, 'valid')/window


def calculate_ma(data: DataNode, boundary, day_set):
    """
    计算必要的均线：
    由于numpy没有series的rolling方法，如果用for循环，效率不高
    这里用全1数组卷积，等价于求和，最后除以窗口大小，得到均值
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
    if buy_indices.size == 0 and sell_indices.size == 0:
        return (_capital, _shares)
    # 有买入信号，无卖出信号
    elif buy_indices.size != 0 and sell_indices.size == 0:
        if _shares == 0:
            var_price = data.open[buy_indices[0]]
            shares = _capital*(1-cr)//var_price
            capital = _capital - shares*var_price*(1+cr)
            return (capital, shares)
        else:
            return (_capital, _shares)
    # 无买入信号，有卖出信号
    elif buy_indices.size == 0 and sell_indices.size != 0:
        if _shares != 0:
            var_price = data.open[sell_indices[0]]
            capital = _capital + _shares*var_price*(1-cr)
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
            capital += shares*var_price*(1-cr)
            shares = 0
        sell_indices = sell_indices[1:]

        if sell_indices.size == 0:  # 如果卖出信号已经用完，由于存在买入信号，那么买入
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


def trade1(d: DataNode, boundary, day_set, _capital, _shares, cr):
    # 计算双均线
    ma = calculate_ma(d, boundary, day_set)
    # 计算交易信号
    buy_indices, sell_indices = calculate_trade_signals(boundary, ma, _shares)

    # 模拟交易
    r = []  # records
    bi, si = 0, 0
    ibegin, iend = boundary
    iend += 1  # 变为不可达

    # 既无买入信号，也无卖出信号
    if buy_indices.size == 0 and sell_indices.size == 0:
        d.net_asset[ibegin:iend] = _capital + _shares*d.close[ibegin:iend]
        return (_capital, _shares, r)
    # 有买入信号，无卖出信号
    elif buy_indices.size != 0 and sell_indices.size == 0:
        if _shares == 0:
            bi = buy_indices[0]
            # [ibegin,bi)之间，无持仓
            d.net_asset[ibegin:bi] = _capital
            # 买入
            var_price = d.open[bi]
            shares = _capital*(1-cr)//var_price
            capital = _capital - shares*var_price*(1+cr)
            r.append(
                (d.trade_date[bi].isoformat(), 1, var_price, shares, capital))
            # [bi,iend)之间，有持仓
            d.net_asset[bi:iend] = capital + shares*d.close[bi:iend]
            return (capital, shares, r)
        else:  # 有持仓，不买入，即 无操作
            d.net_asset[ibegin:iend+1] = _capital + \
                _shares*d.close[ibegin:iend+1]
            return (_capital, _shares, r)
    # 无买入信号，有卖出信号
    elif buy_indices.size == 0 and sell_indices.size != 0:
        if _shares != 0:
            si = sell_indices[0]
            # [ibegin,si)之间，有持仓
            d.net_asset[ibegin:si] = _capital + _shares*d.close[ibegin:si]
            # 卖出
            var_price = d.open[si]
            capital = _capital + _shares*var_price*(1-cr)
            r.append(
                (d.trade_date[si].isoformat(), -1, var_price, 0, capital))
            # [si,iend)之间，无持仓
            d.net_asset[si:iend] = capital
            return (capital, 0, r)
        else:  # 无持仓，不卖出，即 无操作
            d.net_asset[ibegin:iend] = _capital
            return (_capital, _shares, r)

    # 既有买入信号，也有卖出信号
    capital = _capital
    shares = _shares
    var_price = 0.0
    if sell_indices[0] < buy_indices[0]:  # 如果第一个是卖出信号
        if shares != 0:  # 如果有持仓，那么卖出
            si = sell_indices[0]
            # [ibegin,si)之间，有持仓
            d.net_asset[ibegin:si] = capital + shares*d.close[ibegin:si]
            # 卖出
            var_price = d.open[si]
            capital += shares*var_price*(1-cr)
            shares = 0
            r.append(
                (d.trade_date[si].isoformat(), -1, var_price, 0, capital))
        sell_indices = sell_indices[1:]  # 切掉第一个卖出信号

        if sell_indices.size == 0:  # 如果卖出信号已经用完，由于存在买入信号，那么买入
            bi = buy_indices[0]
            # 在[si,bi)之间，无持仓
            d.net_asset[si:bi] = capital
            # 买入
            var_price = d.open[bi]
            shares = capital*(1-cr)//var_price
            capital -= shares*var_price*(1+cr)
            r.append(
                (d.trade_date[bi].isoformat(), 1, var_price, shares, capital))
            # [bi,iend)之间，有持仓
            d.net_asset[bi:iend] = capital + shares*d.close[bi:iend]
            return (capital, shares, r)

    lai = max(si, ibegin)  # last action index
    for b, s in zip(buy_indices, sell_indices):
        # 在[lai,b)之间无持仓
        d.net_asset[lai:b] = capital
        # 买入
        var_price = d.open[b]
        shares = capital*(1-cr)//var_price
        capital -= shares*var_price*(1+cr)
        r.append(
            (d.trade_date[b].isoformat(), 1, var_price, shares, capital))

        # 在[b,s)之间有持仓
        d.net_asset[b:s] = capital+shares*d.close[b:s]
        # 卖出
        var_price = d.open[s]
        capital += shares*var_price*(1-cr)
        shares = 0
        r.append(
            (d.trade_date[s].isoformat(), -1, var_price, 0, capital))

        lai = s  # 记录卖出索引

    if buy_indices[-1] > sell_indices[-1]:  # 可能还存在一个买入信号未处理
        bi = buy_indices[-1]
        # 在[lai,bi)之间无持仓
        d.net_asset[lai:bi] = capital
        # 买入
        var_price = d.open[bi]
        shares = capital*(1-cr)//var_price
        capital -= shares*var_price*(1+cr)
        r.append(
            (d.trade_date[bi].isoformat(), 1, var_price, shares, capital))
        # 在[bi,iend)之间有持仓
        d.net_asset[bi:iend] = capital+shares*d.close[bi:iend]
    else:  # 买入/卖出信号 刚刚好成对用完
        # [lai:iend)之间无持仓
        d.net_asset[lai:iend] = capital

    return (capital, shares, r)


def calculate_max_drawdown(na_arr):
    net_value = na_arr/na_arr[0]
    max_net_value = np.maximum.accumulate(net_value)
    drawdown = (max_net_value - net_value)/max_net_value
    return np.max(drawdown)


def calculate_sharpe_ratio(year_yield, rf=3e-2):
    # 默认无风险利率rf为3%，即银行活期利率
    return (np.mean(year_yield)-rf)/np.std(year_yield)


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
