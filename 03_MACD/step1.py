# 错误的结果的提示：如果能预测MACD的指标，会获得超高收益率
# 版本2：只遍历每个交易信号

import numpy as np
import pandas as pd
import talib as ta
from datetime import date

# 0 - 题目数据
init_capital = 1e6
cr = 5e-4
excel_path = r"./../data/StockData.xlsx"
stock_code = r"399300"
date_interval = (date(2006, 1, 4), date(2023, 8, 31))

# 1 - 读取数据
df = pd.read_excel(excel_path, sheet_name=stock_code)
trade_date = pd.to_datetime(df["Date"].values).date
open_price = df["Open"].values
close_price = df["Close"].values

# 计算必要的指标
dif, dea, hist = ta.MACD(close_price, fastperiod=12,
                         slowperiod=26, signalperiod=9)
ibegin, iend = irange = tuple(map(lambda date: np.where(
    trade_date == date)[0][0], date_interval))
iend += 1  # 左闭右开

# 计算交易信号
buy_indices = np.where((hist[:-1] < 0) & (hist[1:] > 0))[0]+2
sell_indices = np.where((hist[:-1] > 0) & (hist[1:] < 0))[0]+2

# 2 - 模拟交易
capital = init_capital
shares = 0
var_price = 0.0
records = []

# 只保留(ibegin,iend)之间的交易信号
buy_indices = buy_indices[np.where(
    (buy_indices > ibegin) & (buy_indices < iend))].tolist()
sell_indices = sell_indices[np.where(
    (sell_indices > ibegin) & (sell_indices < iend))].tolist()
# 对于首日作特殊处理
if hist[ibegin-1] > 0:
    buy_indices.insert(0, ibegin)
elif hist[ibegin-1] < 0:
    sell_indices.insert(0, ibegin)

# # 若持仓，首先平仓
# if shares>0:
#     k = sell_indices[0]
#     var_price = open_price[k]
#     capital += shares * var_price * (1 - cr)  # 更新资金
#     shares = 0
#     records.append(
#         (trade_date[k].isoformat(), -1, capital, shares, var_price))

i, j = 0, 0
while sell_indices[j] < buy_indices[i]:
    j += 1  # 跳过无意义的0持仓卖出
# 此时buy_indices[i] <= sell_indices[j]，找到第一个买入信号
sell_indices = sell_indices[j:]
for bi, si in zip(buy_indices, sell_indices):
    """
    根据hist的值，它在0轴上下波动，那么上穿/下穿一定是交替出现的，所以无需检查持仓
    """
    # 买入
    var_price = open_price[bi]
    shares = int(capital * (1 - cr) / var_price)  # 买入份额
    capital -= shares * var_price * (1 + cr)  # 更新资金
    records.append(
        (trade_date[bi].isoformat(), 1, capital, shares, var_price))
    # 卖出
    var_price = open_price[si]
    capital += shares * var_price * (1 - cr)  # 更新资金
    shares = 0
    records.append(
        (trade_date[si].isoformat(), -1, capital, shares, var_price))

# 可能还有一个买入信号未处理
if buy_indices[-1] > sell_indices[-1]:
    bi = buy_indices[-1]
    var_price = open_price[bi]
    shares = int(capital * (1 - cr) / var_price)  # 买入份额
    capital -= shares * var_price * (1 + cr)  # 更新资金
    records.append(
        (trade_date[bi].isoformat(), 1, capital, shares, var_price))

net_asset = capital + shares * close_price[iend-1]
print(f"Net Asset: {net_asset:.3f}")

# 3 - 保存结果
result_csv = r"./result4_1.csv"
df1 = pd.DataFrame(records, columns=[
                   "Date", "B/S", "Capital", "Shares", "Price"])
df1.to_csv(result_csv, index=False)

"""
# 版本1：循环每个交易日


import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from datetime import date

# 0 - 题目数据
init_capital = 1e6
cr = 5e-4
date_interval = (date(2006, 1, 4), date(2023, 8, 31))

# 1 - 读取数据
excel_path = r"./StockData.xlsx"
df = pd.read_excel(excel_path)
trade_date = pd.to_datetime(df['Date'].values).date
_open = df['Open'].values
_close = df['Close'].values
del df
_dif, _dea, _hist = talib.MACD(
    _close, fastperiod=12, slowperiod=26, signalperiod=9)
# EMA means Exponential Moving Average
# MACD means Moving Average Convergence Divergence,
# a indicator of technical analysis that consist of three parts:
# DIF means the difference between EMA12 and EMA26, i.e. DIF = EMA12 - EMA26
# DEA means the EMA9 of DIF, i.e. DEA = EMA(DIF, 9)
# Hist, histogram, means the difference between DIF and DEA, i.e. Hist = DIF - DEA

# 2 - 模拟交易
ibegin, iend = (np.where(trade_date == date_interval[0])[0][0],
                np.where(trade_date == date_interval[1])[0][0])
net_asset = np.zeros_like(_close)
flag_bs = 0
position = 0  # 0 - 空仓，1 - 多仓，-1 - 空仓
var_price = 0.0
shares = 0
net_asset[ibegin-1] = capital = init_capital
record = []


# 处理首日
if _hist[ibegin-1] > 0:
    flag_bs = 1
elif _hist[ibegin-1] < 0:
    flag_bs = -1

if flag_bs == 1 and position == 0:  # 空仓时买入
    var_price = _open[ibegin]
    shares = int(capital * (1 - cr) / var_price)  # 买入份额
    if shares > 0:
        position = 1
        capital -= shares * var_price * (1 + cr)  # 更新资金
    record.append((trade_date[ibegin], flag_bs, capital, shares, var_price))
    flag_bs = 0
elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
    var_price = _open[ibegin]
    capital += shares * var_price * (1 - cr)  # 更新资金
    shares = 0
    position = 0
    record.append((trade_date[ibegin], flag_bs, capital, shares, var_price))
    flag_bs = 0
net_asset[ibegin] = capital + shares * _close[ibegin]

# 处理(ibegin,iend)之间的日期
for i in range(ibegin+1, iend):
    if _hist[i-2] < 0 and _hist[i-1] > 0:
        flag_bs = 1
    elif _hist[i-2] > 0 and _hist[i-1] < 0:
        flag_bs = -1

    if flag_bs == 1 and position == 0:
        var_price = _open[i]
        shares = int(capital * (1 - cr) / var_price)
        if shares > 0:
            position = 1
            capital -= shares * var_price * (1 + cr)
        record.append((trade_date[i], flag_bs, capital, shares, var_price))
        flag_bs = 0
    elif flag_bs == -1 and position == 1:
        var_price = _open[i]
        capital += shares * var_price * (1 - cr)
        shares = 0
        position = 0
        record.append((trade_date[i], flag_bs, capital, shares, var_price))
        flag_bs = 0
    net_asset[i] = capital + shares * _close[i]

# 处理最后一日
if _hist[iend-2] < 0 and _hist[iend-1] > 0:
    flag_bs = 1
elif _hist[iend-2] > 0 and _hist[iend-1] < 0:
    flag_bs = -1
if flag_bs == 1 and position == 0:
    pass
elif position == 1:
    if flag_bs == -1:
        var_price = _open[iend]
    else:
        var_price = _close[iend]
    capital += shares * var_price * (1 - cr)
    shares = 0
    position = 0
    record.append((trade_date[iend], flag_bs, capital, shares, var_price))
    flag_bs = 0
net_asset[iend] = capital
print(f"final capital: {capital:.3f}")
# final capital: 3988710.084

# 3 - 绘制资金曲线
# plt.plot(range(ibegin, iend+1), net_asset[ibegin:iend+1])
# plt.show()
# 4 - 保存交易记录
result_file_path = r"./result_4_1.csv"
df = pd.DataFrame(record, columns=[
                  'Date', 'B/S', 'Capital', 'Shares', 'Price'])
df.to_csv(result_file_path, index=False)

"""
