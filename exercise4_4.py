import numpy as np
import pandas as pd
import talib
import multiprocessing as mp
from time import time
from datetime import date
from exercise4_2 import trade,  get_optimal_dea
from exercise4_3 import get_first_date_index



def main():
    s_clk = time()

    # 0 - 题目数据
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    test_interval = (date(2014, 1, 2), date(2023, 8, 31))
    dea1_range = dea2_range = range(-100, 101)
    window_years_range = range(1, 8)

    # 1 - 读取数据
    excel_path = r"./StockData.xlsx"
    df = pd.read_excel(excel_path)
    trade_date = pd.to_datetime(df['Date'].values).date
    _open = df['Open'].values
    _close = df['Close'].values

    # 计算MACD指标
    _dif, _dea, _hist = talib.MACD(
        _close, fastperiod=12, slowperiod=26, signalperiod=9)

    # 判断金/死叉
    gold_cross = np.where((_hist[:-1] < 0) & (_hist[1:] > 0))[0] + 1
    dead_cross = np.where((_hist[:-1] > 0) & (_hist[1:] < 0))[0] + 1

    # 计算杂项
    start_year, end_year = [date.year for date in test_interval]


    # 2 - 模拟交易
    for window_size in window_years_range:
        capital = init_capital
        shares = init_shares
        for y in range(start_year,end_year):
            train_range = [get_first_date_index(trade_date,y-window_size),get_first_date_index(trade_date,y)-1]
            _,dea1,dea2 = get_optimal_dea(_open,_close,gold_cross,dead_cross,_hist,_dea,dea1_range,dea2_range,init_capital,cr,train_range)
            irange = [train_range[1]+1,get_first_date_index(trade_date,y+1)-1]
            capital,shares = trade(_open,gold_cross,dead_cross,_hist,_dea,dea1,dea2,capital,shares,cr,irange)

        y = end_year
        train_range = [get_first_date_index(trade_date,y-window_size),get_first_date_index(trade_date,y)-1]
        _,dea1,dea2 = get_optimal_dea(_open,_close,gold_cross,dead_cross,_hist,_dea,dea1_range,dea2_range,init_capital,cr,train_range)
        irange = [train_range[1]+1,trade_date.shape[0]-1]
        capital,shares = trade(_open,gold_cross,dead_cross,_hist,_dea,dea1,dea2,capital,shares,cr,irange)
        print(f"window_size = {window_size}, net asset = {capital+shares*_close[irange[1]]:.3f}, time cost = {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()

"""
window_size = 1, net asset = 755673.290, time cost = 31.936 s
window_size = 2, net asset = 1011331.694, time cost = 63.718 s
window_size = 3, net asset = 1403988.579, time cost = 97.804 s
window_size = 4, net asset = 1274943.240, time cost = 131.601 s
window_size = 5, net asset = 1815012.200, time cost = 166.826 s
window_size = 6, net asset = 1876097.874, time cost = 202.325 s
window_size = 7, net asset = 905.476, time cost = 239.372 s
"""
