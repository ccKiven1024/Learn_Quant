from math import floor
from time import time
import pandas as pd


def calculate_revenue(df, m_day, start_day, end_day, initial_capital, commission):
    df['mean_day'] = df['Close'].rolling(window=m_day).mean()  # 添加均线
    start_index = df[df['Date'] == start_day].index[0]  # 获取开始日期对应的索引
    end_index = df[df['Date'] == end_day].index[0]  # 获取结束日期对应的索引

    shares = 0  # 购入份数
    capital = initial_capital  # 总资金

    # 执行策略
    for i in range(start_index, end_index+1):
        # 前一日的收盘价大于该日的240日均线 且 没有买入
        if df.at[i-1, 'Close'] > df.at[i-1, 'mean_day'] and shares == 0:
            shares = floor(capital*(1-commission) /
                           df.at[i, 'Open'])  # 计算购入份数
            if (shares < 1):  # 若购买份数不足1，则跳过该日
                shares = 0  # 置零
                continue
            # 全仓买入
            capital -= shares * df.at[i, 'Open']*(1+commission)  # 计算剩余资金
        # 前一日的收盘价低于该日的240日均线 且 持有股票
        elif df.at[i-1, 'Close'] < df.at[i-1, 'mean_day'] and shares > 0:
            # 全仓卖出
            capital += shares * df.at[i, 'Open']*(1-commission)   # 计算剩余资金
            shares = 0  # 重置持有股票份数
    # 检查最后一个交易日是否持有股票，若持有则以收盘价卖出
    if shares > 0:
        capital += shares * df.at[end_index, 'Close']*(1-commission)
        shares = 0
    return capital


def get_optimal_m_day(df, start_day, end_day, initial_capital, commission):
    optimal_m_day = 0.0
    optimal_r = 0.0
    for day in range(120, 241):
        r = calculate_revenue(df, day, start_day, end_day,
                              initial_capital, commission)
        if r > optimal_r:
            optimal_r = r
            optimal_m_day = day
    return optimal_m_day


def main():

    file_path = r'StockData.xlsx'
    df = pd.read_excel(file_path)

    initial_capital = 1e6  # 初始资金
    commission = 5e-4  # 手续费

    train_start_day = ' 2006/01/04'
    train_end_day = ' 2013/12/31'
    test_start_day = ' 2014/01/02'
    test_end_day = ' 2023/08/31'

    t1 = time()
    m_day = get_optimal_m_day(df, train_start_day,
                              train_end_day, initial_capital, commission)
    t2 = time()
    print(f"Training time cost = {t2-t1:.2f}s")
    print(f"The optimal m_day = {m_day}\n")

    r = calculate_revenue(df, m_day, test_start_day,
                          test_end_day, initial_capital, commission)
    t3 = time()
    print(f"Testing time cost = {t3-t2:.2f}s")

    print(
        f" The last capital = {r:.2f} with initial capital = {initial_capital} from {test_start_day} to {test_end_day}")


if __name__ == "__main__":
    main()
