from math import floor
import pandas as pd
from time import time


def calculate_revenue(df, m_day, start_day, end_day, initial_capital, commission):
    df["mean_day"] = df["Close"].rolling(window=m_day).mean()  # 添加均线
    start_index = df[df["Date"] == start_day].index[0]  # 获取开始日期对应的索引
    end_index = df[df["Date"] == end_day].index[0]  # 获取结束日期对应的索引

    shares = 0  # 购入份数
    capital = initial_capital  # 总资金

    # 执行策略
    for i in range(start_index, end_index + 1):
        # 前一日的收盘价大于该日的日均线 且 没有买入
        if df.at[i - 1, "Close"] > df.at[i - 1, "mean_day"] and shares == 0:
            shares = floor(capital * (1 - commission) / df.at[i, "Open"])  # 计算购入份数
            if shares < 1:  # 若购买份数不足1，则跳过该日
                continue
            # 全仓买入
            capital -= shares * df.at[i, "Open"] * (1 + commission)  # 计算剩余资金
        # 前一日的收盘价低于该日的日均线 且 持有股票
        elif df.at[i - 1, "Close"] < df.at[i - 1, "mean_day"] and shares > 0:
            # 全仓卖出
            capital += shares * df.at[i, "Open"] * (1 - commission)  # 计算剩余资金
            shares = 0  # 重置持有股票份数
    # 检查最后一个交易日是否持有股票，若持有则以收盘价卖出
    if shares > 0:
        capital += shares * df.at[end_index, "Close"] * (1 - commission)
        shares = 0
    return capital


def main():
    s_clk = time()

    file_path = r"StockData.xlsx"
    df = pd.read_excel(file_path)

    start_day = " 2006/01/04"
    end_day = " 2023/08/31"
    initial_capital = 1e6  # 初始资金
    commission = 5e-4  # 手续费

    r_list = []
    for day in range(120, 241):
        r = calculate_revenue(df, day, start_day, end_day, initial_capital, commission)
        r_list.append(r)

    e_clk = time()
    print(f"Time cost: {e_clk-s_clk:.2f}s")

    # 写入2-2.csv
    with open("2_2.csv", "w") as f:
        for i in range(len(r_list)):
            f.write(f"{120+i:3},{r_list[i]:.2f}\n")
    print("Data has been written in 2-2.csv successfully.")


if __name__ == "__main__":
    main()
