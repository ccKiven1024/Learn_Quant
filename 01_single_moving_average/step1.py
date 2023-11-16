from math import floor
import pandas as pd


def calculate_profit(df, m_day, start_day, end_day, initial_capital, commission):
    df["mean_day"] = df["Close"].rolling(window=m_day).mean()  # 添加均线
    start_index = df[df["Date"] == start_day].index[0]  # 获取开始日期对应的索引
    end_index = df[df["Date"] == end_day].index[0]  # 获取结束日期对应的索引

    shares = 0  # 购入份数
    capital = initial_capital  # 总资金
    purchase_day = []  # 买入日期对应的索引
    sale_day = []  # 卖出日期对应的索引

    # 执行策略
    for i in range(start_index, end_index + 1):
        # 前一日的收盘价大于该日的240日均线 且 没有买入
        if df.at[i - 1, "Close"] > df.at[i - 1, "mean_day"] and shares == 0:
            shares = floor(capital * (1 - commission) / df.at[i, "Open"])  # 计算购入份数
            if shares < 1:  # 若购买份数不足1，则跳过该日
                shares = 0  # 置零，以便下次循环重新计算
                continue
            # 全仓买入
            capital -= shares * df.at[i, "Open"] * (1 + commission)  # 计算剩余资金
            purchase_day.append(i)
        # 前一日的收盘价低于该日的240日均线 且 持有股票
        elif df.at[i - 1, "Close"] < df.at[i - 1, "mean_day"] and shares > 0:
            # 全仓卖出
            capital += shares * df.at[i, "Open"] * (1 - commission)  # 计算剩余资金
            shares = 0  # 重置持有股票份数
            sale_day.append(i)
    # 检查最后一个交易日是否持有股票，若持有则以收盘价卖出
    if shares > 0:
        capital += shares * df.at[end_index, "Close"] * (1 - commission)
        shares = 0
        sale_day.append(end_index)

    # 打印最后10次交易
    for i in range(min(10, len(purchase_day))):
        print(
            f"The last {i+1:2}th purchase date is {df.at[purchase_day[-i-1], 'Date']}\
            and sale date is {df.at[sale_day[-i-1], 'Date']}"
        )
    return capital


def main():
    file_path = r"StockData.xlsx"
    df = pd.read_excel(file_path)
    m_day = 240
    start_day = " 2006/01/04"
    end_day = " 2023/08/31"
    start_index = df[df["Date"] == start_day].index[0]  # 获取开始日期对应的索引
    initial_capital = 1e6  # 初始资金
    commission = 5e-4  # 手续费

    capital = calculate_profit(
        df, m_day, start_day, end_day, initial_capital, commission
    )

    print(
        f"Last capital = {capital:.2f} with initial capital = {initial_capital:.2f} from {start_day} to {end_day}"
    )


if __name__ == "__main__":
    main()
