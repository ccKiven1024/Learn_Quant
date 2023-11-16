import time
from datetime import datetime
import pandas as pd


def trade_stocks(df, m_day, capital, shares, commission, irange):
    ibegin = irange[0]  # 获取开始日期对应的索引
    iend = irange[1]  # 获取结束日期对应的索引
    df.loc[ibegin - 1 : iend + 1, "mean_day"] = (
        df["Close"].rolling(window=m_day).mean()
    )  # 计算必要部分均线
    td = []  # Transcation Data

    # 临时变量
    flag_bs = 0  # 买卖标志位，0为不买卖，1为买入，-1为卖出
    close_2d_ago = 0
    close_1d_ago = 0
    mean_2d_ago = 0
    mean_1d_ago = 0

    # 处理首日
    close_1d_ago = df.at[ibegin - 1, "Close"]
    mean_1d_ago = df.at[ibegin - 1, "mean_day"]
    if close_1d_ago > mean_1d_ago:
        flag_bs = 1  # 发出买入信号
    elif close_1d_ago < mean_1d_ago:
        flag_bs = -1  # 发出卖出信号

    # 在未持仓时买入
    if flag_bs == 1 and shares == 0:
        shares = int(capital * (1 - commission) / df.at[ibegin, "Open"])  # 计算购入份数
        if shares > 0:  # 如果购买份数大于0
            capital -= shares * df.at[ibegin, "Open"] * (1 + commission)  # 计算剩余资金
        td.append(
            [df.at[ibegin, "Date"], flag_bs, df.at[ibegin, "Open"], shares, capital]
        )
    # 在持仓时卖出
    elif flag_bs == -1 and shares > 0:
        capital += shares * df.at[ibegin, "Open"] * (1 - commission)  # 计算剩余资金
        shares = 0  # 重置持仓份数
        td.append(
            [df.at[ibegin, "Date"], flag_bs, df.at[ibegin, "Open"], shares, capital]
        )
    flag_bs = 0  # 重置买卖标志位

    # 处理(ibegin,iend]之间的日期
    for i in range(ibegin + 1, iend + 1):
        close_2d_ago = df.at[i - 2, "Close"]
        close_1d_ago = df.at[i - 1, "Close"]
        mean_2d_ago = df.at[i - 2, "mean_day"]
        mean_1d_ago = df.at[i - 1, "mean_day"]
        if close_2d_ago <= mean_2d_ago and close_1d_ago > mean_1d_ago:  # 如果收盘价上穿均线
            flag_bs = 1  # 发出买入信号
        if close_2d_ago >= mean_2d_ago and close_1d_ago < mean_1d_ago:  # 如果收盘价下穿均线
            flag_bs = -1  # 发出卖出信号

        # 在未持仓时买入
        if flag_bs == 1 and shares == 0:
            shares = int(capital * (1 - commission) / df.at[i, "Open"])  # 计算购入份数
            if shares > 0:  # 如果购买份数大于0
                capital -= shares * df.at[i, "Open"] * (1 + commission)  # 计算剩余资金
            td.append([df.at[i, "Date"], flag_bs, df.at[i, "Open"], shares, capital])
        # 在持仓时卖出
        if flag_bs == -1 and shares > 0:
            capital += shares * df.at[i, "Open"] * (1 - commission)  # 计算剩余资金
            shares = 0  # 重置持仓份数
            td.append([df.at[i, "Date"], flag_bs, df.at[i, "Open"], shares, capital])
        flag_bs = 0  # 重置买卖标志位

    # 计算净资产
    net_asset = capital + shares * df.at[iend, "Close"]
    return (capital, shares, net_asset, td)


def get_optimal_m_day(df, capital, shares, commission, irange):
    optimal_m_day = 120
    max_ns = 0.0
    for day in range(120, 241):
        ns = trade_stocks(df, day, capital, shares, commission, irange)[2]
        if ns > max_ns:
            max_ns = ns
            optimal_m_day = day
    return (optimal_m_day, max_ns)


def get_first_date_index(df, year):
    return df[
        (df["Date"] > f" {year-1}/12/31") & (df["Date"] <= f" {year}/01/07")
    ].index[0]


def get_last_date_index(df, year):
    return df[
        (df["Date"] >= f" {year}/12/25") & (df["Date"] < f" {year+1}/01/01")
    ].index[-1]


def split_year(date):
    return int(date.split("/")[0])


def main():
    s_clk = time.time()  # 记录开始时间

    result_path = r"result2_4.xlsx"
    data_path = r"StockData.xlsx"
    df = pd.read_excel(data_path)

    initial_capital = 1e6  # 初始资金
    initial_shares = 0  # 持有份数
    commission = 5e-4  # 手续费
    inital_date_interval = (" 2006/01/04", " 2013/12/31")  # 初始日期区间
    sliding_years = 1  # 滑动年数
    end_date = " 2023/08/31"  # 结束日期

    start_year = split_year(inital_date_interval[1]) + 1  # 获取开始年份
    end_year = split_year(end_date)  # 获取结束年份
    window_years = start_year - split_year(inital_date_interval[0])  # 窗口长度，单位为年
    sample_irange = [
        df[df["Date"] == inital_date_interval[0]].index[0],
        df[df["Date"] == inital_date_interval[1]].index[0],
    ]
    start_date = df.at[sample_irange[1] + 1, "Date"]  # 获取开始日期

    capital = initial_capital
    shares = initial_shares
    net_asset = 0.0
    sheet1 = []
    sheet2 = []

    for y in range(start_year, end_year):
        # 先用样本年份计算最优窗口长度
        m_day, sns = get_optimal_m_day(df, capital, shares, commission, sample_irange)
        # 获取当年的交易区间的索引范围
        irange = [sample_irange[1] + 1, get_last_date_index(df, y)]
        # 应用到下一年
        capital, shares, net_asset, td = trade_stocks(
            df, m_day, capital, shares, commission, irange
        )
        sheet1.append([f"{y-window_years}-{y-1}", m_day, sns, f"{y}", net_asset])
        sheet2.extend(td)
        # 滑动窗口
        sample_irange[0] = get_first_date_index(df, y - window_years + 1)
        sample_irange[1] = irange[1]

    # 处理最后一年
    m_day, sns = get_optimal_m_day(df, capital, shares, commission, sample_irange)
    irange = [sample_irange[1] + 1, df[df["Date"] == end_date].index[0]]
    capital, shares, net_asset, td = trade_stocks(
        df, m_day, capital, shares, commission, irange
    )
    # 如果持有，则以收盘价卖出
    if shares > 0:
        # 因为算过净资产，这里减掉手续费，作以更正
        net_asset -= shares * df.at[irange[1], "Close"] * commission
        capital = net_asset
        shares = 0
    # 计算复利收益率
    num_day = (
        datetime.strptime(end_date.strip(), "%Y/%m/%d")
        - datetime.strptime(start_date.strip(), "%Y/%m/%d")
    ).days + 1
    ci = (net_asset / initial_capital) ** (365.0 / num_day) - 1

    sheet1.append(
        [f"{end_year-window_years}-{end_year-1}", m_day, sns, f"{end_year}", net_asset]
    )
    sheet2.extend(td)

    e_clk = time.time()  # 记录结束时间

    # 保存结果至Excel
    df1 = pd.DataFrame(
        sheet1,
        columns=[
            "Sample Years Interval",
            "Optimal Mean Day",
            "Net Asset of Sample Years",
            "Training Year",
            "Net Asset",
        ],
    )
    df1.loc[len(df1)] = [
        f"Time Cost = {e_clk - s_clk} s",
        f"Compound Interest = {ci:%}",
        "",
        "",
        "",
    ]
    df2 = pd.DataFrame(
        sheet2,
        columns=[
            "Date",
            "Buy/Sell",
            "Price",
            "Shares",
            "Capital",
        ],
    )
    with pd.ExcelWriter(result_path) as writer:
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Sheet2", index=True)


if __name__ == "__main__":
    main()
