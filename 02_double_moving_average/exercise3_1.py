import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from time import time
from datetime import date
from matplotlib.dates import YearLocator, DateFormatter
from numpy import ndarray


def pre_process_data(file_path: str):
    """
    读取数据，并将其全部设置为全局变量，便于访问
    """
    global trade_date, open_price, close_price
    df = pd.read_excel(file_path)
    trade_date = pd.to_datetime(df["Date"].values).date
    open_price = df["Open"].values
    close_price = df["Close"].values
    high_price = df["High"].values
    low_price = df["Low"].values
    # volume = df["Volume"].values
    return df, high_price, low_price


def calculate_ma(df: pd.DataFrame, ma_set: tuple, irange: tuple[int, int]) -> ndarray:
    """
    计算必要的均线：
    从df["Close"]取出[ibegin-1-ma_set[i],iend+1)部分，计算均线，记为arr1
    从arr1中取[ma_set[i]:iend+1)部分，它对应df["Close"]中的[ibegin-1:iend+1)的均线
    """
    ibegin, iend = irange
    ma = np.zeros(shape=(len(df), len(ma_set)), dtype=np.float64)
    for i in range(len(ma_set)):
        ma[ibegin - 1: iend + 1, i] = (
            df.loc[ibegin - 1 - ma_set[i]: iend + 1, "Close"]
            .rolling(window=ma_set[i])
            .mean()
            .to_numpy()[ma_set[i]: iend + 1]
        )
    return ma


def trade_dm(
    init_capital: float,
    init_shares: float,
    ma: ndarray,
    irange: tuple[int, int],
) -> tuple:
    capital = init_capital
    shares = init_shares
    ibegin, iend = irange
    flag_bs = 0  # 买卖信号
    position = 0  # 持仓情况：1代表多仓，0代表空仓，-1代表少仓
    var_price = 0  # 开盘价/收盘价
    ma1_2d_ago = 0
    ma2_2d_ago = 0
    ma1_1d_ago = 0
    ma2_1d_ago = 0

    # 处理首日
    ma1_1d_ago, ma2_1d_ago = ma[ibegin - 1]
    if ma1_1d_ago > ma2_1d_ago:
        flag_bs = 1
    elif ma1_1d_ago < ma2_1d_ago:
        flag_bs = -1

    if flag_bs == 1 and position == 0:  # 空仓时买入
        var_price = open_price[ibegin]
        shares = int(capital * (1 - cr) / var_price)  # 买入份额
        if shares > 0:
            position = 1
            capital -= shares * var_price * (1 + cr)  # 更新资金
        flag_bs = 0
    elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
        var_price = open_price[ibegin]
        capital += shares * var_price * (1 - cr)  # 更新资金
        shares = 0
        position = 0
        flag_bs = 0

    # 处理(ibegin,iend]之间的日期
    for i in range(ibegin + 1, iend + 1):
        ma1_2d_ago, ma2_2d_ago = ma[i - 2]
        ma1_1d_ago, ma2_1d_ago = ma[i - 1]
        if (ma1_2d_ago < ma2_2d_ago) and (ma1_1d_ago > ma2_1d_ago):
            flag_bs = 1
        elif (ma1_2d_ago > ma2_2d_ago) and (ma1_1d_ago < ma2_1d_ago):
            flag_bs = -1

        if flag_bs == 1 and position == 0:  # 空仓时买入
            var_price = open_price[i]
            shares = int(capital * (1 - cr) / var_price)  # 买入份额
            if shares > 0:
                position = 1
                capital -= shares * var_price * (1 + cr)  # 更新资金
            flag_bs = 0
        elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
            var_price = open_price[i]
            capital += shares * var_price * (1 - cr)  # 更新资金
            shares = 0
            position = 0
            flag_bs = 0
    return (shares, capital, flag_bs)  # 返回最后一天的持仓，资金，买卖信号


def trade_dm1(
    init_capital: float,
    init_shares: float,
    ma: ndarray,
    irange: tuple[int, int],
    net_asset: ndarray,
    records: list,
) -> tuple:
    ibegin, iend = irange
    net_asset[ibegin - 1] = capital = init_capital  # 最开始的资金
    shares = init_shares
    flag_bs = 0  # 买卖信号
    position = 0  # 持仓情况：1代表多仓，0代表空仓，-1代表少仓
    var_price = 0  # 开盘价/收盘价
    ma1_2d_ago = 0
    ma2_2d_ago = 0
    ma1_1d_ago = 0
    ma2_1d_ago = 0

    # 处理首日
    ma1_1d_ago, ma2_1d_ago = ma[ibegin - 1]
    if ma1_1d_ago > ma2_1d_ago:
        flag_bs = 1
    elif ma1_1d_ago < ma2_1d_ago:
        flag_bs = -1

    if flag_bs == 1 and position == 0:  # 空仓时买入
        var_price = open_price[ibegin]
        shares = int(capital * (1 - cr) / var_price)  # 买入份额
        if shares > 0:
            position = 1
            capital -= shares * var_price * (1 + cr)  # 更新资金
            records.append([trade_date[ibegin], flag_bs,
                           var_price, shares, capital])
        flag_bs = 0
    elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
        var_price = open_price[ibegin]
        capital += shares * var_price * (1 - cr)  # 更新资金
        shares = 0
        position = 0
        records.append(
            [trade_date[ibegin].isoformat(), flag_bs, var_price, shares, capital]
        )
        flag_bs = 0
    net_asset[ibegin] = capital + shares * close_price[ibegin]  # 记录当日净资产

    # 处理(ibegin,iend]之间的日期
    for i in range(ibegin + 1, iend + 1):
        ma1_2d_ago, ma2_2d_ago = ma[i - 2]
        ma1_1d_ago, ma2_1d_ago = ma[i - 1]
        if (ma1_2d_ago < ma2_2d_ago) and (ma1_1d_ago > ma2_1d_ago):
            flag_bs = 1
        elif (ma1_2d_ago > ma2_2d_ago) and (ma1_1d_ago < ma2_1d_ago):
            flag_bs = -1

        if flag_bs == 1 and position == 0:  # 空仓时买入
            var_price = open_price[i]
            shares = int(capital * (1 - cr) / var_price)  # 买入份额
            if shares > 0:
                position = 1
                capital -= shares * var_price * (1 + cr)  # 更新资金
                records.append(
                    [trade_date[i].isoformat(), flag_bs, var_price, shares, capital]
                )
            flag_bs = 0
        elif flag_bs == -1 and position == 1:  # 多仓时，直接平仓
            var_price = open_price[i]
            capital += shares * var_price * (1 - cr)  # 更新资金
            shares = 0
            position = 0
            records.append(
                [trade_date[i].isoformat(), flag_bs, var_price, shares, capital]
            )
            flag_bs = 0
        net_asset[i] = capital + shares * close_price[i]  # 记录当日净资产
    return (shares, capital, flag_bs)  # 返回最后一天的持仓，资金，买卖信号


def trade_once(
    init_capital: float,
    init_shares: float,
    ma: ndarray,
    irange: tuple[int, int],
) -> float:
    shares, capital, flag_bs = trade_dm(init_capital, init_shares, ma, irange)
    # 由于只交易一次，需要单独处理最后一天
    iend = irange[1]
    if flag_bs == 1 and shares == 0:  # 空仓时买入
        # 根据T+1规则，当日买入无法卖出，且这样做会扣除手续费，直接拒绝买入
        # 由于没有保存交易记录
        var_price = open_price[iend]  # 找到最后一日的开盘价
        capital += shares * var_price * (1 + cr)  # 添加上手续费，恢复资金
    elif shares > 0:  # 如果持仓
        if flag_bs != -1:  # 如果最后一日没有卖出，则以收盘价卖出
            var_price = close_price[iend]
            capital += shares * var_price * (1 - cr)
        # else:# 如果最后一日是卖出，记录均正确
        #     pass
    return capital


def trade_once1(
    init_capital: float,
    init_shares: float,
    ma: ndarray,
    irange: tuple[int, int],
    net_asset: ndarray,
    records: list,
) -> float:
    shares, capital, flag_bs = trade_dm1(
        init_capital, init_shares, ma, irange, net_asset, records
    )
    # 由于只交易一次，需要单独处理最后一天
    iend = irange[1]
    if flag_bs == 1 and shares == 0:  # 空仓时买入
        # 根据T+1规则，当日买入无法卖出，且这样做会扣除手续费，直接拒绝买入
        records.pop()  # 删除买入记录
        capital = records[-1][-1]  # 可以直接读取记录来恢复资金
        net_asset[iend] = capital  # 修正净资产
    elif shares > 0:  # 如果持仓
        if flag_bs != -1:  # 如果最后一日没有卖出，则以收盘价卖出
            var_price = close_price[iend]
            capital += shares * var_price * (1 - cr)
            records.append(
                [trade_date[iend].isoformat(), -1, var_price, shares, capital]
            )
            net_asset[iend] = capital  # 修正净资产
        # else:# 如果最后一日是卖出，记录均正确
        #     pass
    return capital


def calculate_max_drawdown(h: ndarray, l: ndarray, _irange: tuple) -> float:
    peek_index = _irange[0]
    max_drawdown = (h[_irange[0]] - l[_irange[0]]) / h[_irange[0]]
    for i in range(_irange[0], _irange[1] + 1):
        if h[i] > h[peek_index]:
            peek_index = i
        drawdown = (h[peek_index] - l[i]) / h[peek_index]
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def calculate_annual_compound(
    cur_capital: float, init_capital: float, _date_interval: tuple
) -> float:
    final_yield = (cur_capital - init_capital) / init_capital
    year_span = (_date_interval[1] - _date_interval[0]).days / 365
    annual_compound = (1 + final_yield) ** (1 / year_span) - 1
    return annual_compound


def get_last_date_index(_date_arr: ndarray, year: int) -> int:
    return np.where(
        ((_date_arr > date(year, 12, 25)) & (_date_arr < date(year + 1, 1, 1)))
    )[0][-1]


def main():
    global cr
    s_clk = time()

    excel_path = r"./StockData.xlsx"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4  # commission rate
    date_interval = (date(2006, 1, 4), date(2023, 8, 31))
    ma_set = (5, 20)

    # 1 读取数据
    df, high_price, low_price = pre_process_data(excel_path)
    net_asset = np.zeros_like(close_price, dtype=np.float64)  # 记录净资产
    records = []  # 交易记录
    irange = (
        np.where(trade_date == date_interval[0])[0][0],
        np.where(trade_date == date_interval[1])[0][0],
    )
    print("Reading data costs {:.2f}s".format(time() - s_clk))
    # 2 计算均线
    ma = calculate_ma(df, ma_set, irange)
    df = None  # 释放内存
    print("Calculating ma costs {:.2f}s".format(time() - s_clk))
    # 3 交易
    capital = trade_once1(init_capital, init_shares, ma,
                          irange, net_asset, records)
    print("Trading costs {:.2f}s".format(time() - s_clk))

    # 4 画图
    result_file_path = r"result3_1.xlsx"
    img_path = r"close_vs_net_asset3_1.png"
    ibegin, iend = irange
    # 4.1 生成sheet1：最大回撤，夏普率，年化复利
    final_yield = (capital - init_capital) / init_capital
    max_drawdown = calculate_max_drawdown(high_price, low_price, irange)
    high_price = None  # 释放内存
    low_price = None
    sharp_radio = np.nan  # 由于只有一只股票，无法计算夏普率
    year_span = (date_interval[1] - date_interval[0]).days / 365
    annual_compound = (1 + final_yield) ** (1 / year_span) - 1
    df1 = pd.DataFrame(
        {
            "Max Drawdown": [max_drawdown],
            "Sharp Radio": [sharp_radio],
            "Annual Compound": [annual_compound],
        }
    )
    print(
        f"Calculating Max Drawdown, Sharp Radio, Annual Compound is done, time cost = {time()-s_clk} s"
    )
    # 4.2 生成sheet2：年度收益率
    start_year = date_interval[0].year
    end_year = date_interval[1].year
    length = end_year - start_year + 2
    year_list = [None] * length
    year_net_asset = [0.0] * length
    year_yield = [0.0] * length

    year_list[0] = date_interval[0].isoformat()
    year_net_asset[0] = net_asset[ibegin]
    year_yield[0] = 0.0

    for y in range(start_year, end_year):
        ldi = get_last_date_index(trade_date, y)
        i = y - start_year + 1
        year_list[i] = trade_date[ldi].isoformat()
        year_net_asset[i] = net_asset[ldi]
        year_yield[i] = (year_net_asset[i] - year_net_asset[i - 1]) / year_net_asset[
            i - 1
        ]

    year_list[-1] = date_interval[1].isoformat()
    year_net_asset[-1] = net_asset[iend]
    year_yield[-1] = (year_net_asset[-1] - year_net_asset[-2]
                      ) / year_net_asset[-2]

    df2 = pd.DataFrame(
        {
            "Date": year_list,
            "Net Asset": year_net_asset,
            "Yield": year_yield,
        }
    )
    print(f"Calculating Year Yield is done, time cost = {time()-s_clk} s")

    # 4.3 生成sheet3：交易记录
    df3 = pd.DataFrame(
        records, columns=["Date", "Buy/Sale", "Price", "Shares", "Capital"]
    )
    print(f"Generating trade records is done, time cost = {time()-s_clk} s")

    # 4.4 生成sheet4：日净资产记录
    df4 = pd.DataFrame({"Date": trade_date, "Net Asset": net_asset})
    print(f"Generating net asset is done, time cost = {time()-s_clk} s")

    # 4.5 生成日净资产和收盘价的日涨跌幅对比图
    y1 = ((close_price - close_price[ibegin]) /
          close_price[ibegin])[ibegin: iend + 1]
    y2 = ((net_asset - net_asset[ibegin]) /
          net_asset[ibegin])[ibegin: iend + 1]

    fig, ax = plt.subplots()
    ax.plot(trade_date[ibegin: iend + 1], y1, label="Close Price", color="red")
    ax.plot(trade_date[ibegin: iend + 1], y2, label="Net Asset", color="blue")
    ax.set_title("Colse Price vs. Net Asset Daily Yield")
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield")
    ax.legend(loc="upper left")

    years = YearLocator()
    yearsfmt = DateFormatter("%Y")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsfmt)
    fig.autofmt_xdate()

    fig.savefig(img_path)
    print(f"Generating image is done, time cost = {time()-s_clk} s")

    # 4.6 将结果写入Excel
    with pd.ExcelWriter(result_file_path) as writer:
        df1.to_excel(writer, sheet_name="Sheet1", index=False)
        df2.to_excel(writer, sheet_name="Every Year Yield", index=True)
        df3.to_excel(writer, sheet_name="Trade Records", index=True)
        df4.to_excel(writer, sheet_name="Net Asset", index=False)

    # 将图片插入到Sheet4中
    wb = openpyxl.load_workbook(result_file_path)
    img = openpyxl.drawing.image.Image(img_path)
    wb["Net Asset"].add_image(img, f"{chr(ord('B')+df4.shape[1])}1")

    # 在Sheet1中写入运行时间
    wb["Sheet1"][f"{chr(ord('A')+df1.shape[1])}1"] = f"Time Cost = {time()-s_clk} s"
    wb.save(result_file_path)
    wb.close()
    print(f"Writing to Excel is done, time cost = {time()-s_clk} s")


if __name__ == "__main__":
    main()
