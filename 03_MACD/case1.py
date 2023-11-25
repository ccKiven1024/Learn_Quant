import matplotlib.pyplot as plt
import openpyxl
from matplotlib.dates import YearLocator, DateFormatter

from step2 import pd, np, date, time, DataNode,  trade1, get_optimal_dea, calculate_max_drawdown, calculate_sharpe_ratio
from step3 import get_first_date_index


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"000016"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    sample_date_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    train_date_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    dea1_range = dea2_range = range(-100, 101)

    # 1 - 处理数据
    data = DataNode(file_path, stock_code)

    # 计算范围等
    sample_range = list(map(lambda date: np.where(
        data.trade_date == date)[0][0], sample_date_interval))
    start_year, end_year = [date.year for date in train_date_interval]
    window_yrs = start_year - sample_date_interval[0].year
    ibegin = sample_range[1]+1

    # 2 - 模拟交易
    records = []
    capital = init_capital
    shares = init_shares
    for y in range(start_year, end_year):
        sna, dea1, dea2 = get_optimal_dea(
            data, sample_range, dea1_range, dea2_range, init_capital, cr)
        train_range = [sample_range[1]+1,
                       get_first_date_index(data.trade_date, y+1)-1]
        capital, shares, record = trade1(
            data, train_range, dea1, dea2,  shares, capital, cr)
        records.extend(record)
        print(
            f"sample year: {y-window_yrs}-{y-1}, dea1 = {dea1}, dea2 = {dea2}, net asset = {sna:.3f}\ntrain year: {y}, capital = {capital:.3f}, shares = {shares}, net_asset = {data.net_asset[train_range[1]]:.3f}, time cost = {time()-s_clk:.3f} s")

        sample_range = [get_first_date_index(
            data.trade_date, y-window_yrs+1), train_range[1]]

    # 处理最后一年
    y = end_year
    sna, dea1, dea2 = get_optimal_dea(
        data, sample_range, dea1_range, dea2_range, init_capital, cr)
    train_range[0] = sample_range[1]+1
    iend = train_range[1] = np.where(
        data.trade_date == train_date_interval[1])[0][0]
    capital, shares, record = trade1(
        data, train_range, dea1, dea2, shares, capital, cr)
    records.extend(record)
    print(
        f"sample year: {y-window_yrs}-{y-1}, dea1 = {dea1}, dea2 = {dea2}, net asset = {sna:.3f}\ntrain time: {data.trade_date[train_range[0]].isoformat()} - {train_date_interval[1].isoformat()}, capital = {capital:.3f}, shares = {shares}, net_asset = {data.net_asset[train_range[1]]:.3f}, time cost = {time()-s_clk:.3f} s")

    # 3 - 保存结果
    result_path = r"./../result/03case1.xlsx"
    figure_path = r"./../result/figures/03case1.png"
    # 3.1 - 基本信息：最终收益、最大回撤、夏普比率、程序用时
    final_yield = data.net_asset[iend]/init_capital - 1
    max_drawdown = calculate_max_drawdown(data.net_asset[ibegin:iend+1])
    sharpe_ratio = calculate_sharpe_ratio(
        data.close[ibegin:iend+1], final_yield)
    df1 = pd.DataFrame({
        "Trade Span": [train_date_interval[0].isoformat()+" - "+train_date_interval[1].isoformat()],
        "Final Yield": [final_yield],
        "Max Drawdown": [max_drawdown],
        "Sharp Ratio": [sharpe_ratio],
        "Cost Time (s)": [0]
    })
    print(f"Generating basic information costs = {time()-s_clk:.3f} s")

    # 3.2 - 年度收益率，年化收益率
    length = end_year - start_year+2
    year_list = [None]*length
    year_na = [0.0]*length  # year net asset list
    year_yield = [0.0]*length
    acy_list = [0.0]*length  # annual compound yield list

    year_list[0] = train_date_interval[0].isoformat()
    year_na[0] = data.net_asset[ibegin]

    for y in range(start_year, end_year):
        # last date index in one year
        ldi = get_first_date_index(data.trade_date, y+1)-1
        i = y-start_year+1
        year_list[i] = data.trade_date[ldi]
        year_na[i] = data.net_asset[ldi]
        ratio = 365/((year_list[i]-year_list[i-1]).days+1)  # 一年的总日数/实际交易的跨度
        year_yield[i] = (year_na[i]/year_na[i-1]-1) * ratio

        acy_list[i] = (year_na[i]/init_capital)**(1/i)-1

    year_list[-1] = train_date_interval[1].isoformat()
    year_na[-1] = data.net_asset[iend]
    # 由于不是完整一年，这里用比例换算
    span1 = (data.trade_date[train_range[1]] -
             data.trade_date[train_range[0]]).days/365
    year_yield[-1] = (year_na[-1]/year_na[-2]-1)/span1
    span2 = (train_date_interval[1]-train_date_interval[0]).days/365
    acy_list[-1] = (1+year_yield[-1])**(1/span2)-1

    df2 = pd.DataFrame({
        "Year": year_list,
        "Net Asset": year_na,
        "Year Yield": year_yield,
        "Annual Compound Yield": acy_list
    })
    print(f"Generating annual yield costs = {time()-s_clk:.3f} s")

    # 3.3  - 交易记录
    df3 = pd.DataFrame(records, columns=[
                       "Date", "B/S", "Price", "Shares", "Capital"])
    print(f"Generating trade records costs = {time()-s_clk:.3f} s")

    # 3.4 - 日净资产以及与标的涨跌幅对比图
    # 生成表
    df4 = pd.DataFrame({
        "Date": [d.isoformat() for d in data.trade_date[ibegin:iend+1]],
        "Net Asset": data.net_asset[ibegin:iend+1],
    })

    # 绘制日净资产与标的涨跌幅对比图
    y1 = data.close[ibegin:iend+1]/data.close[ibegin-1]-1
    y2 = data.net_asset[ibegin:iend+1]/init_capital-1

    fig, ax = plt.subplots()
    ax.plot(data.trade_date[ibegin:iend+1], y1,
            label="Close Price", color="blue")
    ax.plot(data.trade_date[ibegin:iend+1], y2, label="Net Asset", color="red")
    ax.set_title("Close Price vs. Net Asset")
    ax.set_xlabel("Year")
    ax.set_ylabel("Daily Yield")
    ax.legend(loc="upper left")

    years = YearLocator()
    yearFmt = DateFormatter("%Y")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearFmt)
    fig.autofmt_xdate()

    plt.savefig(figure_path)
    print(f"Generating net asset figure costs = {time()-s_clk:.3f} s")

    # 3.5 - 保存结果
    with pd.ExcelWriter(result_path) as writer:
        df1.to_excel(writer, sheet_name="Basic Info", index=False)
        df2.to_excel(writer, sheet_name="Annual Yield", index=True)
        df3.to_excel(writer, sheet_name="Trade Records", index=True)
        df4.to_excel(writer, sheet_name="Net Asset", index=False)

    # 插入图片
    wb = openpyxl.load_workbook(result_path)
    img = openpyxl.drawing.image.Image(figure_path)
    pos = chr(ord('B')+df4.shape[1])+"1"
    wb["Net Asset"].add_image(img, pos)

    # 插入程序运行时间
    pos = "E2"
    wb["Basic Info"][pos] = time()-s_clk

    # 保存
    wb.save(result_path)
    wb.close()
    print(f"Saving result costs = {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()
