import matplotlib.pyplot as plt
import openpyxl
from matplotlib.dates import YearLocator, DateFormatter

from step1 import np, pd, time, date, DataNode, trade1, calculate_max_drawdown, calculate_sharpe_ratio
from step2 import get_optimal_days
from step4 import get_first_date_index


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"000016"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    sample_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    train_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    day1_range = range(1, 15 + 1)
    day2_range = range(20, 100 + 1)

    # 1 - 处理数据
    data = DataNode(file_path, stock_code)
    start_year, end_year = [d.year for d in train_interval]
    window_years = start_year - sample_interval[0].year  # 窗口长度，单位为年
    sample_range = [np.where(data.trade_date == d)[0][0]
                    for d in sample_interval]
    ibegin = sample_range[1]+1

    # 2 - 模拟交易
    capital = init_capital
    shares = init_shares
    records = []

    length = end_year - start_year+2
    year_interval = [None]*length
    year_na = [0.0]*length  # year net asset list
    year_yield = [0.0]*length

    year_interval[0] = sample_interval[1]
    year_na[0] = init_capital

    for y in range(start_year, end_year):
        s_na, day1, day2 = get_optimal_days(
            data, sample_range, day1_range, day2_range, init_capital, cr)
        train_range = [sample_range[1] + 1,
                       get_first_date_index(data.trade_date, y+1)-1]
        capital, shares, record = trade1(data, train_range, [
            day1, day2], capital, shares, cr)
        records.extend(record)
        # 记录年度信息
        i = y-start_year+1
        year_interval[i] = data.trade_date[train_range[1]]
        year_na[i] = data.net_asset[train_range[1]]
        year_yield[i] = year_na[i]/year_na[i-1]-1
        print(
            f"sample year: {y-window_years}-{y-1}, day1 = {day1}, day2 = {day2}, net asset = {s_na:.3f}\ntrain year: {y}, capital = {capital:.3f}, shares = {shares}, net_asset = {year_na[i]:.3f}\nTime Cost = {time()-s_clk:.3f} s"
        )
        sample_range = [get_first_date_index(
            data.trade_date, y-window_years+1), train_range[1]]

    # 由于训练集的第一年不满一年，这里作以修正
    year_yield[1] = (1+year_yield[1])**(365/364) - 1  # 训练年份的第一年的跨度为364天

    # 处理最后一年
    y = end_year
    s_na, day1, day2 = get_optimal_days(
        data, sample_range, day1_range, day2_range, init_capital, cr)
    train_range[0] = sample_range[1] + 1
    iend = train_range[1] = np.where(
        data.trade_date == train_interval[1])[0][0]
    capital, shares, record = trade1(data, train_range, [
        day1, day2], capital, shares, cr)
    records.extend(record)
    # 记录年度信息
    year_interval[-1] = data.trade_date[iend]
    year_na[-1] = data.net_asset[iend]
    ratio = 365/((train_interval[1]-date(2023, 1, 1)).days+1)
    year_yield[-1] = (year_na[-1]/year_na[-2]) ** ratio - 1
    print(
        f"sample year: {y-window_years}-{y-1}, day1 = {day1}, day2 = {day2}, net asset = {s_na:.3f}\ntrain time: {data.trade_date[train_range[0]].isoformat()}-{train_interval[1].isoformat()}, capital = {capital:.3f}, shares = {shares}, net_asset = {year_na[-1]:.3f}\nTime Cost = {time()-s_clk:.3f} s"
    )

    # 3 - 保存结果
    result_path = r"./../result/02case1.xlsx"
    figure_path = r"./../result/figures/02case1.png"
    iend1 = iend+1  # 使iend不可达，方便后续切片

    # 3.1 - 基本信息：最终收益、最大回撤、夏普比率、程序用时
    final_yield = data.net_asset[iend]/init_capital - 1
    max_drawdown = calculate_max_drawdown(data.net_asset[ibegin:iend1])
    sharpe_ratio = calculate_sharpe_ratio(year_yield[1:])
    df1 = pd.DataFrame({
        "Trade Span": [train_interval[0].isoformat()+" - "+train_interval[1].isoformat()],
        "Final Yield": [final_yield],
        "Max Drawdown": [max_drawdown],
        "Sharpe Ratio": [sharpe_ratio],
        "Time Cost (s)": [0]
    })
    print(f"Generating basic information cost {time()-s_clk:.3f} s")

    # 3.2 - 年度收益率，年化收益率
    df2 = pd.DataFrame({
        "Year": [d.isoformat() for d in year_interval],  # 将日期转换为字符串
        "Net Asset": year_na,
        "Year Yield": year_yield
    })
    print(f"Generating annual yield cost {time()-s_clk:.3f} s")

    # 3.3  - 交易记录
    df3 = pd.DataFrame(records, columns=[
                       "Date", "B/S", "Price", "Shares", "Capital"])
    print(f"Generating trade record cost {time()-s_clk:.3f} s")

    # 3.4 - 日净资产以及与标的涨跌幅对比图
    # 生成表
    df4 = pd.DataFrame({
        "Date": [d.isoformat() for d in data.trade_date[ibegin:iend+1]],
        "Net Asset": data.net_asset[ibegin:iend+1],
    })
    print(f"Generating net asset cost {time()-s_clk:.3f} s")

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
    yearsFmt = DateFormatter("%Y")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    fig.autofmt_xdate()

    plt.savefig(figure_path)
    print(f"Generating figure cost {time()-s_clk:.3f} s")

    # 3.5 - 保存结果
    with pd.ExcelWriter(result_path) as writer:
        df1.to_excel(writer, sheet_name="Basic Info", index=False)
        df2.to_excel(writer, sheet_name="Annual Yield", index=True)
        df3.to_excel(writer, sheet_name="Trade Records", index=True)
        df4.to_excel(writer, sheet_name="Net Asset", index=False)

    # 插入图片
    wb = openpyxl.load_workbook(result_path)
    img = openpyxl.drawing.image.Image(figure_path)
    pos = chr(ord('B')+df3.shape[1])+"1"
    wb["Net Asset"].add_image(img, pos)

    # 插入程序运行时间
    pos = "E2"
    wb["Basic Info"][pos] = time()-s_clk

    # 保存并关闭
    wb.save(result_path)
    wb.close()
    print(f"Writting to Excel cost {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()
