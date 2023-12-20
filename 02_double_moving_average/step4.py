from step1 import np, time, date, DataNode, trade
from step2 import get_optimal_days


def get_first_date_index(date_arr, year):
    """
    寻找一年中第一天的索引
    """
    return np.where((date_arr > date(year-1, 12, 31)) & (date_arr < date(year, 1, 7)))[0][0]


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"399300"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    train_interval = [date(2006, 1, 4), date(2013, 12, 31)]
    test_interval = [date(2014, 1, 2), date(2023, 8, 31)]
    sliding_years = 1  # 滑动年数
    short_range = np.array(range(1, 15 + 1))
    long_range = np.array(range(20, 100 + 1))

    # 1 - 处理数据
    data = DataNode(file_path, stock_code, cr, short_range, long_range)
    start_year, end_year = [d.year for d in test_interval]
    window_years = start_year - train_interval[0].year  # 窗口长度，单位为年
    train_range = [np.where(data.trade_date == d)[0][0]
                   for d in train_interval]

    # 2 - 模拟交易
    capital = init_capital
    shares = init_shares
    for y in range(start_year, end_year):
        s_na, short, long = get_optimal_days(data, train_range, init_capital)
        test_range = [train_range[1] + 1,
                      get_first_date_index(data.trade_date, y+1)-1]
        capital, shares = trade(
            data, test_range, [short, long], capital, shares)
        net_asset = capital + shares*data.close[test_range[1]]
        print(
            f"sample year: {y-window_years}-{y-1}, short = {short}, long = {long}, net asset = {s_na:.3f}\ntrain year: {y}, capital = {capital:.3f}, shares = {shares}, net_asset = {net_asset:.3f}\nTime Cost = {time()-s_clk:.3f} s\n"
        )
        train_range = [get_first_date_index(
            data.trade_date, y-window_years+sliding_years), test_range[1]]

    # 处理最后一年
    y = end_year
    s_na, short, long = get_optimal_days(data, train_range, init_capital)
    test_range[0] = train_range[1] + 1
    test_range[1] = np.where(
        data.trade_date == test_interval[1])[0][0]
    capital, shares = trade(data, test_range, [short, long], capital, shares)
    net_asset = capital + shares*data.close[test_range[1]]
    print(
        f"sample year: {y-window_years}-{y-1}, short = {short}, long = {long}, net asset = {s_na:.3f}\ntrain time: {data.trade_date[test_range[0]].isoformat()}-{test_interval[1].isoformat()}, capital = {capital:.3f}, shares = {shares}, net_asset = {net_asset:.3f}\nTime Cost = {time()-s_clk:.3f} s\n"
    )


if __name__ == "__main__":
    main()

"""
sample year: 2006-2013, short = 7, long = 41, net asset = 7576352.119
train year: 2014, capital = 69.869, shares = 389.0, net_asset = 1374683.059
Time Cost = 3.224

sample year: 2007-2014, short = 6, long = 79, net asset = 4887152.431
train year: 2015, capital = 2816.328, shares = 437.0, net_asset = 1633263.328
Time Cost = 5.274

sample year: 2008-2015, short = 6, long = 39, net asset = 2570290.728
train year: 2016, capital = 1502158.895, shares = 0, net_asset = 1502158.895
Time Cost = 7.349

sample year: 2009-2016, short = 1, long = 42, net asset = 3110051.208
train year: 2017, capital = 1652609.276, shares = 0, net_asset = 1652609.276
Time Cost = 9.433

sample year: 2010-2017, short = 6, long = 30, net asset = 2132626.852
train year: 2018, capital = 1233479.942, shares = 0, net_asset = 1233479.942
Time Cost = 11.490

sample year: 2011-2018, short = 1, long = 42, net asset = 1758644.517
train year: 2019, capital = 3773.346, shares = 341.0, net_asset = 1400707.126
Time Cost = 13.562

sample year: 2012-2019, short = 10, long = 25, net asset = 2122122.607
train year: 2020, capital = 2820.644, shares = 287.0, net_asset = 1498460.874
Time Cost = 15.580

sample year: 2013-2020, short = 8, long = 55, net asset = 2401897.582
train year: 2021, capital = 344.902, shares = 273.0, net_asset = 1349065.912
Time Cost = 17.653

sample year: 2014-2021, short = 14, long = 46, net asset = 2419951.992
train year: 2022, capital = 3081.918, shares = 350.0, net_asset = 1358152.418
Time Cost = 19.784

sample year: 2015-2022, short = 13, long = 41, net asset = 1572324.350
train time: 2023-01-03-2023-08-31, capital = 1292834.658, shares = 0, net_asset = 1292834.658
Time Cost = 21.840
"""
