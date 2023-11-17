from step1 import np, time, date, DataNode, trade
from step2 import get_optimal_days


def get_first_date_index(date_arr, year):
    return np.where((date_arr > date(year-1, 12, 31)) & (date_arr < date(year, 1, 7)))[0][0]


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_name = r"399300"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    day1_range = range(1, 15 + 1)
    day2_range = range(20, 100 + 1)
    sample_date_interval = [date(2006, 1, 4), date(2013, 12, 31)]
    train_date_interval = [date(2014, 1, 2), date(2023, 8, 31)]
    sliding_years = 1  # 滑动年数

    # 1 - 处理数据
    data = DataNode(file_path, stock_name)
    start_year, end_year = [d.year for d in train_date_interval]
    window_years = start_year - sample_date_interval[0].year  # 窗口长度，单位为年
    sample_range = [np.where(data.trade_date == d)[0][0]
                    for d in sample_date_interval]

    capital = init_capital
    shares = init_shares
    for y in range(start_year, end_year):
        s_na, day1, day2 = get_optimal_days(
            data, sample_range, day1_range, day2_range, init_capital, cr)
        train_boundary = [sample_range[1] + 1,
                          get_first_date_index(data.trade_date, y+1)-1]
        capital, shares = trade(data, train_boundary, [
                                day1, day2], capital, shares, cr)
        net_asset = capital + shares*data.close[train_boundary[1]]
        print(
            f"sample year: {y-window_years}-{y-1}, day1 = {day1}, day2 = {day2}, net asset = {s_na:.3f}\ntrain year: {y}, capital = {capital:.3f}, shares = {shares}, net_asset = {net_asset:.3f}\nTime Cost = {time()-s_clk:.3f} s\n"
        )
        sample_range = [get_first_date_index(
            data.trade_date, y-window_years+sliding_years), train_boundary[1]]

    # 处理最后一年
    y = end_year
    s_na, day1, day2 = get_optimal_days(
        data, sample_range, day1_range, day2_range, init_capital, cr)
    train_boundary = [sample_range[1] + 1, data.open.shape[0]-1]
    capital, shares = trade(data, train_boundary, [
                            day1, day2], capital, shares, cr)
    net_asset = capital + shares*data.close[train_boundary[1]]
    print(
        f"sample year: {y-window_years}-{y-1}, day1 = {day1}, day2 = {day2}, net asset = {s_na:.3f}\ntrain time: {data.trade_date[train_boundary[0]].isoformat()}-{train_date_interval[1].isoformat()}, capital = {capital:.3f}, shares = {shares}, net_asset = {net_asset:.3f}\nTime Cost = {time()-s_clk:.3f} s\n"
    )


if __name__ == "__main__":
    main()

"""
sample year: 2006-2013, day1 = 7, day2 = 41, net asset = 7576352.119
train year: 2014, capital = 69.869, shares = 389.0, net_asset = 1374683.059
Time Cost = 3.224

sample year: 2007-2014, day1 = 6, day2 = 79, net asset = 4887152.431
train year: 2015, capital = 2816.328, shares = 437.0, net_asset = 1633263.328
Time Cost = 5.274

sample year: 2008-2015, day1 = 6, day2 = 39, net asset = 2570290.728
train year: 2016, capital = 1502158.895, shares = 0, net_asset = 1502158.895
Time Cost = 7.349

sample year: 2009-2016, day1 = 1, day2 = 42, net asset = 3110051.208
train year: 2017, capital = 1652609.276, shares = 0, net_asset = 1652609.276
Time Cost = 9.433

sample year: 2010-2017, day1 = 6, day2 = 30, net asset = 2132626.852
train year: 2018, capital = 1233479.942, shares = 0, net_asset = 1233479.942
Time Cost = 11.490

sample year: 2011-2018, day1 = 1, day2 = 42, net asset = 1758644.517
train year: 2019, capital = 3773.346, shares = 341.0, net_asset = 1400707.126
Time Cost = 13.562

sample year: 2012-2019, day1 = 10, day2 = 25, net asset = 2122122.607
train year: 2020, capital = 2820.644, shares = 287.0, net_asset = 1498460.874
Time Cost = 15.580

sample year: 2013-2020, day1 = 8, day2 = 55, net asset = 2401897.582
train year: 2021, capital = 344.902, shares = 273.0, net_asset = 1349065.912
Time Cost = 17.653

sample year: 2014-2021, day1 = 14, day2 = 46, net asset = 2419951.992
train year: 2022, capital = 3081.918, shares = 350.0, net_asset = 1358152.418
Time Cost = 19.784

sample year: 2015-2022, day1 = 13, day2 = 41, net asset = 1572324.350
train time: 2023-01-03-2023-08-31, capital = 1292834.658, shares = 0, net_asset = 1292834.658
Time Cost = 21.840
"""
