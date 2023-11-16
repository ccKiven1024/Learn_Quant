from exercise4_2 import np, date, time, DataNode, trade, get_optimal_dea


def get_first_date_index(date_arr, year):
    return np.where(
        ((date_arr > date(year-1, 12, 31)) & (date_arr < date(year, 1, 7)))
    )[0][0]


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    train_interval = (date(2006, 1, 4), date(2013, 12, 31))
    test_interval = (date(2014, 1, 2), date(2023, 8, 31))
    dea1_range = dea2_range = range(-100, 101)

    # 1 - 处理数据
    data = DataNode(file_path)

    # 计算范围等
    sample_range = list(map(lambda date: np.where(
        data.trade_date == date)[0][0], train_interval))
    start_year, end_year = [date.year for date in test_interval]
    window_yrs = start_year - train_interval[0].year

    # 2 模拟交易
    capital = init_capital
    shares = init_shares
    for y in range(start_year, end_year):
        _, dea1, dea2 = get_optimal_dea(
            data, sample_range, dea1_range, dea2_range, init_capital, cr)
        irange = [sample_range[1]+1,
                  get_first_date_index(data.trade_date, y+1)-1]
        capital, shares = trade(data, irange, dea1, dea2, shares, capital, cr)

        print(
            f"year: {y}, dea1 = {dea1}, dea2 = {dea2}, capital = {capital:.3f}, shares = {shares}, net_asset = {capital+shares*data.close_price[irange[1]]:.3f}, time cost = {time()-s_clk:.3f} s")

        sample_range = [get_first_date_index(
            data.trade_date, y-window_yrs+1), irange[1]]

    # 处理最后一年
    y = end_year
    _, dea1, dea2 = get_optimal_dea(
        data, sample_range, dea1_range, dea2_range, init_capital, cr)
    irange = [sample_range[1]+1, data.trade_date.shape[0]-1]
    capital, shares = trade(data, irange, dea1, dea2, shares, capital, cr)
    print(
        f"year: {y}, dea1 = {dea1}, dea2 = {dea2}, capital = {capital:.3f}, shares = {shares}, net_asset = {capital+shares*data.close_price[irange[1]]:.3f}, time cost = {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()

"""
year: 2014, dea1 = 71, dea2 = -11, capital = 1195517.504, shares = 0, net_asset = 1195517.504, time cost = 4.105 s
year: 2015, dea1 = 89, dea2 = -11, capital = 1381678.688, shares = 0, net_asset = 1381678.688, time cost = 6.950 s
year: 2016, dea1 = 71, dea2 = -100, capital = 1517931.303, shares = 0, net_asset = 1517931.303, time cost = 10.060 s
year: 2017, dea1 = 71, dea2 = 0, capital = 2114.761, shares = 410, net_asset = 1654767.361, time cost = 12.989 s
year: 2018, dea1 = -82, dea2 = 46, capital = 665.719, shares = 510, net_asset = 1536097.219, time cost = 16.175 s
year: 2019, dea1 = -70, dea2 = 58, capital = 1916450.548, shares = 0, net_asset = 1916450.548, time cost = 19.203 s
year: 2020, dea1 = -63, dea2 = 58, capital = 2406372.492, shares = 0, net_asset = 2406372.492, time cost = 22.363 s
year: 2021, dea1 = -63, dea2 = 46, capital = 2490489.296, shares = 0, net_asset = 2490489.296, time cost = 25.255 s
year: 2022, dea1 = 28, dea2 = 34, capital = 2239608.331, shares = 0, net_asset = 2239608.331, time cost = 28.227 s
year: 2023, dea1 = -63, dea2 = 46, capital = 2239608.331, shares = 0, net_asset = 2239608.331, time cost = 31.314 s
"""
