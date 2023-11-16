from exercise4_2 import np, date, time, DataNode, trade, get_optimal_dea
from exercise4_3 import get_first_date_index


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./data/StockData.xlsx"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    test_interval = (date(2014, 1, 2), date(2023, 8, 31))
    dea1_range = dea2_range = range(-100, 101)
    window_years_range = range(1, 8)

    # 1 - 读取数据
    data = DataNode(file_path)

    # 计算年份
    start_year, end_year = [date.year for date in test_interval]

    # 2 - 模拟交易
    for window_size in window_years_range:
        capital = init_capital
        shares = init_shares
        for y in range(start_year, end_year):
            train_range = [get_first_date_index(
                data.trade_date, y-window_size), get_first_date_index(data.trade_date, y)-1]
            _, dea1, dea2 = get_optimal_dea(
                data, train_range, dea1_range, dea2_range, init_capital, cr)
            irange = [train_range[1]+1,
                      get_first_date_index(data.trade_date, y+1)-1]
            capital, shares = trade(
                data, irange, dea1, dea2, shares, capital, cr)

        y = end_year
        train_range = [get_first_date_index(
            data.trade_date, y-window_size), get_first_date_index(data.trade_date, y)-1]
        _, dea1, dea2 = get_optimal_dea(
            data, train_range, dea1_range, dea2_range, init_capital, cr)
        irange = [train_range[1]+1, data.trade_date.shape[0]-1]
        capital, shares = trade(data, irange, dea1, dea2, shares, capital, cr)
        print(
            f"window_size = {window_size}, net asset = {capital+shares*data.close_price[irange[1]]:.3f}, time cost = {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()

"""
window_size = 1, net asset = 755673.290, time cost = 25.526 s
window_size = 2, net asset = 1011331.694, time cost = 50.618 s
window_size = 3, net asset = 1403988.579, time cost = 76.891 s
window_size = 4, net asset = 1274943.240, time cost = 105.019 s
window_size = 5, net asset = 1815012.200, time cost = 135.040 s
window_size = 6, net asset = 1876097.874, time cost = 164.978 s
window_size = 7, net asset = 1144684.350, time cost = 195.916 s
"""
