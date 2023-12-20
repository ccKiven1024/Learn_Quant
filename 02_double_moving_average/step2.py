from step1 import np, time, date, DataNode, trade
import multiprocessing as mp


def func(data: DataNode, boundary, day_set, _capital, _shares):
    c, s = trade(data, boundary, day_set, _capital, _shares)
    na = c + s*data.close[boundary[1]]
    return (na, day_set[0], day_set[1])


def get_optimal_days(data: DataNode, boundary,  _capital):
    with mp.Pool() as pool:
        res = pool.starmap(func, [
            (data, boundary, [short, long], _capital, 0)
            for short in data.short_range
            for long in data.long_range
        ])
    return max(res, key=lambda x: x[0])


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"399300"
    init_capital = 1e6
    cr = 5e-4
    date_interval = [date(2006, 1, 4), date(2023, 8, 31)]
    short_range = np.array(range(1, 15 + 1))
    long_range = np.array(range(20, 100 + 1))

    # 1 - 处理数据
    data = DataNode(file_path, stock_code, cr, short_range, long_range)
    boundary = list(map(lambda d: np.where(
        data.trade_date == d)[0][0], date_interval))

    na, short, long = get_optimal_days(
        data, boundary,  init_capital)
    print(
        f"short={short}, long={long}, net asset = {na:.3f}, time cost = {time()-s_clk:.3f} s")
    # short=7, long=41, net asset = 13316466.334, time cost = 3.889 s


if __name__ == "__main__":
    main()
