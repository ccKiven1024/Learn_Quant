from step1 import np, time, date, DataNode, trade
import multiprocessing as mp


def func(data: DataNode, boundary, day_set, _capital, _shares, cr):
    c, s = trade(data, boundary, day_set, _capital, _shares, cr)
    na = c + s*data.close[boundary[1]]
    return (na, day_set[0], day_set[1])


def get_optimal_days(data: DataNode, boundary, day1_range, day2_range, _capital, cr):
    with mp.Pool() as pool:
        res = pool.starmap(func, [
            (data, boundary, [day1, day2], _capital, 0, cr)
            for day1 in day1_range
            for day2 in day2_range
        ])
    return max(res, key=lambda x: x[0])


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_name = r"399300"
    init_capital = 1e6
    commission = 5e-4
    date_interval = [date(2006, 1, 4), date(2023, 8, 31)]
    day1_range = range(1, 15 + 1)
    day2_range = range(20, 100 + 1)

    # 1 - 处理数据
    data = DataNode(file_path, stock_name)
    boundary = list(map(lambda d: np.where(
        data.trade_date == d)[0][0], date_interval))

    na, day1, day2 = get_optimal_days(
        data, boundary, day1_range, day2_range, init_capital, commission)
    print(
        f"day1={day1}, day2={day2}, net asset = {na:.3f}, time cost = {time()-s_clk:.3f} s")
    # day1=7, day2=41, net asset = 13316466.334, time cost = 5.359 s


if __name__ == "__main__":
    main()
