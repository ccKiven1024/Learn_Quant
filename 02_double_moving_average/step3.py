from step2 import np, time, date, DataNode, trade, get_optimal_days


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"399300"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    sample_date_interval = [date(2006, 1, 4), date(2013, 12, 31)]
    train_date_interval = [date(2014, 1, 2), date(2023, 8, 31)]
    short_range = np.array(range(1, 15 + 1))
    long_range = np.array(range(20, 100 + 1))

    # 1 - 处理数据
    data = DataNode(file_path, stock_code, cr, short_range, long_range)
    sample_range = [np.where(data.trade_date == date)[0][0]
                    for date in sample_date_interval]
    train_range = [np.where(data.trade_date == date)[0][0]
                   for date in train_date_interval]

    # 2- 模拟交易
    # 2.1 - 在样本区间寻找最佳双均线参数
    na, short, long = get_optimal_days(
        data, sample_range, init_capital)
    print(
        f"short = {short}, long = {long}, net asset = {na:.3f}, time cost = {time()-s_clk:.3f} s"
    )
    # short = 7, long = 41, net asset = 7576352.119, time cost = 3.878 s

    # 2.2 应用在训练区间
    c, s = trade(data, train_range, [short, long],
                 init_capital, init_shares)
    print(
        f"net asset = {c+s*data.close[train_range[1]]:.3f} from {train_date_interval[0].isoformat()} to {train_date_interval[1].isoformat()}, time cost = {time()-s_clk:.3f} s")
    # net asset = 1757143.191 from 2014-01-02 to 2023-08-31, time cost = 3.880 s


if __name__ == "__main__":
    main()
