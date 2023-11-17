from step2 import np, time, date, DataNode, trade, get_optimal_days


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_name = r"399300"
    init_capital = 1e6
    commission = 5e-4

    day1_range = range(1, 15 + 1)
    day2_range = range(20, 100 + 1)
    sample_date_interval = [date(2006, 1, 4), date(2013, 12, 31)]
    train_date_interval = [date(2014, 1, 2), date(2023, 8, 31)]

    # 1 - 处理数据
    data = DataNode(file_path, stock_name)
    sample_irange = [np.where(data.trade_date == date)[0][0]
                     for date in sample_date_interval]
    train_irange = [np.where(data.trade_date == date)[0][0]
                    for date in train_date_interval]

    # 2- 模拟交易
    # 2.1 - 在样本区间寻找最佳双均线参数
    na, day1, day2 = get_optimal_days
    print(
        f"md1={md1}, md2={md2}, c={c} from {sample_date_interval[0]} to {sample_date_interval[1]}"
    )
    # md1=7, md2=41, c=7576352.119150016

    # 应用在训练区间
    c = trade_once(df, md1, md2, init_capital,
                   init_shares, commission, train_irange)
    print(
        f"capital = {c} from {train_date_interval[0]} to {train_date_interval[1]}")
    # capital = 1757143.190505008


if __name__ == "__main__":
    main()
