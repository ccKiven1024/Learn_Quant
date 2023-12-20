from dateutil.relativedelta import relativedelta

from step1 import np, time, date, DataNode, trade
from step2 import get_optimal_days
from case2_1 import get_last_index


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"000016"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    train_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    test_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    short_range = np.array(range(1, 21))
    long_range = np.array(range(5, 121))
    step = 1  # 单位为月
    window_range = range(1, 109)  # 窗口长度范围为[1,108]月

    # 1 - 处理数据
    data = DataNode(file_path, stock_code, cr, short_range, long_range)
    date_arr = data.trade_date.astype("datetime64[M]")  # 将日期数组转为月份数组
    delta = relativedelta(test_interval[1], test_interval[0])
    month_difference = delta.years * 12 + delta.months+1
    print(f"Processing data is done, cost {time()-s_clk:.3f} s")

    # 准备表头
    head_list = ["window", "net asset", "year compound", "time cost(s)"]
    width_list = [6, 11, 14, 12]
    header = " ".join([f"{head:>{width}}" for head,
                      width in zip(head_list, width_list)])
    print(header)

    # 2 - 模拟交易
    for window in window_range:
        # 初始化参数
        d1 = train_interval[1]-relativedelta(months=window)
        d2 = train_interval[1]+relativedelta(months=step)
        train_range = [get_last_index(date_arr, d1)+1,
                       np.where(data.trade_date == train_interval[1])[0][0]]

        capital = init_capital
        shares = init_shares
        for i in range(month_difference//step-1):
            sna, short, long = get_optimal_days(
                data, train_range,  init_capital)
            test_range = [train_range[1]+1, get_last_index(date_arr, d2)]
            capital, shares = trade(
                data, test_range, (short, long), capital, shares)
            # 更新参数
            d1 = d1+relativedelta(months=step)
            d2 = d2+relativedelta(months=step)
            train_range = [get_last_index(date_arr, d1)+1, test_range[1]]

        # 测试最后一次
        sna, short, long = get_optimal_days(
            data, train_range,  init_capital)
        test_range = [train_range[1]+1, get_last_index(date_arr, d2)]
        capital, shares = trade(
            data, test_range, (short, long), capital, shares)

        # 计算最终收益
        na = capital+shares*data.close[test_range[1]]
        year_compound = (na/init_capital)**(12/(step*(i+2)))-1

        msg = f"{window:>{width_list[0]}}"+" "+f"{na:>{width_list[1]}.3f}"+" " + \
            f"{year_compound:>{width_list[2]}.3%}" + \
            " "+f"{time()-s_clk:>{width_list[3]}.3f}"
        print(msg)


if __name__ == "__main__":
    main()
