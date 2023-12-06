from dateutil.relativedelta import relativedelta

from step1 import np, time, date, DataNode, trade
from step2 import get_optimal_days


def get_last_index(date_arr, d):
    return np.where(date_arr == np.datetime64(d, "M"))[0][-1]


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
    short_range = range(1, 21)
    long_range = range(5, 121)
    step = 1  # 单位为月
    window_range = range(1, 108)  # 窗口长度范围为[1,108]月

    # 1 - 处理数据
    data = DataNode(file_path, stock_code)
    date_arr = data.trade_date.astype("datetime64[M]")  # 将日期数组转为月份数组
    delta = relativedelta(test_interval[1], test_interval[0])
    month_difference = delta.years * 12 + delta.months+1

    # 准备表头
    head_list = ["window", "net asset", "year compound", "time cost(s)"]
    width_list = [6, 11, 14, 12]
    header = " ".join([f"{head:>{width}}" for head,
                      width in zip(head_list, width_list)])
    print(header)

    # 2 - 模拟交易
    for window in window_range:
        # 初始化参数
        d1 = train_interval[1]-relativedelta(month=window)
        d2 = train_interval[1]+relativedelta(months=step)
        train_range = [get_last_index(date_arr, d1)+1,
                       np.where(data.trade_date == train_interval[1])[0][0]]

        capital = init_capital
        shares = init_shares
        for i in range(month_difference//step-1):
            sna, short, long = get_optimal_days(
                data, train_range, short_range, long_range, init_capital, cr)
            test_range = [train_range[1]+1, get_last_index(date_arr, d2)]
            capital, shares = trade(
                data, test_range, (short, long), capital, shares, cr)
            # 更新参数
            d1 = d1+relativedelta(months=step)
            d2 = d2+relativedelta(months=step)
            train_range = [get_last_index(date_arr, d1)+1, test_range[1]]

        na = capital+shares*data.close[test_range[1]]
        year_compound = (na/init_capital)**(12/(step*(i+1)))-1

        msg = f"{window:>{width_list[0]}}"+" "+f"{na:>{width_list[1]}.3f}"+" " + \
            f"{year_compound:>{width_list[2]}.3%}" + \
            " "+f"{time()-s_clk:>{width_list[3]}.3f}"
        print(msg)


if __name__ == "__main__":
    main()

"""
280s/单个长度 * 108个长度 / 3600s/h = 8.4h

window   net asset  year compound time cost(s)
     1  701734.673        -3.568%      278.798
     2  639307.351        -4.485%      557.833
     3  671207.704        -4.007%      836.009
     4  872709.084        -1.387%     1116.042
     5 1167414.384         1.600%     1395.748
     6  722042.256        -3.285%     1676.359
     7  728969.531        -3.190%     1956.786
"""
