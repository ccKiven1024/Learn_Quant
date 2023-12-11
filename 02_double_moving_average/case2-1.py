from dateutil.relativedelta import relativedelta

from step1 import pd, np, time, date, DataNode, trade, trade1
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
    sliding_step = [1, 3, 6]  # 单位为月

    # 1 - 处理数据
    data = DataNode(file_path, stock_code)
    date_arr = data.trade_date.astype("datetime64[M]")  # 将日期数组转为月份数组
    delta = relativedelta(test_interval[1], test_interval[0])
    month_difference = delta.years * 12 + delta.months+1

    step = 1
    train_range = list(map(lambda d: np.where(
        data.trade_date == d)[0][0], train_interval))
    d1 = train_interval[0]-relativedelta(days=4)  # 2004-12-31
    # 2013-012-31 + setp month
    d2 = train_interval[1]+relativedelta(months=step)

    # 13 train: 2006-02-06 - 2015-01-30
    # 13 test: 2015-02-02 - 2015-02-27
    # short=3, long =9, net_asset = 8032873.238

    d1 = date(2006, 2, 6)
    d2 = date(2015, 1, 30)
    train_range = list(map(lambda d: np.where(
        data.trade_date == d)[0][0], [d1, d2]))
    short = 3
    long = 9
    c, s, r = trade1(data, train_range, (short, long),
                     init_capital, init_shares, cr)
    df1 = pd.DataFrame(
        r, columns=["date", "b/s", "price", "shares", "capital"])
    result_path = r"./../../108-1-13.xlsx"
    df1.to_excel(result_path, index=True)

    return
    # 2 - 模拟交易
    for step in sliding_step:
        # 初始化参数
        train_range = list(map(lambda d: np.where(
            data.trade_date == d)[0][0], train_interval))
        d1 = train_interval[0]-relativedelta(days=4)  # 2004-12-31
        # 2013-012-31 + setp month
        d2 = train_interval[1]+relativedelta(months=step)

        print(f"short long net_asset")
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
            print(f"{short:>{5}} {long:>{4}} {sna:.3f}")

        na = capital+shares*data.close[test_range[1]]
        year_compound = (na/init_capital)**(12/(step*(i+1)))-1
        print(
            f"window_month = {step}, net asset = {na:.3f}, year compound = {year_compound:.3%}, time cost = {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()

"""
window_month = 1, net asset = 1382362.392, year compound = 3.377%, time cost = 292.635 s
window_month = 3, net asset = 1438396.037, year compound = 3.901%, time cost = 387.461 s
window_month = 6, net asset = 1396595.829, year compound = 3.781%, time cost = 432.433 s

此外，发现一个很明显的问题：
ma被重复计算了多次，可以考虑将ma的计算结果保存起来，避免重复计算
"""
