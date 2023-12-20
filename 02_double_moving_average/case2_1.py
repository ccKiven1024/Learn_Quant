import sys
from dateutil.relativedelta import relativedelta

from step1 import pd, np, time, date, DataNode, trade
from step2 import get_optimal_days


def progress_bar(iterable, total,  time_cost, prefix='', length=50, fill='█'):
    ratio = f"{iterable}/{total}"
    filled_length = int(length * iterable // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(
        f'\r{prefix} |{bar}| {ratio} Complete, cost {time_cost:.3f} s')
    sys.stdout.flush()
    if iterable == total:
        print()


def get_last_index(date_arr, d):
    """
    寻找某月的最后一天的索引
    """
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
    short_range = np.array(range(1, 21))
    long_range = np.array(range(5, 121))
    sliding_step = [1, 3, 6]  # 单位为月

    # 1 - 处理数据
    data = DataNode(file_path, stock_code, cr, short_range, long_range)
    date_arr = data.trade_date.astype("datetime64[M]")  # 将日期数组转为月份数组
    delta = relativedelta(test_interval[1], test_interval[0])
    month_difference = delta.years * 12 + delta.months+1
    print(f"Processing data is done, cost {time()-s_clk:.3f} s")

    result_path = "./../result/02case2_1.xlsx"
    writer = pd.ExcelWriter(result_path)
    # 2 - 模拟交易
    for step in sliding_step:
        # 初始化参数
        train_range = list(map(lambda d: np.where(
            data.trade_date == d)[0][0], train_interval))
        d1 = train_interval[0]-relativedelta(days=4)  # 2004-12-31
        # 2013-012-31 + setp month
        d2 = train_interval[1]+relativedelta(months=step)

        capital = init_capital
        shares = init_shares
        records = []
        iter_num = month_difference//step
        for i in range(month_difference//step-1):
            # 训练
            sna, short, long = get_optimal_days(
                data, train_range, init_capital)
            records.append((short, long, sna))  # 保存记录
            # 测试
            test_range = [train_range[1]+1, get_last_index(date_arr, d2)]
            capital, shares = trade(
                data, test_range, (short, long), capital, shares)
            # 更新参数
            d1 = d1+relativedelta(months=step)
            d2 = d2+relativedelta(months=step)
            train_range = [get_last_index(date_arr, d1)+1, test_range[1]]
            # 输出进度条
            progress_bar(i+1, iter_num, time()-s_clk,
                         prefix=f"step = {step} months:")

        # 最后一次测试
        sna, short, long = get_optimal_days(data, train_range, init_capital)
        records.append((short, long, sna))
        test_range = [train_range[1]+1, get_last_index(date_arr, d2)]
        capital, shares = trade(
            data, test_range, (short, long), capital, shares)
        progress_bar(iter_num, iter_num, time()-s_clk,
                     prefix=f"step = {step} months:")

        # 计算最终收益
        na = capital+shares*data.close[test_range[1]]
        year_compound = (na/init_capital)**(12/(step*iter_num))-1
        records.append(("NA", "Year Compound", "Time Cost(s)"))
        records.append((na, year_compound, time()-s_clk))

        # 保存结果
        sheet_name = f"step{step}"
        tmp_df = pd.DataFrame(
            records, columns=["short", "long", "NA in Sample"])
        tmp_df.to_excel(writer, sheet_name=sheet_name, index=True)

    writer.close()


if __name__ == "__main__":
    main()
