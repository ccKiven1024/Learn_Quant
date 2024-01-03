from Data import Data, np, pd
from time import time
from datetime import date
from sklearn.linear_model import LinearRegression


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"000016"
    cr = 5e-4
    init_capital = 1e6
    init_shares = 0
    train_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    test_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    sliding_step = [1, 3, 6]  # 单位为月
    md_set = np.array([3, 8, 21])  # 均线天数
    fd_arr = np.array([1, 2, 5])  # 预测未来第几天
    ud = 5  # 用到的天数

    # 1 - 处理数据
    data = Data(file_path, stock_code, cr, md_set)
    month_difference = (test_interval[1].year-test_interval[0].year) * \
        12 + test_interval[1].month-test_interval[0].month+1
    print(f"Processing data is done, cost {time()-s_clk:.3f} s")

    # 2 - 模拟预测
    result_path = "./../result/04case1.xlsx"
    writer = pd.ExcelWriter(result_path)
    model = LinearRegression()
    for step in sliding_step:
        iter_num = month_difference//step
        train_list = [""]*iter_num
        test_list = [""]*iter_num
        err_arr = np.zeros(shape=(iter_num, fd_arr.shape[0]))
        for i in range(fd_arr.shape[0]):
            fd = fd_arr[i]  # 预测所需天数
            # 初始化参数
            train_range = [np.where(data.trade_date == d)[0][0]
                           for d in train_interval]
            if train_list[0] != "":
                train_list[0] = data.scope2str(train_range)  # 保存训练区间
            tw_pre = np.searchsorted(
                data.ld, train_interval[0])-1  # 训练区间起始位置在ld中的索引减1
            tw_end = np.searchsorted(
                data.ld, train_interval[1])+step  # 测试区间结束位置在ld中的索引

            for j in range(iter_num-1):
                # 划分输入和输出集
                x_train = data.get_input(train_range, fd, ud)
                y_train = data.close[train_range[0]:train_range[1]+1]
                # 训练
                model.fit(x_train, y_train)
                # print(model.coef_, model.intercept_)
                # 测试
                test_range = [train_range[1]+1, data.ldi[tw_end]]
                if test_list[j] != "":
                    test_list[j] = data.scope2str(test_range)  # 保存测试区间
                x_test = data.get_input(test_range, fd, ud)
                y_pred = model.predict(x_test)
                y_true = data.close[test_range[0]:test_range[1]+1]
                y_err = np.abs(y_pred/y_true-1)
                err_arr[j, i] = y_err.mean()
                # 更新参数
                tw_pre += step
                tw_end += step
                train_range = [data.ldi[tw_pre]+1, test_range[1]]
                if train_list[j+1] != "":
                    train_list[j+1] = data.scope2str(train_range)  # 保存训练区间

            # 最后一次
            x_train = data.get_input(train_range, fd, ud)
            y_train = data.close[train_range[0]:train_range[1]+1]
            model.fit(x_train, y_train)
            test_range = [train_range[1]+1, data.ldi[tw_end]]
            if test_list[-1] != "":
                test_list[-1] = data.scope2str(test_range)  # 保存测试区间
            x_test = data.get_input(test_range, fd, ud)
            y_pred = model.predict(x_test)
            y_true = data.close[test_range[0]:test_range[1]+1]
            y_err = np.abs(y_pred/y_true-1)
            err_arr[-1, i] = y_err.mean()

            print(f"step={step}m, fd={fd}d is done, cost {time()-s_clk:.3f} s")

        # 保存结果
        sheet_name = f"step={step}m"
        values = np.column_stack((train_list, test_list, err_arr))
        cols = ["train", "test"] + [f"err-{d}d" for d in fd_arr]
        tmp_df = pd.DataFrame(values, columns=cols)
        tmp_df.loc[iter_num, tmp_df.columns[0]
                   ] = f"time cost = {time()-s_clk:.3f} s"
        tmp_df.to_excel(writer, sheet_name=sheet_name, index=True)
    writer.close()


if __name__ == "__main__":
    main()
