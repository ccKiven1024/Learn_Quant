from Data import np, pd, Data
from time import time
from datetime import date
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def judge(_close: np.ndarray, eta: float = 1e-3):
    """
    返回收盘价的涨跌情况

    参数：
    - _close: 收盘价数组
    - eta: 阈值，这里默认为千分之一
    """
    array = _close[1:]/_close[:-1]-1
    # 1:上涨，-1:下跌，0:震荡
    return np.where(array > eta, 1, np.where(array < -eta, -1, 0))


def func1(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    输入实际收盘价和预测收盘价，返回查准率和查全率

    参数：
    - a1: 实际值
    - a2: 预测值
    注意：两个数组等长！

    返回：涨跌情况，实际涨跌情况，预测涨跌情况，命中次数，查准率，查全率
    """
    # 将涨跌转换为-1/0/1数组
    a1 = judge(a1)
    a2 = judge(a2)

    ca, actual = np.unique(a1, return_counts=True)  # ca为choice_array，有序递增
    pred = np.unique(a2, return_counts=True)[1]

    d = dict.fromkeys(ca, 0)  # 初始化字典
    for i in range(a1.size):  # 计算命中次数
        if a1[i] == a2[i]:
            d[a1[i]] += 1

    hit = np.zeros(shape=ca.shape)
    for i in range(hit.size):  # 保存命中次数
        hit[i] = d[ca[i]]

    precision_rate = hit/pred  # 查准率
    recall_rate = hit/actual  # 查全率
    return np.column_stack((ca, actual, pred, hit, precision_rate, recall_rate))


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
    sliding_step = [1]  # 单位为月
    md_set = np.array([3, 8, 21])  # 均线天数
    fd = 5  # 预测未来1至fd天
    ud = 5  # 用到的天数

    # 1 - 处理数据
    data = Data(file_path, stock_code, cr, md_set)
    data.create_output(fd)  # 生成输出集
    month_difference = (test_interval[1].year-test_interval[0].year) * \
        12 + test_interval[1].month-test_interval[0].month+1
    print(f"Processing data is done, cost {time()-s_clk:.3f} s")

    # 2 - 模拟预测
    # result_path = "./../result/05case2.xlsx"
    # writer = pd.ExcelWriter(result_path)
    # 设置模型
    rgl = regularizers.l2(0.01)  # 设置正则化
    model = Sequential()
    model.add(LSTM(units=20, activation='leaky_relu',
              input_shape=(ud, data.m.shape[1]), return_sequences=True, kernel_regularizer=rgl))
    # model.add(Dropout(0.3))
    model.add(LSTM(units=10, activation='leaky_relu', kernel_regularizer=rgl))
    # model.add(Dropout(0.3))
    # model.add(Dense(units=6, activation='tanh', kernel_regularizer=rgl))
    model.add(Dense(units=fd))
    model.compile(optimizer='adam', loss='mse', metrics=['mape'])

    for step in sliding_step:
        iter_num = month_difference//step  # 迭代次数
        train_list = [None]*iter_num
        test_list = [None]*iter_num
        err_arr = np.zeros(shape=(iter_num, 1))

        # 初始化参数
        train_range = [np.where(data.trade_date == d)[0][0]
                       for d in train_interval]
        if train_list[0] is None:  # 保存训练区间
            train_list[0] = data.scope2str(train_range)
        # 确定训练和测试区间
        pre = np.searchsorted(
            data.ld, train_interval[0])-1  # 训练区间起始位置在ld中的索引减1
        end = np.searchsorted(
            data.ld, train_interval[1])+step  # 测试区间结束位置在ld中的索引
        # 预测的收盘价
        tmp_index = train_range[1]  # 测试区间前一天的索引
        pred_price = [data.close[tmp_index]]  # 第一个值是测试区间前的收盘价
        for j in range(iter_num-1):
            # 划分输入和输出集
            x_train = data.get_input(train_range, fd, ud)
            y_train = data.yields1[train_range[0]:train_range[1]+1]

            test_range = [train_range[1]+1, data.ldi[end]]
            if test_list[j] is None:
                test_list[j] = data.scope2str(test_range)

            x_test = data.get_input(test_range, fd, ud)
            y_test = data.yields1[test_range[0]:test_range[1]+1]
            # 训练
            model.fit(x_train, y_train, epochs=20, batch_size=32,
                      validation_data=(x_test, y_test))
            # 在测试集上预测
            y_pred = model.predict(x_test)
            # 计算误差
            y_pred = (y_pred+1)*data.close[0]  # 将收益率还原为收盘价

            # 更新训练区间
            pre += step
            end += step
            train_range = [data.ldi[pre]+1, test_range[1]]
            if train_list[j+1] is None:  # 保存训练区间
                train_list[j+1] = data.scope2str(train_range)

        # 最后一次
        # 划分输入和输出集
        x_train = data.get_input(train_range, fd, ud)
        y_train = data.yields1[train_range[0]:train_range[1]+1]
        test_range = [train_range[1]+1, data.ldi[end]]
        if test_list[-1] is None:
            test_list[-1] = data.scope2str(test_range)
        x_test = data.get_input(test_range, fd, ud)
        y_test = data.yields1[test_range[0]:test_range[1]+1]
        # 训练
        model.fit(x_train, y_train, epochs=50, batch_size=32,
                  validation_data=(x_test, y_test))
        # 在测试集上预测
        y_pred = model.predict(x_test)
        # 计算误差
        y_pred = (y_pred+1)*data.close[0]  # 将收益率还原为收盘价
        y_true = data.close[test_range[0]:test_range[1]+1]
        err_arr[-1, i] = np.abs(y_pred/y_true-1).mean()
        pred_price.extend(y_pred)

        print(f"step={step}m, fd={fd}d is done, cost {time()-s_clk:.3f} s")


if __name__ == "__main__":
    main()
