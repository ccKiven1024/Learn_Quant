from Data import Data, np, pd
from time import time
from datetime import date


def main():
    s_clk = time()

    # 0 - 题目数据
    file_path = r"./../data/StockData.xlsx"
    stock_code = r"000016"
    init_capital = 1e6
    init_shares = 0
    cr = 5e-4
    sample_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    train_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    sliding_step = [1, 3, 6]  # 单位为月
    md_set = np.array([3, 8, 21])  # 均线天数
    fd = [1, 2, 5]  # 预测所需天数

    # 1 - 处理数据
    d = Data(file_path, stock_code, cr, md_set)


if __name__ == "__main__":
    main()
