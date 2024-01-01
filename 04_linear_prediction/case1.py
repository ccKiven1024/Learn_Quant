import numpy as np
import pandas as pd
from time import time
from datetime import date


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
    sample_interval = [date(2005, 1, 4), date(2013, 12, 31)]
    train_interval = [date(2014, 1, 2), date(2023, 10, 31)]
    sliding_step = [1, 3, 6]  # 单位为月

    # 1 - 处理数据
    df = pd.read_excel(file_path, sheet_name=stock_code)
    md_set = [3, 8, 21]


if __name__ == "__main__":
    main()
