from exercise3_2 import get_optiaml_md, trade_once
import pandas as pd


def main():
    data_path = r"StockData.xlsx"
    df = pd.read_excel(data_path)

    init_capital = 1e6
    init_shares = 0
    commission = 5e-4
    md1_range = range(1, 15 + 1)
    md2_range = range(20, 100 + 1)
    sample_date_interval = [" 2006/01/04", " 2013/12/31"]
    train_date_interval = [" 2014/01/02", " 2023/08/31"]

    sample_irange = [df[df["Date"] == date].index[0] for date in sample_date_interval]
    train_irange = [df[df["Date"] == date].index[0] for date in train_date_interval]

    # 在样本区间寻找最佳双均线参数
    c, md1, md2 = get_optiaml_md(
        df, md1_range, md2_range, init_capital, commission, sample_irange
    )
    print(
        f"md1={md1}, md2={md2}, c={c} from {sample_date_interval[0]} to {sample_date_interval[1]}"
    )
    # md1=7, md2=41, c=7576352.119150016

    # 应用在训练区间
    c = trade_once(df, md1, md2, init_capital, init_shares, commission, train_irange)
    print(f"capital = {c} from {train_date_interval[0]} to {train_date_interval[1]}")
    # capital = 1757143.190505008


if __name__ == "__main__":
    main()
