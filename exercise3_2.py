from exercise3_1 import trade_once
import multiprocessing as mp
import pandas as pd
from time import time


def func(df, md1, md2, capital, shares, cr, ir):
    return (trade_once(df, md1, md2, capital, shares, cr, ir), md1, md2)


def get_optiaml_md(df, md1_range, md2_range, capital, cr, ir):
    with mp.Pool() as pool:
        res = pool.starmap(
            func,
            [
                (df, md1, md2, capital, 0, cr, ir)
                for md1 in md1_range
                for md2 in md2_range
            ],
        )
    return max(res, key=lambda x: x[0])


def main():
    s_clk = time()
    data_path = r"StockData.xlsx"
    df = pd.read_excel(data_path)

    init_capital = 1e6
    commission = 5e-4
    date_interval = [" 2006/01/04", " 2023/08/31"]
    md1_range = range(1, 15 + 1)
    md2_range = range(20, 100 + 1)

    irange = [
        df[df["Date"] == date_interval[0]].index[0],
        df[df["Date"] == date_interval[1]].index[0],
    ]

    c, md1, md2 = get_optiaml_md(
        df, md1_range, md2_range, init_capital, commission, irange
    )
    print(f"md1={md1}, md2={md2}, c={c}\nTime Cost = {time()-s_clk} s")
    # md1=7, md2=41, c=13316466.333995085
    # Time Cost = 51s


if __name__ == "__main__":
    main()
