from time import time
import pandas as pd
from exercise3_1 import trade_dm, trade_once
from exercise3_2 import get_optiaml_md


def main():
    s_clk = time()

    data_path = r"StockData.xlsx"
    df = pd.read_excel(data_path)

    init_c = 1e6
    cr = 5e-4
    md1_range = range(1, 15 + 1)
    md2_range = range(20, 100 + 1)
    init_sample_interval = [" 2006/01/04", " 2013/12/31"]
    train_interval = [" 2014/01/02", " 2023/08/31"]
    sliding_years = 1

    start_year, end_year = [split_year(date) for date in train_interval]
    window_years = start_year - split_year(init_sample_interval[0])
    init_sample_ir = [df[df["Date"] == date].index[0] for date in init_sample_interval]

    capital = init_c
    shares = 0
    net_asset = 0.0
    sample_ir = init_sample_ir
    for y in range(start_year, end_year):
        tmp_c, md1, md2 = get_optiaml_md(
            df, md1_range, md2_range, init_c, cr, sample_ir
        )
        ir = [sample_ir[1] + 1, get_last_date_index(df, y)]
        capital, shares, net_asset = trade_dm(df, md1, md2, capital, shares, cr, ir)
        print(
            f"sample year: {y-window_years}-{y-1}, md1 = {md1}, md2 = {md2}, net asset = {tmp_c}\ntrain year: {y}, capital = {capital}, shares = {shares}, net_asset = {net_asset}\nTime Cost = {time()-s_clk}\n"
        )
        sample_ir = [get_first_date_index(df, y - window_years + 1), ir[1]]

    # 处理最后一年
    y = end_year
    tmp_c, md1, md2 = get_optiaml_md(df, md1_range, md2_range, init_c, cr, sample_ir)
    ir = [sample_ir[1] + 1, df.index[-1]]
    capital = trade_once(df, md1, md2, capital, shares, cr, ir)
    print(
        f"sample year: {y-window_years}-{y-1}, md1 = {md1}, md2 = {md2}, net asset = {tmp_c}\ntrain time: {get_first_date_index(y)}-{train_interval[1].strip()}, capital = {capital}\nTime Cost = {time()-s_clk}\n"
    )


if __name__ == "__main__":
    main()

"""
sample year: 2006-2013, md1 = 7, md2 = 41, net asset = 7576352.119150016
train year: 2014, capital = 69.86913500027731, shares = 389, net_asset = 1374683.0591350002
Time Cost = 25.165719032287598

sample year: 2007-2014, md1 = 6, md2 = 79, net asset = 4884708.87043501
train year: 2015, capital = 2816.3282550000586, shares = 437, net_asset = 1633263.328255
Time Cost = 51.055410623550415

sample year: 2008-2015, md1 = 6, md2 = 39, net asset = 2569007.264495003
train year: 2016, capital = 1502158.8946700008, shares = 0, net_asset = 1502158.8946700008
Time Cost = 76.88014268875122

sample year: 2009-2016, md1 = 1, md2 = 42, net asset = 3110051.2083350127
train year: 2017, capital = 1652609.275580002, shares = 0, net_asset = 1652609.275580002
Time Cost = 102.86518883705139

sample year: 2010-2017, md1 = 6, md2 = 30, net asset = 2132626.8524750043
train year: 2018, capital = 1233479.9423200034, shares = 0, net_asset = 1233479.9423200034
Time Cost = 128.78573513031006

sample year: 2011-2018, md1 = 1, md2 = 42, net asset = 1758644.5168400102
train year: 2019, capital = 3773.3455350045115, shares = 341, net_asset = 1400707.1255350045
Time Cost = 154.58201050758362

sample year: 2012-2019, md1 = 10, md2 = 25, net asset = 2121061.5929500074
train year: 2020, capital = 2820.6435550060123, shares = 287, net_asset = 1498460.873555006
Time Cost = 180.47012042999268

sample year: 2013-2020, md1 = 8, md2 = 55, net asset = 2400698.9854200035
train year: 2021, capital = 344.9017000065651, shares = 273, net_asset = 1349065.9117000066
Time Cost = 207.00906133651733

sample year: 2014-2021, md1 = 14, md2 = 46, net asset = 2418744.071940006
train year: 2022, capital = 3081.9175350065343, shares = 350, net_asset = 1358152.4175350065
Time Cost = 233.2613549232483

sample year: 2015-2022, md1 = 13, md2 = 41, net asset = 1571538.4087550035
train time: 2023/01/03-2023/08/31, capital = 1292834.658345007
Time Cost = 259.056245803833
"""
