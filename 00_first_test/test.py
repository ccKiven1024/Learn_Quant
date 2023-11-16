"""
第一版

import pandas as pd

file_path = r'StockData.xlsx'
df = pd.read_excel(file_path)  # 该数据与本文件处于同一目录下
m = len(df)


start_day = ' 2006/01/04'
end_day = ' 2023/08/31'
start_index = df[df['Date'] == start_day].index[0]  # 获取开始日期对应的索引
end_index = df[df['Date'] == end_day].index[0]  # 获取结束日期对应的索引

shares = 1  # 购入份数
day = 240  # 均线日期
m240 = [0.0]*m
df['m240'] = m240  # 添加均线240

r = 0.0  # 收益
flag = False  # 买入为True，否则为False
purchase_day = []  # 买入日期对应的索引
sale_day = []  # 卖出日期对应的索引

# 计算开始日期前一日的240日均线
s = 0.0  # 记录窗口求和值
for i in range(start_index-1, start_index-1-day, -1):
    s += df.at[i, 'Close']
df.at[start_index-1, 'm240'] = s/day  # 添加到m240

# 计算240日均线
for i in range(start_index, end_index+1):
    if df.at[i-1, 'Close'] > df.at[i-1, 'm240'] and flag == False:  # 前一日的收盘价大于该日的240日均线
        # 买入
        purchase_day.append(i)
        flag = True  # 置反
    elif df.at[i-1, 'Close'] < df.at[i-1, 'm240'] and flag == True:  # 前一日的收盘价低于该日的240日均线
        # 卖出
        r += shares * (df.at[i, 'Open'] -
                       df.at[purchase_day[-1], 'Open'])  # 计算收益
        sale_day.append(i)
        flag = False  # 置反
    # 计算该日均线
    s += df.at[i, 'Close'] - df.at[i-day, 'Close']
    df.at[i, 'm240'] = s/day

for i in range(10):
    print(
        f"The last {i+1:2}th purchase date is {df.at[purchase_day[-i-1], 'Date']}\
        and sale date is {df.at[sale_day[-i-1], 'Date']}")

print(f"Total profit = {r}")
"""

# 第二版
import pandas as pd


def calculate_profit(file_path, m_day, start_day, end_day):
    df = pd.read_excel(file_path)  # 该数据与本文件处于同一目录下
    df['mean_day'] = df['Close'].rolling(window=m_day).mean()  # 添加均线
    start_index = df[df['Date'] == start_day].index[0]  # 获取开始日期对应的索引
    end_index = df[df['Date'] == end_day].index[0]  # 获取结束日期对应的索引

    shares = 1  # 购入份数
    p = 0.0  # 收益
    flag = False  # 买入为True，否则为False
    purchase_day = []  # 买入日期对应的索引
    sale_day = []  # 卖出日期对应的索引

    # 执行策略
    for i in range(start_index, end_index+1):
        if df.at[i-1, 'Close'] > df.at[i-1, 'mean_day'] and not flag:  # 前一日的收盘价大于该日的240日均线
            # 买入
            purchase_day.append(i)
            flag = True  # 置反
        elif df.at[i-1, 'Close'] < df.at[i-1, 'mean_day'] and flag:  # 前一日的收盘价低于该日的240日均线
            # 卖出
            p += shares * (df.at[i, 'Open'] -
                           df.at[purchase_day[-1], 'Open'])  # 计算收益
            sale_day.append(i)
            flag = False  # 置反
    # 检查最后一个交易日是否持有股票，若持有则卖出
    if flag:
        p += shares * (df.at[end_index, 'Close'] -
                       df.at[purchase_day[-1], 'Open'])
        sale_day.append(end_index)

    # 打印最后10次交易
    for i in range(10):
        print(
            f"The last {i+1:2}th purchase date is {df.at[purchase_day[-i-1], 'Date']}\
            and sale date is {df.at[sale_day[-i-1], 'Date']}")
    return p


def main():
    file_path = r'./../data/StockData.xlsx'
    m_day = 240
    start_day = ' 2006/01/04'
    end_day = ' 2023/08/31'
    p = calculate_profit(file_path, m_day, start_day, end_day)
    print(f"Total profit = {p:.2f} from {start_day} to {end_day}")


if __name__ == "__main__":
    main()

"""
The last  1th purchase date is  2023/07/31            and sale date is  2023/08/14
The last  2th purchase date is  2023/05/09            and sale date is  2023/05/10
The last  3th purchase date is  2023/03/31            and sale date is  2023/04/24
The last  4th purchase date is  2023/03/24            and sale date is  2023/03/27
The last  5th purchase date is  2023/03/01            and sale date is  2023/03/09
The last  6th purchase date is  2023/02/21            and sale date is  2023/02/28
The last  7th purchase date is  2023/02/10            and sale date is  2023/02/20
The last  8th purchase date is  2023/01/17            and sale date is  2023/02/09
The last  9th purchase date is  2020/06/02            and sale date is  2021/07/27
The last 10th purchase date is  2020/04/30            and sale date is  2020/05/25
Total profit = 4677.30 from  2006/01/04 to  2023/08/31
"""
