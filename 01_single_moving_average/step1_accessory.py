import pandas as pd
import matplotlib.pyplot as plt

# 从2-2.csv中读取数据
file_path = r'2-2.csv'
df = pd.read_csv(file_path)

# 打印最佳均线日期和收益
# 返回最大收益对应的索引
index = df.iloc[:, 1].idxmax()
print(
    f"Best mean day = {df.iloc[index, 0]}, the last capital = {df.iloc[index, 1]} from 2006/01/04 to 2023/08/31")

# 画图
plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('day')
plt.ylabel('revenue')
plt.show()
