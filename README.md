## 任务0的知识点

- Pandas从Excel读取数据

  ```python
  df = pd.read_excel(excel_path)
  ```

- `df`访问数据

  ```python
  df['attribute1'] # 访问某属性列
  df.at[row1,'attribute1'] # 访问某行的某属性
  df.loc[row1,'attribute1'] # 同上
  df.iat[row2,column2] # 访问坐标(row2,column2)的值
  df.iloc[row2,column2] # 同上
  ```

  

- 计算均线

  1. 手动循环计算
  2. 使用`df['attribute1'].rolling(window = mean_day)`



## 任务1的知识点

- 



## 任务2的知识点

将图片插入Excel中

```python
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt

res_path = "tmp.xlsx"
x = np.random.random_sample(100)
y = x**2

plt.plot(x, y)
plt.savefig("tmp.png")

df = pd.DataFrame({"x": x, "y": y})
df.to_excel(res_path, sheet_name="Sheet1", index=False)


# 将tmp.png插入到Sheet1的C1单元格
wb = openpyxl.load_workbook(res_path)
img = openpyxl.drawing.image.Image("tmp.png")
wb["Sheet1"].add_image(img, "C1")
wb.save(res_path)
wb.close()
```





绘制日净资产涨跌幅和标的涨跌幅（沪深300）

1. 自行设计

   ```python
   # 计算收盘价和净资产的日均涨跌幅
   t1 = (close_price - close_price[irange[0]]) / close_price[irange[0]] 
   t2 = (net_asset - init_capital) / init_capital
   
   fig, ax = plt.subplots()
   ax.plot(trade_date, t1, label="Close Price", color="red")
   ax.plot(trade_date, t2, label="Net Asset", color="yellow")
   ax.set_xlabel("Date")
   ax.set_ylabel("Yield")
   ax.set_title("Close Price vs Net Asset")
   ax.legend()
   
   # 由于日期数据太多，只绘制年份，减少写时间，提高效率，使看图更清晰
   years = YearLocator()
   years_fmt = DateFormatter("%Y")
   ax.xaxis.set_major_locator(years)
   ax.xaxis.set_major_formatter(years_fmt)
   fig.autofmt_xdate()
   
   image_path = r"close_vs_net_asset.png"
   fig.savefig(image_path)
   ```

2. 使用`mplfinace`

   参考其官方项目地址：https://github.com/matplotlib/mplfinance

   ```py
   import mplfinance as mpf
   import pandas as pd
   
   target_path = r"Learn_Quant\StockData.xlsx"
   df = pd.read_excel(target_path)
   df["Date"] = pd.to_datetime(df["Date"])
   df.set_index("Date", inplace=True)
   df.rename(columns={"Vol": "Volume"}, inplace=True)
   mpf.plot(df, type="line")
   mpf.show()
   ```

   