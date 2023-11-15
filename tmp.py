import numpy as np
import pandas as pd
import talib
import multiprocessing as mp
from datetime import date
from time import time

class MyMACD_Strategy:
    def __init__(self,file_path) :
        df = pd.read_excel(file_path)
        self.trade_date = pd.to_datetime(df['Date'].values).date
        self.open_price = df['Open'].values
        self.close_price = df['Close'].values
        self.dif, self.dea, self.hist = talib.MACD(
            self.close_price, fastperiod=12, slowperiod=26, signalperiod=9)
        self.gold_cross = np.where((self.hist[:-1] < 0) & (self.hist[1:] > 0))[0] + 1
        self.dead_cross = np.where((self.hist[:-1] > 0) & (self.hist[1:] < 0))[0] + 1

    def calculate_trade_signals(self,dea1,dea2,_shares,index_range):
        ibegin,iend = index_range
        # 剪枝金死叉
        gc = self.gold_cross[(self.gold_cross>ibegin-1) & (self.gold_cross<iend)]
        dc = self.dead_cross[(self.dead_cross>ibegin-1) & (self.dead_cross<iend)]

        buy_set = gc[self.dea[gc]<dea1]+1
        sell_set = dc[self.dea[dc]>dea2]+1

        # 对首日作单独判断
        if self.dea[ibegin-1]<dea1 and self.hist[ibegin-1]>0:
            buy_set = np.append(ibegin,buy_set)
