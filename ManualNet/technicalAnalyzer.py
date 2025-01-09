import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import talib as ta
import os
from IPython.display import display
import time
from matplotlib import dates as mdates
import matplotlib.animation as animation
import threading


class TechnicalAnalyzer:

    # ボリンジャーバンドは、移動平均線を中心に、上下に標準偏差の幅を取ったバンドを描くことで、
    # 価格の変動の範囲を示す指標です。
    # バンドの幅が狭いときはボックス相場、幅が広いときはトレンド相場と判断することができます。
    # 4時間足で使用すると良いかも
    def calculate_bollinger_bands(self, df, window=20) -> pd.DataFrame:
        """
        ボリンジャーバンドを計算\n
        バンドの幅が狭いときはボックス相場、幅が広いときはトレンド相場と判断\n
        """
        df1 = df.copy()
        sma = df1['Close'].rolling(window=window).mean()
        df1['SMA'] = sma
        # 標準偏差
        std = df1['Close'].rolling(window=window).std()
        df1['upper1Sigma'] = sma + std
        df1['lower1Sigma'] = sma - std
        df1['upper2Sigma'] = sma + std * 2
        df1['lower2Sigma'] = sma - std * 2
        df1['upper3Sigma'] = sma + std * 3
        df1['lower3Sigma'] = sma - std * 3
        return df1
    
    # ストキャスティクスは、過去の価格の範囲に対する現在の価格の位置を示す指標です。
    # MACDがトレンド相場で価格の反転が認識できるが、ストキャスティクスはボックス相場で有効です。
    # %Dが0～20%の範囲にある場合は売られ過ぎと見て買い
    # %Dが80～100%の範囲にある場合は買われ過ぎと見て売り
    def calculate_fast_stochastic(self, df, n=42, m=3) -> pd.DataFrame:
        """
        ストキャスティクスを計算\n
        %Dが0～20%の範囲にある場合は売られ過ぎと見て買い\n
        %Dが80～100%の範囲にある場合は買われ過ぎと見て売り\n"""
        # highest_highの値とlowest_lowの値が同じ場合
        # つまり、最高値と最安値が同じ場合は、%Kは0となる
        df1 = df.copy()
        highest_high = df1['High'].rolling(window=n).max()
        lowest_low = df1['Low'].rolling(window=n).min()
        df1['%K'] = (df1['Close'] - lowest_low) / (highest_high - lowest_low)
        # df1の%KでNanが交じる場合がある。これは暫くhighest_highとlowest_lowが同じ値の場合に発生する
        # これを防止するため、%KがNanの値のindexの一つ前の値を代入する
        # これを行うとfillされた周りの数値が 1とか0のおかしな数字になるため、ストキャスティクスはなんか微妙
        # n=14とかの短いwindowだとこの問題が発生しやすい 40とかにするとそこまでマーケットが停止していることはほぼ無いため問題が発生しない
        # あまりに大きいとM15とかの時にwindowサイズが大きすぎてdfの頭の方にNanが発生する
        # windowが大きいことにより値の変動が少なくなるが、通常Timeframe.D1とかの大きいTimeframeで使うが今回はM1, M5, M15程度で使うので問題ないと思う
        # 一応、%KがNanの値のindexの一つ前の値を代入する処理を残しておく
        df1["%K"] = df1['%K'].ffill()
        df1['%D'] = df1['%K'].rolling(window=m).mean()
        
        return df1.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Spread'], axis=1)

    def ema(self, df: pd.DataFrame, span) -> pd.core.series.Series:
        EMA = df['Close'].ewm(span=span, adjust=False).mean()
        return EMA
    
    # MACDは、移動平均線のクロスオーバーを利用して、トレンドの転換を捉える指標です。
    # このグラフの場合MACDがsignalを超えた時に買いトレンドで
    # MACDがsignalを下回った時に売りトレンドと判断することができます。
    # ただし、MACDはボックス相場の際にダマシが多いため、他の指標と組み合わせて利用することが望ましいです。
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACDを計算\n
        MACDがsignalを超えた時に買いトレンド\n
        MACDがsignalを下回った時に売りトレンドと判断\n
        """
        df1 = df.copy()
        macd, macdsignal, macdhist = ta.MACD(df1['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df1['MACD'] = macd
        df1['Signal'] = macdsignal
        # df1のOpen High Low Close Volume Spreadを削除
        df1 = df1.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Spread'], axis=1)
        return df1
    
    # RSIを計算
    # RSIは現在の相場の相対的な強弱を示す指標でボックス相場に強い
    # RSIが30%以下の場合は売られ過ぎと見て買い、70%以上の場合は買われ過ぎと見て売り
    # ダイバージェンスとして価格が高値を更新しながらRSIが高値を更新していない動きの場合は相場の反転を示す
    # 逆に価格が安値を更新しながらRSIが安値を更新していない場合も相場の反転を示す
    # windowは14が一般的だが、9,22,42,52なども使われる
    def  calculate_rsi(self, df: pd.DataFrame, window=14) -> pd.DataFrame:
        """
        RSIを計算 RSIが30%以下の場合は売られ過ぎと見て買い、70%以上の場合は買われ過ぎと見て売り
        \ndfがindex>window以上の時間でRSIは計算可能になる、つまり(欲しい期間)+window以上のデータが必要
        return: culumにRSIを追加したDataFrame
        """
        rsi_df = ta.RSI(df['Close'], timeperiod=window)
        rsi_df = rsi_df.to_frame("RSI")
        # 100分率で表示されているためrsi_dfの各要素を100で割る
        rsi_df['RSI'] = rsi_df['RSI'] / 100
        return rsi_df