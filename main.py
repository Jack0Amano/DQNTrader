import datetime
import sys
import pandas as pd
import numpy as np
import enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from model import LSTMAgent
import torch
import talib as ta
import CommonAnalyzer as ca

def interpolate(df: pd.DataFrame, start_index, end_index, crop=True) -> pd.DataFrame:

    """
    DataframeにNaneとして入っている欠損値を補完
    df: DataFrame
    start_index: 補完開始インデックス start_indexのデータがない場合は補完しない
    end_index: 補完終了インデックス
    """
    
    # 欠損値を補完
    df.loc[(df.index >= start_index) & (df.index <= end_index), df.columns] = df.loc[(df.index >= start_index) & (df.index <= end_index), df.columns].interpolate()
    # start_index~end_indexの範囲外のデータを削除
    if crop:
        df = df.loc[(df.index >= start_index) & (df.index <= end_index)]
    return df

def interpolate_df(df: pd.DataFrame):
    index = 0
    while index + 1 < len(df.index):
        new_minute_data = df.iloc[index]
        if index != df.index[-1]:
            current_time = df.index[index]
            next_time = df.index[index+1]
            time_delta = next_time - current_time
            # dataに抜けが存在する場合で期間が一日以下の場合これを補完する
            if time_delta > pd.Timedelta(minutes=1) and time_delta < pd.Timedelta(days=1):
                #print("Data is missing at", current_time, "to", next_time, "Data will be interpolated.", end="", flush=True)
                # 抜けている分のデータをnoneで埋めて作成
                none_df = pd.DataFrame(index=pd.date_range(current_time+pd.Timedelta(minutes=1), next_time-pd.Timedelta(minutes=1), freq='1min'))
                none_df['Current'] = none_df.index
                none_df['Current'] = pd.to_datetime(none_df['Current'])
                none_df = none_df.set_index('Current')
                none_df = none_df.interpolate()
                df = pd.concat([df, none_df], ignore_index=False)
                df = df.sort_index()
                df = interpolate(df, current_time, next_time, False)
        index += 1
    
    df = df.reset_index(drop=False)
    df = df.set_index("Current")
    return df

class OrderType(enum.Enum):
    NOTHING = 0
    BUY = 1
    SELL = 2
    TAKE_PROFIT = 3

class Order:
    # 注文の時間
    order_time: pd.Timestamp
    # 注文の価格 実際の値
    raw_value: float
    # 注文の種類
    order_type: OrderType
    # 決済の時間
    checkout_time: pd.Timestamp = None
    # 講座の1ロットあたりの通貨量
    ONE_LOT_CURRENCY = 100000
    # レバレッジ
    leverage = 15
    # 注文のロット
    order_lot = 0.01
    # 今回の注文での通貨量
    order_currency = 0
    # 決済時のスプレッドを考慮するか 実際はスプレッドがあるためTrueにする
    spread_active = True
    # 1pipあたりの基本通貨における価格
    pip = 0.01
    # Checkout時に取得した損益
    profit_pips = 0
    # calculate_profitで取得した損益のの履歴
    calculate_profits = {}
    # spreadの係数 (たぶん通貨ペアによって異なる?)
    spread_coefficient = 0.001

    def __init__(self, time: pd.Timestamp, value: float, order_type: OrderType, order_lot=0.01):
        """
        注文を作成する"""
        self.order_time = time
        self.raw_value = value
        self.order_type = order_type
        self.order_lot = order_lot
        self.order_currency = self.ONE_LOT_CURRENCY * self.order_lot
    
    def __str__(self):
        return f"Time: {self.order_time}, Value: {self.raw_value}, OrderType: {self.order_type}"

    def is_in_time(self, time: pd.Timestamp) -> bool:
        """
        注文がtimeの時間内にあるかを判定する
        """
        after_order_time = self.order_time <= time
        before_checkout_time = self.checkout_time is None or self.checkout_time >= time
        return after_order_time and before_checkout_time
    
    # 現在の価格と比較しての損益を計算する
    def calculate_profit(self, time, price, spread) -> float:
        """
        現在の価格と比較しての損益を計算する   
        current_price: 現在の価格   
        spread: スプレッド   
        return: 損益（pip)
        """
        if type(time) is pd.Timestamp:
            time = time.to_numpy()

        output = self.calculate_profits.get(time, None)
        if output is not None:
            return output
        # spreadがself.spread_confficientで割られた数であるため戻す
        # 実際の環境では要変更
        spread_real = spread * self.spread_coefficient if self.spread_active else 0
        if self.order_type == OrderType.BUY:
            output = (price - self.raw_value - spread_real) / self.pip
        elif self.order_type == OrderType.SELL:
            output = (self.raw_value - price - spread_real) / self.pip
        else:
            output = 0
        self.calculate_profits[time] = output
        return output
        
    def checkout(self, time: pd.Timestamp, current_price, spread) -> float:
        """
        ポジションの決済   
        time: 決済時間   
        return: 損益(pip) 
        """
        self.checkout_time = time
        self.profit_pips = self.calculate_profit(time.to_numpy(), current_price, spread)
        return self.profit_pips
    
    # 必要証拠金を計算する
    def calculate_margin(self, current_price) -> float:
        """
        必要証拠金を計算する   
        current_price: 現在の価格   
        spread: スプレッド   
        return: 必要証拠金
        """
        order_value = self.order_currency * current_price
        return order_value / self.leverage

class Analyzer:
    """
    データを解析するクラス
    """
    length = 60

    def __init__(self):
        pass

    # ボリンジャーバンドは、移動平均線を中心に、上下に標準偏差の幅を取ったバンドを描くことで、
    # 価格の変動の範囲を示す指標です。
    # バンドの幅が狭いときはボックス相場、幅が広いときはトレンド相場と判断することができます。
    def bollinger_band(self, df, window=20, sigma=1) -> np.ndarray:
        """
        ボリンジャーバンドを計算するメソッド   
        """
        sma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        # 最初の方にNanが発生するため削除
        sma = sma.dropna()
        std = std.dropna()

        upperSigma = sma + std * sigma
        lowerSigma = sma - std * sigma
        gap = upperSigma - lowerSigma
        return gap.values[-self.length:]

    # RSIを計算
    # RSIは現在の相場の相対的な強弱を示す指標でボックス相場に強い
    # RSIが30%以下の場合は売られ過ぎと見て買い、70%以上の場合は買われ過ぎと見て売り
    # ダイバージェンスとして価格が高値を更新しながらRSIが高値を更新していない動きの場合は相場の反転を示す
    # 逆に価格が安値を更新しながらRSIが安値を更新していない場合も相場の反転を示す
    # windowは14が一般的だが、9,22,42,52なども使われる
    def rsi(self, df, window=14, downtrend_threshold=0.7, uptrend_threshold=0.3) -> np.ndarray:
        """
        RSIを計算するメソッド   
        return: [(0を上回ったら売りの可能性), (0を上回ったら買いの可能性)]   
        """
        rsi: np.array = ta.RSI(df['Close'].values, timeperiod=window)
        rsi = rsi[~np.isnan(rsi)] / 100
        
        # 70%のラインを越えたら売りを見込む
        downtrends = rsi - downtrend_threshold
        downtrends = downtrends[-self.length:]
        # 30%のラインを下回ったら買いを見込む
        uptrends = (1 - rsi) - (1 - uptrend_threshold)
        uptrends = uptrends[-self.length:]
        output = np.array([uptrends, downtrends])
        return output
        
    # MACDは、移動平均線のクロスオーバーを利用して、トレンドの転換を捉える指標です。
    # このグラフの場合MACDがsignalを超えた時に買いトレンドで
    # MACDがsignalを下回った時に売りトレンドと判断することができます。
    # ただし、MACDはボックス相場の際にダマシが多いため、他の指標と組み合わせて利用することが望ましいです。
    def macd(self, df) -> np.ndarray:
        """
        MACDを計算するメソッド   
        return: MACDの交差点
        """
        macd, macdsignal, macdhist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[~np.isnan(macd)]
        macdsignal = macdsignal[~np.isnan(macdsignal)]
        # 
        macd_cross_over = macd - macdsignal
        macd_cross_over = macd_cross_over[-self.length:]
        return macd_cross_over

    
    # ストキャスティクスは、過去の価格の範囲に対する現在の価格の位置を示す指標です。
    # MACDがトレンド相場で価格の反転が認識できるが、ストキャスティクスはボックス相場で有効です。
    # %Dが0～20%の範囲にある場合は売られ過ぎと見て買い
    # %Dが80～100%の範囲にある場合は買われ過ぎと見て売り
    def stochastic_oscillator(self, df) -> np.ndarray:
        """
        ストキャスティクスオシレータを計算するメソッド   
        return: ゴールデンクロス&デッドクロス、overbought, oversold、トレンドフォロー
        """
        k, d = ta.STOCH(df['High'].values, df['Low'].values, df['Close'].values, fastk_period=5, slowk_period=3, slowd_period=3)
        k = k[~np.isnan(k)]
        d = d[~np.isnan(d)]
        # ゴールデンクロス、デッドクロスを計算
        golden_dead_cross = k - d
        golden_dead_cross = golden_dead_cross[-self.length:]
        # ダイバージェンス
        # divergence = k 
        # divergence = divergence[-self.length:]
        overbought = k - 0.8
        oversold = (1 - k) - (1 - 0.2)
        # 中立域でのトレンド方向 トレンドフォロー
        # トレンドフォローは 0.2, 0.8を越えるとトレンドフォローではなくoverbought, oversoldとなる
        # 0.5を基準に上昇トレンドか下降トレンドかを判断する
        trend = k - 0.5
        trend = trend[-self.length:]

        output = np.array([golden_dead_cross, overbought, oversold, trend])
        return output

    
    def adx(self, df, window=14, trend_threshold=25) -> np.ndarray:
        """
        ADXを計算するメソッド   
        ADXが1を上回るとトレンド相場、diff_diが+になると上昇トレンド、-になると下降トレンド   
        return: トレンドの方向
        """
        adx = ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        minus_di = ta.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        plus_di = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        adx = adx[~np.isnan(adx)]
        minus_di = minus_di[~np.isnan(minus_di)]
        plus_di = plus_di[~np.isnan(plus_di)]
        adx = adx / trend_threshold
        diff_di = plus_di - minus_di

        # トレンドの方向を示す
        trend_direction = diff_di / adx.abs()
        trend = adx * trend_direction
        trend = trend[-self.length:]

        return trend
    
class Trader:
    orders_history = []
    current_order: Order = None
    first_time = None
    # 利確圧力を加えていくまでの時間
    limit_time_after_checkout = pd.Timedelta(minutes=120)

    state_channels = {
        "Close": False,
        "Profit": False,
        "OrderType": False,
        "Volume": True,
        "BollingerBand": True,
        "RSI": True,
        "MACD": True,
    }

    def __init__(self):
        self.analyzer = Analyzer()
        self.close_intervals = []
        self.profit_invtervals = []
        self.order_type_intervals = []
        self.volume_intervals = []
        self.bollinger_band_intervals = []
        self.rsi_intervals = []
        self.macd_intervals = []

    def show_intervals(self):
        print("Close:", np.min(self.close_intervals), np.max(self.close_intervals), np.mean(self.close_intervals))
        print("Profit:", np.min(self.profit_invtervals), np.max(self.profit_invtervals), np.mean(self.profit_invtervals))
        print("OrderType:", np.min(self.order_type_intervals), np.max(self.order_type_intervals), np.mean(self.order_type_intervals))
        print("Volume:", np.min(self.volume_intervals), np.max(self.volume_intervals), np.mean(self.volume_intervals))
        print("BollingerBand:", np.min(self.bollinger_band_intervals), np.max(self.bollinger_band_intervals), np.mean(self.bollinger_band_intervals))
        print("RSI:", np.min(self.rsi_intervals), np.max(self.rsi_intervals), np.mean(self.rsi_intervals))
        print("MACD:", np.min(self.macd_intervals), np.max(self.macd_intervals), np.mean(self.macd_intervals))


    def take_action(self, value: float, spread: float, time: pd.Timestamp, order_type: OrderType) -> tuple:
        """
        現在の価格と行う場合は注文を受け取り、注文を行う   
        ここの報酬を変えることによって行動の評価を変えることができる   
        つまりここはルールと報酬を決める部分である   
        value: 現在の価格   
        spread: スプレッド   
        order_time: 現在の時間   
        order_type: 注文の種類   
        returns: (利益)
        """

        if self.first_time is None:
            self.first_time = time
        
        if self.current_order is None:
            if order_type == OrderType.BUY:
                self.current_order = Order(time, value, OrderType.BUY)
                # print("Buy order is taken at", time, "Value is", value)
                return 0
            elif order_type == OrderType.SELL:
                self.current_order = Order(time, value, OrderType.SELL)
                # print("Sell order is taken at", time, "Value is", value)
                return 0
            elif order_type == OrderType.TAKE_PROFIT:
                #print("There is no order to checkout")
                return 0
            elif order_type == OrderType.NOTHING:
                # 最後の注文から一定時間経過している場合は報酬を減らしていく
                if len(self.orders_history) > 0:
                    last_order: Order = self.orders_history[-1]
                    order_time_delta: pd.Timedelta = time - last_order.checkout_time
                    if order_time_delta > self.limit_time_after_checkout:
                        return 0
                else:
                    # first_timeから一定時間経過している場合は報酬を減らしていく
                    # 何も買わずに終えるのを防ぐため
                    order_time_delta: pd.Timedelta = time - self.first_time
                    if order_time_delta > self.limit_time_after_checkout:
                        return 0
        else:
            if order_type == OrderType.TAKE_PROFIT:
                profit = self.current_order.checkout(time, value, spread)
                # print("Checkout order is taken at", time, "Value is", value, "Profit is", profit)
                self.orders_history.append(self.current_order)
                pip = self.current_order.pip
                self.current_order = None
                return profit
            elif order_type == OrderType.BUY or order_type == OrderType.SELL:
                return 0
            elif order_type == OrderType.NOTHING:
                order_time_delta: pd.Timedelta = time - self.current_order.order_time
                if order_time_delta > self.limit_time_after_checkout:
                    profit = self.current_order.calculate_profit(time, value, spread)
                    if profit < 0:
                        return 0
                    else:
                        return 0
                else:
                    pass
                    
        return 0


    def get_state(self, sequence_data: pd.DataFrame, sequence_length=1440) -> np.array:
        """
        現在の値動きの状態とorderの状態を取得する   
        returnのprofit軸とorder_type軸はcurrentorderが無い時間は0で埋める   
        sequence_data: -sequence_length分から現在のデータフレーム   
        sequence_length: この長さにトリミングして返す      
        returns: [[close(-sequence_length), ... close(current)],    
                [profit(-sequence_length), ... profit(current)],   
                [order_type(-sequence_length), ... order_type(current)]]  
        """
        output = []

        start_time = datetime.datetime.now()

        times = np.array(sequence_data.index[-sequence_length:])
        closes = sequence_data['Close'].values[-sequence_length:]
        spreads = np.array(sequence_data['Spread'].values[-sequence_length:])
        profits = np.zeros(sequence_length)
        order_type = np.zeros(sequence_length)

        # output.append(times)

        if self.state_channels["Close"]:
            closes_normaled = (closes - closes.min()) / (closes.max() - closes.min()) 
            closes_normaled = np.array(closes_normaled)
            output.append(closes_normaled)

        self.close_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        
        if self.state_channels["Profit"]:
            # 範囲内のorders_historyとcurrent_orderのcalcurate_profitsをcalcuralte_profitを呼び出して更新
            marged_profits = {}
            if self.current_order is not None:
                [self.current_order.calculate_profit(t, c, s) for c, s, t in zip(closes, spreads, times)]
                marged_profits = self.current_order.calculate_profits.copy()
            
            for order in self.orders_history:
                if times[0] > order.checkout_time:
                    continue
                times_in_order = times[(times >= order.order_time) & (times <= order.checkout_time)]
                [order.calculate_profit(t, c, s) for c, s, t in zip(closes, spreads, times_in_order)]
                marged_profits.update(order.calculate_profits)

            update_series = pd.Series(marged_profits)
            sequence_data = sequence_data.assign(Profit=update_series.reindex(sequence_data.index, fill_value=0))
            profits = np.array(sequence_data['Profit'].values[-sequence_length:])

            output.append(profits)

        self.profit_invtervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        if self.state_channels["OrderType"]:
            # 注文の種類は 0,0,0,... ,(order), order_type(0), order_type(1)となる
            if self.current_order is not None:
                order_time = self.current_order.order_time.to_datetime64()
                order_type = [self.current_order.order_type.value if t >= order_time else 0 for t in times]
            
            # 過去の注文がある場合もそれを埋める
            for order in self.orders_history:
                order_time = order.time.to_datetime64()
                checkout_time = order.checkout_time.to_datetime64()
                old_order_type = [order.order_type.value if t >= order_time and t <= checkout_time else ot for t, ot in zip(times, order_type)]
                order_type = [max(ot, oot) for ot, oot in zip(order_type, old_order_type)]

            output.append(order_type)
        
        self.order_type_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        if self.state_channels["Volume"]:
            volumes = np.array(sequence_data['Volume'].values[-sequence_length:])
            output.append(volumes)

        self.volume_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        # ボリンジャーバンドの計算
        if self.state_channels["BollingerBand"]:
            bollinger_band = self.tec_analy.calculate_bollinger_bands(sequence_data)
            gap2 = (bollinger_band["upper2Sigma"] - bollinger_band["lower2Sigma"]).dropna(how='all').to_frame("gap2")
            gap2 = np.array(gap2.iloc[-sequence_length:]["gap2"].values)
            output.append(gap2)

        self.bollinger_band_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()
        
        # RSIの計算
        if self.state_channels["RSI"]:
            rsi = self.tec_analy.calculate_rsi(sequence_data)
            rsi = np.array(rsi.iloc[-sequence_length:]["RSI"].values)
            output.append(rsi)

        self.rsi_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()
        
        # MACDの計算
        if self.state_channels["MACD"]:
            macd = self.tec_analy.calculate_macd(sequence_data)
            macd = np.array(macd.iloc[-sequence_length:]["MACD"].values)
            output.append(macd)
        
        self.macd_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        return np.array(output)
    
    def get_state_new(self, sequence_data: pd.DataFrame) -> np.array:

        # ボリンジャーバンド
        bollinger_band = self.analyzer.bollinger_band(sequence_data)
        # RSI
        rsi = self.analyzer.rsi(sequence_data)
        # MACD
        macd = self.analyzer.macd(sequence_data)
        # ストキャスティクス
        stochastic_oscillator = self.analyzer.stochastic_oscillator(sequence_data)
        stochastic_trend = stochastic_oscillator[3]
        stochastic_oscillator = stochastic_oscillator[:3]
        # ADX
        adx = self.analyzer.adx(sequence_data)

        # arrayの重ね方は 最初の方にボリンジャーバンド, 各解析のトレンド系, トレンド反転系
        return np.array([bollinger_band, rsi, macd, stochastic_oscillator, stochastic_trend, adx])


        
    def get_state_channel_size(self):
        """
        get_stateが返すのチャンネル数を取得する  
        """
        return np.array(list(self.state_channels.values())).sum()

    def __calculate_feature_profits(self, action, target_price, future_prices, spreads) -> list:
        """
        target_priceでの取引を行った場合の将来の利益を計算   
        action: str, "Buy" or "Sell"
        target_price: float, 未来の価格を予測するための基準価格 (ここでポジションを設定する)
        future_prices: list[float], 未来の価格のリスト
        """
        if action == OrderType.BUY:
            # Buy の場合の将来の利益
            future_profits = []
            for (fp, spread) in zip(future_prices, spreads):
                future_profit = (fp - target_price) - spread
                future_profits.append(future_profit)
            return future_profits
        elif action == OrderType.SELL:
            # Sell の場合の将来の利益
            future_profits = []
            for (fp, spread) in zip(future_prices, spreads):
                future_profit = (target_price - fp) - spread
                future_profits.append(future_profit)
            return future_profits
    
    def calculate_reward(self, action, current_price, current_spread, future_prices, future_spread) -> list:
        """アクションに応じた報酬を計算"""
        rewards = []
        if action == OrderType.BUY:
            if not self.current_order:
                feature_profits = self.__calculate_feature_profits(OrderType.BUY, current_price, future_prices, future_spread)
                if max(feature_profits) > 0:
                    # 将来の価格変動から利益が出せる場合
                    rewards = feature_profits
                else:
                    rewards.append(-1)
            else:
                rewards.append(0)
        elif action == OrderType.SELL:
            if not self.current_order:
                feature_profits = self.__calculate_feature_profits(OrderType.SELL, current_price, future_prices, future_spread)
                if max(feature_profits) > 0:
                    rewards = feature_profits
                else:
                    rewards.append(-1)
            else:
                rewards.append(0)
        elif action == OrderType.TAKE_PROFIT:
            if self.current_order:
                if self.current_order.order_type == OrderType.BUY:
                    reward = (current_price - self.current_order.raw_value) - current_spread
                elif self.current_order.order_type == OrderType.SELL:
                    reward = (self.current_order.raw_value - current_price) - current_spread
                rewards.append(reward)
            else:
                rewards.append(0)  # ポジションがない場合の報酬は 0
        elif action == OrderType.NOTHING:
            if self.current_order is None:
                # ポジションがない場合、将来の価格変動から利益が出せるかチェック
                buy_profitable = any((fp - current_price) - current_spread > 0 for fp in future_prices)
                sell_profitable = any((current_price - fp) - current_spread > 0 for fp in future_prices)
                if buy_profitable or sell_profitable:
                    # 利益を出せる可能性があるため、機会損失としてマイナス報酬
                    # 最大利益の期待値をマイナスとして設定
                    possible_rewards = []
                    for (fp, spread) in zip(future_prices, future_spread):
                        possible_buy = (fp - current_price) - spread
                        possible_sell = (current_price - fp) - spread
                        possible_rewards.append(max(possible_buy, possible_sell, 0))
                    reward = -np.mean([r for r in possible_rewards if r > 0])  # 平均機会損失
                    rewards.append(reward)
                else:
                    # 利益を出せないため、報酬は 0
                    rewards.append(0)
            else:
                # ポジションがある場合
                # TakeProfit が必要かどうかを判断
                if self.current_order.order_type == OrderType.BUY:
                    # 現在の価格とポジション価格から利益が出ているか
                    current_profit = current_price - self.current_order.raw_value
                elif self.current_order.order_type == OrderType.SELL:
                    current_profit = self.current_order.raw_value - current_price
                else:
                    current_profit = 0

                # 将来の期待利益を計算
                buy_profit = max(self.__calculate_feature_profits(OrderType.BUY, self.current_order.raw_value, future_prices, future_spread))
                sell_profit = max(self.__calculate_feature_profits(OrderType.SELL, self.current_order.raw_value, future_prices, future_spread))
                max_future_profit = max([buy_profit, sell_profit], default=0)
                
                if current_profit > max_future_profit:
                    # 現在が最大の利益の場合、現在利確すべきであるためNothingの報酬は -current_profit
                    rewards.append(-current_profit)
                else:
                    # 現在の利益より期待される未来の利益が大きい場合、ポジションを保持するためにNothingの報酬はmax_future_profit
                    rewards.append(max_future_profit)
        return rewards

    def calculate_q_values(self, current_price, current_spread, future_sequence) -> dict:
        """
        各アクションのQ値を計算   
        current_price: 現在の価格   
        future_sequence: 未来の価格のデータフレーム 
        future_sequenceの長さがLabelの予測能となるため、modelの予測間隔を調整する場合これを変更する必要がある重要なパラメーター   
        """
        future_prices = future_sequence['Close'].values
        future_spread = future_sequence['Spread'].values

        actions = [OrderType.NOTHING, OrderType.BUY, OrderType.SELL, OrderType.TAKE_PROFIT]
        q_values = {}
        for action in actions:
            if action == OrderType.TAKE_PROFIT and not self.current_order:
                # ポジションがない場合、TakeProfitは無効
                q_values[action] = -10  # 無効な選択肢として極小値を設定
            elif (action == OrderType.BUY or action == OrderType.SELL) and self.current_order:
                # ポジションがある場合、Buy と Sell は無効
                q_values[action] = -10  # 無効な選択肢として極小値を設定
            else:
                rewards = self.calculate_reward(action, current_price, current_spread, future_prices, future_spread)
                q_values[action] = np.mean(rewards)  # 平均で期待値を計算
        return q_values



class TraderDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot([], [], label="Close")  
        self.ax1 = self.ax.twinx()
        self.tranings = []

    def update_graph(self, state: np.array, order: Order, profits, traning:pd.Timestamp=None, show_length=120):
        """
        状態をグラフに描画する
        """
        self.fig.delaxes(self.ax)
        self.fig.delaxes(self.ax1)

        times = state[0][-show_length:]
        # timesをdatetimeに変換
        times = [pd.Timestamp(t) for t in times]
        closes = state[1][-show_length:]

        # times外のtraingを削除
        self.tranings = [t for t in self.tranings if t in times]

        self.ax = self.fig.add_subplot(211)
        self.ax.plot(times, closes, label="Close")

        if order is not None:
            if order.order_type == OrderType.BUY:
                self.ax.axhline(order.raw_value, color='r')
            elif order.order_type == OrderType.SELL:
                self.ax.axhline(order.raw_value, color='g')

        if traning is not None:
            self.tranings.append(traning)
        
        for traning in self.tranings:
            self.ax.axvline(traning, color='b')

        # profitをax1に描画
        self.ax1 = self.fig.add_subplot(212)
        profit_times = profits[:, 0]
        profit_values = profits[:, 1]
        self.ax1.plot(profit_times, profit_values, label="Profit")
        
        plt.pause(0.0001)

    def save_graph(self, path):
        plt.savefig(path)


def remove_cmd_line():
    """
    コマンドラインの表示を消す
    """
    sys.stdout.write("\033[2K\033[G")
    sys.stdout.flush()

# 終了時などに音で通知する関数
def notify_sound():
    import winsound
    # 1000Hzで1秒間鳴らす
    winsound.Beep(500, 500)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    path = "D:/TestAdvisor/Analyzers/Data/RawData/USDJPY/2015.01.02_2015.12.30.json"
    df = pd.read_json(path)
    df.columns = map(str.capitalize, df.columns)
    df['Current'] = pd.to_datetime(df['Current'])
    df = df.set_index('Current')

    pips = 0.01
    # interpolated_data = interpolate_df(df)
    interpolated_data = df
    # interpolated_data.loc[:, 'Close'] = interpolated_data.loc[:, 'Close'] / pips
    # interpolated_data.loc[:, 'High'] = interpolated_data.loc[:, 'High'] / pips
    # interpolated_data.loc[:, 'Low'] = interpolated_data.loc[:, 'Low'] / pips
    # interpolated_data.loc[:, 'Open'] = interpolated_data.loc[:, 'Open'] / pips
    # interpolated_data.loc[:, 'Spread'] = interpolated_data.loc[:, 'Spread'] / pips

    # 12時間程度で過学習になって上昇が止まる、10時間くらいのスパンで行ったほうがいいか
    # またmodelにdropoutなどを採用して過学習を防ぐ
    # DRQNに変更して過去の状態を考慮する
    # もしくはR2D2を採用

    # double dqnはTargetとQnetを使い分ける奴なのでこの初期コードですでに実現されている
    # 入力の長さ
    input_sequence_length = 512
    # データを切り取る長さ 解析の際に戦闘のパディングが必要なためinput_sequence_lengthより長い必要がある
    sequence_length = int(input_sequence_length * 1.5)
    # ラベルが考慮する値動きの長さ これを調整するとmodelがどの程度までの値動きを考慮するか変わる
    label_sequence_length = 30

    trader = Trader()
    state_channels = trader.get_state_channel_size()
    print("State Channel Size is", state_channels)
    # draw_trader = TraderDrawer()

    agent_model = LSTMAgent(sequence_length=input_sequence_length, state_channel_size=state_channels, batch_size=180)

    total_profit = 0
    min_limit_profit = -10000
    done = False

    profit_histries = np.array([[interpolated_data.index[sequence_length], 0]])

    limit_time = 60 * 24 * 16

    learning_times = []

    last_cmd_time = interpolated_data.index[0]
    show_cmd_time_interval = pd.Timedelta(minutes=120)

    tecAnalyzer = TechnicalAnalyzer()

    print("Start learning:", datetime.datetime.now())

    for i in range(0, len(interpolated_data) - sequence_length, 1):
        sequence = interpolated_data[i:i+sequence_length]

        if (i > limit_time):
            break

        now = sequence.index[-1]
        if now - last_cmd_time > show_cmd_time_interval:
            remove_cmd_line()
            print(now, end="", flush=True)
            last_cmd_time = now
        close = sequence.iloc[-1]['Close']
        spread = sequence.iloc[-1]['Spread']

        g_state = trader.get_state(sequence, input_sequence_length)

        # index=0がgraph用のtimeなので削除
        state = g_state[1:].astype(np.float32)
        # stateのshapeが (input_size, sequence_length) なので(input_size, sequence_length)に変換
        state = np.array(np.split(state, state.shape[1], axis=1)).squeeze(2).astype(np.float32)

        order_raw_value = agent_model.action(state)
        order_type = OrderType(order_raw_value)
        
        profit = trader.take_action(close, spread, now, order_type)
        # next_state = trader.get_state(sequence, input_sequence_length)
        # next_state = next_state[1:].astype(np.float32)
        # next_state = np.array(np.split(next_state, next_state.shape[1], axis=1)).squeeze(2).astype(np.float32)

        feature_sequence = interpolated_data[i+1:i+label_sequence_length+1]
        label_q_values = trader.calculate_q_values(close, spread, feature_sequence)
        label_q_values = np.array(list(label_q_values.values()), dtype=np.float32)

        total_profit += profit
        if profit != 0:
            profit_histries = np.append(profit_histries, [[now, total_profit]], axis=0)

        if total_profit < min_limit_profit:
            done = True
        done = total_profit < min_limit_profit

        agent_model.remember(state, order_raw_value, label_q_values, profit)
        learning = agent_model.replay()

        if learning:
            learning_times.append(now)
    
    notify_sound()
    print("\nEnd learning:", datetime.datetime.now())
    trader.show_intervals()
    
    profits = [ o.profit_pips for o in trader.orders_history]
    profits = np.cumsum(profits)
    times = [o.checkout_time for o in trader.orders_history]
    fig, ax = plt.subplots()
    ax.plot(times, profits, label="Profit")
    for learning_time in learning_times:
        ax.axvline(learning_time, color='r', linestyle='--', linewidth=0.5)
    plt.savefig("models/bidirectional.png")
    plt.show()
    

