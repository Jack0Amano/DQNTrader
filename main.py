import datetime
import sys
import pandas as pd
import numpy as np
import enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from agent import RunTimeAgent
import torch
import talib as ta
import CommonAnalyzer as ca
from sklearn.preprocessing import StandardScaler

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

    def __init__(self, total_df):
        self.price_standard_scaler = StandardScaler()
        closes = total_df['Close'].values
        closes = closes.reshape(-1, 1)
        self.price_standard_scaler.fit(closes)

        self.volume_standard_scaler = StandardScaler()
        volumes = total_df['Volume'].values
        volumes = volumes.reshape(-1, 1)
        self.volume_standard_scaler.fit(volumes)


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

        upper_band = sma + std * sigma
        lower_band = sma - std * sigma
        band_width = upper_band - lower_band

        close_values = df.loc[upper_band.index[0]:upper_band.index[-1], 'Close']
        price_positions = (close_values - lower_band) / (upper_band - lower_band)

        upper_band = upper_band.values[-self.length:] / 100
        lower_band = lower_band.values[-self.length:] / 100
        band_width = band_width.values[-self.length:]
        price_positions = price_positions.values[-self.length:]

        upper_band = upper_band.reshape((upper_band.shape[0], 1))
        lower_band = lower_band.reshape((lower_band.shape[0], 1))
        band_width = band_width.reshape((band_width.shape[0], 1))
        price_positions = price_positions.reshape((price_positions.shape[0], 1))

        # band_widthの値はおよそ0~0.5であるためこれを0~1に正規化する
        band_width = band_width / 0.5

        return upper_band, lower_band, band_width, price_positions

    # RSIを計算
    # RSIは現在の相場の相対的な強弱を示す指標でボックス相場に強い
    # RSIが30%以下の場合は売られ過ぎと見て買い、70%以上の場合は買われ過ぎと見て売り
    # ダイバージェンスとして価格が高値を更新しながらRSIが高値を更新していない動きの場合は相場の反転を示す
    # 逆に価格が安値を更新しながらRSIが安値を更新していない場合も相場の反転を示す
    # windowは14が一般的だが、9,22,42,52なども使われる
    def rsi(self, df, window=14, downtrend_threshold=0.7, uptrend_threshold=0.3) -> np.ndarray:
        """
        RSIを計算するメソッド   
        return: []   
        """
        rsi: np.array = ta.RSI(df['Close'].values, timeperiod=window)
        rsi = rsi[~np.isnan(rsi)] / 100
        rsi = rsi[-self.length:]
        rsi = rsi.reshape((rsi.shape[0], 1))
        return rsi
        
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
        macd_diff = macd - macdsignal
        macd_diff = macd_diff[-self.length:]
        macd_cross_direction = np.sign(np.diff(np.sign(macd_diff)))
        macd_cross_direction = np.concatenate([[0], macd_cross_direction])
        macd_cross_direction = macd_cross_direction.reshape((macd_cross_direction.shape[0], 1))

        macd_diff = macd_diff.reshape((macd_diff.shape[0], 1))
        # macd_cross_overの取る値は およそ 0.05 ~ -0.05であるためこれを-1~1に正規化する
        macd_diff = macd_diff / 0.05
        
        return macd_diff, macd_cross_direction
    
    # ストキャスティクスは、過去の価格の範囲に対する現在の価格の位置を示す指標です。
    # MACDがトレンド相場で価格の反転が認識できるが、ストキャスティクスはボックス相場で有効です。
    # %Dが0～20%の範囲にある場合は売られ過ぎと見て買い
    # %Dが80～100%の範囲にある場合は買われ過ぎと見て売り
    def stochastic_value(self, df) -> np.ndarray:
        """
        ストキャスティクスの値を取得するメソッド   
        return: %K, %D
        """
        k, d = ta.STOCH(df['High'].values, df['Low'].values, df['Close'].values, fastk_period=5, slowk_period=3, slowd_period=3)
        k = k[~np.isnan(k)]
        d = d[~np.isnan(d)]
        k = k[-self.length:] / 100
        d = d[-self.length:] / 100
        k = k.reshape((k.shape[0], 1))
        d = d.reshape((d.shape[0], 1))
        golden_dead_diff = k - d
        golden_dead_cross = np.sign(np.diff(np.sign(golden_dead_diff.flatten())))
        golden_dead_cross = np.concatenate([[0], golden_dead_cross])
        golden_dead_cross = golden_dead_cross.reshape((golden_dead_cross.shape[0], 1))
        # golden_dead_crossの取る値は およそ-0.3 ~ 0.3であるためこれを-1~1に正規化する
        golden_dead_diff = golden_dead_diff / 0.3
        return k, d, golden_dead_diff, golden_dead_cross

    # ADXは、トレンドの強さを示す指標です。
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
        # adxの値はおよそ60が最高であるため50で割る
        adx = adx[-self.length:]/50
        minus_di = minus_di[~np.isnan(minus_di)]
        minus_di = minus_di[-self.length:]/50
        plus_di = plus_di[~np.isnan(plus_di)]
        plus_di = plus_di[-self.length:]/50
        adx = adx / trend_threshold
        diff_di = plus_di - minus_di

        # トレンドの方向を示す
        trend_direction = diff_di / abs(adx)
        trend = adx * trend_direction

        trend = trend.reshape((trend.shape[0], 1))

        return trend
    
    # ATRはボラティリティを計測できるため相場分類器の入力として使用
    def atr(self, df, window=14) -> np.ndarray:
        """
        ATRを計算するメソッド   
        return: ATR
        """
        atr = ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        atr = atr[~np.isnan(atr)]
        atr = atr[-self.length:]
        # ATRの幅はおよそ0~0.15であるためこれを0~1に正規化する
        atr = atr / 0.15
        atr = atr.reshape((atr.shape[0], 1))
        return atr

    def moving_average(self, df, window=5) -> np.ndarray:
        """
        移動平均を計算するメソッド   
        return: 移動平均
        """
        ma = df['Close'].rolling(window=window).mean()
        ma = ma.dropna()
        ma = ma[-self.length:]
        ma = ma.values.reshape((ma.shape[0], 1))
        ma = self.price_standard_scaler.transform(ma)
        return ma

    def price(self, df):
        """
        値動きの値を正規化した物を取得する   
        return: [[Open, High, Low, Close], ...]
        """
        # open = df['Open']
        # open = (open - open.min()) / (open.max() - open.min())
        # high = df['High']
        # high = (high - high.min()) / (high.max() - high.min())
        # low = df['Low']
        # low = (low - low.min()) / (low.max() - low.min())
        close = df['Close']
        close = close[-self.length:]
        close = close.values.reshape((close.shape[0], 1))
        close = self.price_standard_scaler.transform(close)
        return close
    
    def volume(self, df):
        """
        取引量を正規化した物を取得する
        return: [[Volume(t)], [Volume(t+1)], ...]
        """
        volume = df['Volume']
        volume = volume[-self.length:]
        volume = np.array(volume)
        volume = volume.reshape((volume.shape[0], 1))
        volume = self.volume_standard_scaler.transform(volume)
        return volume
    
class Trader:
    orders_history = []
    current_order: Order = None
    first_time = None

    def __init__(self, length, total_df):
        self.analyzer = Analyzer(total_df)
        self.analyzer.length = length

    def get_state(self, sequence_data: pd.DataFrame) -> tuple:
        """
        現在までの値動きの状態を取得する   
        return (分類器, ボックス相場, トレンド相場)の各種入力データ
        """

        # ボリンジャーバンド
        upper_band, lower_band, band_width, price_positions = self.analyzer.bollinger_band(sequence_data)
        # ATR
        atr = self.analyzer.atr(sequence_data)
        # RSI
        rsi = self.analyzer.rsi(sequence_data)
        # MACD
        macd_diff, macd_cross = self.analyzer.macd(sequence_data)
        # ストキャスティクス
        stochastic_k, stochastic_d, golden_dead_diff, golden_dead_cross = self.analyzer.stochastic_value(sequence_data)
        # ADX
        adx = self.analyzer.adx(sequence_data)
        # 値動きの値  
        price = self.analyzer.price(sequence_data)
        # 取引量
        volumes = self.analyzer.volume(sequence_data)
        # 移動平均
        ma_5 = self.analyzer.moving_average(sequence_data)
        ma_15 = self.analyzer.moving_average(sequence_data, 15)

        # print("upper_band", upper_band.max(), upper_band.min())
        # print("lower_band", lower_band.max(), lower_band.min())
        # print("band_width", band_width.max(), band_width.min())
        # print("price_positions", price_positions.max(), price_positions.min())
        # print("rsi", rsi.max(), rsi.min())
        # print("macd", macd.max(), macd.min())
        # print("stochastic_k", stochastic_k.max(), stochastic_k.min())
        # print("stochastic_d", stochastic_d.max(), stochastic_d.min())
        # print("golden_dead_cross", golden_dead_cross.max(), golden_dead_cross.min())
        # print("adx", adx.max(), adx.min())
        # print("price", price.max(), price.min())
        # print("volumes", volumes.max(), volumes.min())
        # exit()

        # 分類器の入力データ
        classifier_x = np.concatenate([upper_band, lower_band, band_width, price_positions, atr], 1, dtype=np.float32)
        #　トレンド相場の入力データ
        trend_x = np.concatenate([price, ma_5, ma_15, volumes, macd_diff, macd_cross, adx], 1, dtype=np.float32)
        # ボックス相場の入力データ        
        box_x = np.concatenate([price, band_width, price_positions, rsi, stochastic_k, stochastic_d, golden_dead_diff, golden_dead_cross], 1, dtype=np.float32)
        return classifier_x, box_x, trend_x

    def get_label(self, df: pd.DataFrame, now, pip=0.001, window=13) -> np.ndarray:
        """
        ラベルを取得する   
        return: window後に価格がスプレッドを越えて上昇するか下降するか [横這い, 下降, 上昇]   
        """
        now_index = df.index.get_loc(now)
        start = now_index + 1
        end = now_index + window + 1

        now_close = df.loc[now, 'Close']

        feature_spread = df.iloc[start:end]['Spread'].mean() * pip
        feature_high = df.iloc[start:end]['High'].mean()
        feature_low = df.iloc[start:end]['Low'].mean()
        feature_mean = (feature_high + feature_low) / 2

        if feature_mean - now_close > feature_spread:
            return np.array([[0, 0, 1]], dtype=np.float32)
        elif now_close - feature_mean > feature_spread:
            return np.array([[0, 1, 0]], dtype=np.float32)
        else:
            return np.array([[1, 0, 0]], dtype=np.float32)


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

def load_df(path):
    """
    データを読み込む   
    """
    df = pd.read_json(path)
    df.columns = map(str.capitalize, df.columns)
    df['Current'] = pd.to_datetime(df['Current'])
    df = df.set_index('Current')
    return df

if __name__ == "__main__":
    train_path = "D:/TestAdvisor/Analyzers/Data/RawData/USDJPY/2015.01.02_2015.12.30.json"
    test_path = "D:/TestAdvisor/Analyzers/Data/RawData/USDJPY/2016.01.04_2016.12.30.json"

    train_df = load_df(train_path)
    test_df = load_df(test_path)

    train_start_time = train_df.index[0]
    train_end_time = pd.Timestamp("2015-12-30")

    test_start_time = pd.Timestamp("2016-02-29")
    test_end_time = pd.Timestamp("2016-06-29")

    train_df = train_df.loc[(train_df.index >= train_start_time) & (train_df.index <= train_end_time)]
    test_df = test_df.loc[(test_df.index >= test_start_time) & (test_df.index <= test_end_time)]

    pips = 0.01

    # 入力の長さ
    input_sequence_length = 60
    # データを切り取る長さ 解析の際に戦闘のパディングが必要なためinput_sequence_lengthより長い必要がある
    sequence_length = int(input_sequence_length * 4)
    # ラベルが考慮する期間の長さ
    label_sequence_length = 13

    train_trader = Trader(input_sequence_length, train_df)
    test_trader = Trader(input_sequence_length, test_df)

    classifier_dim = 5
    box_dim = 8
    trend_dim = 7
    label_dim = 3

    agent_model = RunTimeAgent(classifier_dim, box_dim, trend_dim, label_sequence_length)

    last_cmd_time = train_df.index[0]
    show_cmd_time_interval = pd.Timedelta(days=1)

    print("Start learning:", datetime.datetime.now())

    epoch = 10
    
    train_loss_history = []
    test_loss_history = []

    # agent_model.load_model("models/agent_model.pth")

    model_name = "agent_model_2"

    for e in range(epoch):
        # trainデータでの学習を行う
        train_loss = []
        last_cmd_time = train_df.index[0]
        for i in range(0, len(train_df) - sequence_length, 1):
            sequence = train_df[i:i+sequence_length]

            now = sequence.index[-1]
            if now - last_cmd_time > show_cmd_time_interval:
                last_cmd_time = now
                remove_cmd_line()
                print(now, end="", flush=True)

            classifier_x, box_x, trend_x = train_trader.get_state(sequence)
            label = train_trader.get_label(train_df, now, window=label_sequence_length)

            agent_model.remember(classifier_x, box_x, trend_x, label)
            training_loss = agent_model.replay(True)

            if training_loss is not None:
                train_loss.append(training_loss)
        agent_model.clear_memory()
        agent_model.clear_hidden_state()
        train_loss_mean = np.mean(train_loss)
        train_loss_history.append(train_loss_mean)

        print("\nEpoch:", e, "Train loss:", train_loss_mean)

        # testデータでの評価を行う
        test_loss = []
        last_cmd_time = test_df.index[0]
        for i in range(0, len(test_df) - sequence_length, 1):
            sequence = test_df[i:i+sequence_length]
            now = sequence.index[-1]
            if now - last_cmd_time > show_cmd_time_interval:
                last_cmd_time = now
                remove_cmd_line()
                print(now, end="", flush=True)

            classifier_x, box_x, trend_x = test_trader.get_state(sequence)
            label = test_trader.get_label(test_df, now, window=label_sequence_length)

            agent_model.remember(classifier_x, box_x, trend_x, label)
            test_run_loss = agent_model.replay(False)

            if test_run_loss is not None:
                test_loss.append(test_run_loss)
        agent_model.clear_memory()
        agent_model.clear_hidden_state()
        test_loss_mean = np.mean(test_loss)
        test_loss_history.append(test_loss_mean)

        print("\nEpoch:", e, "Train loss:", train_loss_mean, "Test loss:", test_loss_mean)
        agent_model.save_model(f"models/{model_name}_{e}.pth")
    
    notify_sound()
    print("\nEnd learning:", datetime.datetime.now())

    # draw graph
    train_loss_history = np.array(train_loss_history)
    test_loss_history = np.array(test_loss_history)

    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    ax.plot(train_loss_history, label="Train Loss")
    ax.plot(test_loss_history, label="Test Loss")

    ax.legend()
    plt.savefig("models/loss2.png")
    plt.show()

    
    

