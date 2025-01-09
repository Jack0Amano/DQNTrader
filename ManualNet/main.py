import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import talib as ta
import CommonAnalyzer as ca
import enum
import keyboard

class Analyzer:
    """
    データを解析するクラス
    """
    def __init__(self):
        pass

    # ボリンジャーバンドは、移動平均線を中心に、上下に標準偏差の幅を取ったバンドを描くことで、
    # 価格の変動の範囲を示す指標です。
    # バンドの幅が狭いときはボックス相場、幅が広いときはトレンド相場と判断することができます。
    def bollinger_band(self, df, window=20, sigma=1) -> np.array:
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
        return gap.values

    # RSIを計算
    # RSIは現在の相場の相対的な強弱を示す指標でボックス相場に強い
    # RSIが30%以下の場合は売られ過ぎと見て買い、70%以上の場合は買われ過ぎと見て売り
    # ダイバージェンスとして価格が高値を更新しながらRSIが高値を更新していない動きの場合は相場の反転を示す
    # 逆に価格が安値を更新しながらRSIが安値を更新していない場合も相場の反転を示す
    # windowは14が一般的だが、9,22,42,52なども使われる
    def rsi(self, df, window=14) -> np.array:
        """
        RSIを計算するメソッド
        """
        rsi: np.array = ta.RSI(df['Close'].values, timeperiod=window)
        rsi = rsi[~np.isnan(rsi)] / 100
        
        return rsi
        
    # MACDは、移動平均線のクロスオーバーを利用して、トレンドの転換を捉える指標です。
    # このグラフの場合MACDがsignalを超えた時に買いトレンドで
    # MACDがsignalを下回った時に売りトレンドと判断することができます。
    # ただし、MACDはボックス相場の際にダマシが多いため、他の指標と組み合わせて利用することが望ましいです。
    def macd(self, df) -> tuple:
        """
        MACDを計算するメソッド   
        return: MACD, MACDsignal
        """
        macd, macdsignal, macdhist = ta.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[~np.isnan(macd)]
        macdsignal = macdsignal[~np.isnan(macdsignal)]
        return macd, macdsignal
    
    # ストキャスティクスは、過去の価格の範囲に対する現在の価格の位置を示す指標です。
    # MACDがトレンド相場で価格の反転が認識できるが、ストキャスティクスはボックス相場で有効です。
    # %Dが0～20%の範囲にある場合は売られ過ぎと見て買い
    # %Dが80～100%の範囲にある場合は買われ過ぎと見て売り
    def stochastic_oscillator(self, df):
        """
        ストキャスティクスオシレータを計算するメソッド
        """
        k, d = ta.STOCH(df['High'].values, df['Low'].values, df['Close'].values, fastk_period=5, slowk_period=3, slowd_period=3)
        k = k[~np.isnan(k)]
        d = d[~np.isnan(d)]
        return k, d
    
    def adx(self, df, window=14, trend_threshold=25) -> np.array:
        """
        ADXを計算するメソッド   
        ADXが1を上回るとトレンド相場、diff_diが+になると上昇トレンド、-になると下降トレンド   
        return: ADX, diff_di
        """
        adx = ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        minus_di = ta.MINUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        plus_di = ta.PLUS_DI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window)
        adx = adx[~np.isnan(adx)]
        minus_di = minus_di[~np.isnan(minus_di)]
        plus_di = plus_di[~np.isnan(plus_di)]
        adx = adx / trend_threshold
        diff_di = plus_di - minus_di
        return adx, diff_di

class Action(enum.Enum):
    """
    Actionを定義するクラス   
    """
    BUY = 1
    SELL = -1
    NONE = 0
    TAKE_PROFIT = 2

class Position:
    # 注文の時間
    order_time: pd.Timestamp
    # 注文の価格 実際の値
    raw_value: float
    # 注文の種類
    order_type: Action
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
    # 損切りライン　0の場合は損切りラインを設定しない
    stop_loss_line = 0

    def __init__(self, time: pd.Timestamp, value: float, order_type: Action, stop_loss_line=0, order_lot=0.01):
        """
        注文を作成する"""
        self.order_time = time
        self.raw_value = value
        self.order_type = order_type
        self.order_lot = order_lot
        self.order_currency = self.ONE_LOT_CURRENCY * self.order_lot
        self.stop_loss_line = stop_loss_line
    
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
        if self.order_type == Action.BUY:
            output = (price - self.raw_value - spread_real) / self.pip
        elif self.order_type == Action.SELL:
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
    
    def __str__(self):
        return f"{self.order_time}, {self.raw_value}, {self.order_type}, stop: {self.stop_loss_line}"

class Judge:
    """
    Analyzerからの情報をもとにどのような行動を取るかを決定するクラス   
    """
    # これらの各プロパティは手動で調整する方法
    # トレンドの方向性を判断する際のwindow
    trend_direction_window = 14

    # ボリンジャーバンドがボックスかトレンドかを判断するWindow
    bollinger_band_trend_rate_window = 5
    # ボリンジャーバンドがトレンドと判断する閾値
    bollinger_band_trend_rate = 0.3
    # RSIの信用度を計算する際に使う係数
    rsi_coefficient = 1
    # MACDの信用度を計算する際に使う係数
    macd_coefficient = 1
    # ストキャスティクスの信用度を計算する際に使う係数
    stochastic_oscillator_coefficient = 1

    # rsiのトレンド反転を判断する際のwindow
    rsi_trend_reversal_window = 14
    # rsiで売られすぎと判断する閾値
    rsi_sell_threshold = 0.3
    # rsiで買われすぎと判断する閾値
    rsi_buy_threshold = 0.7
    # rsiで最新のトレンドの傾向を判断する際のwindow
    rsi_fast_trend_window = 5

    # MACDのトレンド反転を判断する際のwindow
    macd_trend_reversal_window = 14

    # ストキャスティクスのトレンド反転を判断する際のwindow
    stochastic_oscillator_trend_reversal_window = 14
    # ストキャスティクスで売られすぎと判断する閾値
    stochastic_oscillator_sell_threshold = 0.2
    # ストキャスティクスで買われすぎと判断する閾値
    stochastic_oscillator_buy_threshold = 0.8
    # ストキャスティクスのダイバージェンス（乖離）を判断する際のwindow
    stochastic_oscillator_divergence_window = 14

    # トレンドの反転が起こりうると判断する閾値 0~1の値を取る
    trend_reversal_threshold = 0.6
    # トレンドの反転を期待しているときの閾値
    trend_reversal_expect_threshold = 0.9


    def __init__(self):
        self.analyzer = Analyzer()

    def credits(self, df):
        """
        現在の状況からどのanalyzerの各メソッドの信用度を計算する
        """
        bollinger_band = self.analyzer.bollinger_band(df)
        # 現在がトレンドかボックス相場かを判定
        bollinger_band = bollinger_band[-self.bollinger_band_trend_rate_window:]
        trend_rate = bollinger_band.mean() / self.bollinger_band_trend_rate
        trend_rate = 1 if trend_rate > 1 else trend_rate

        # trend_rateとrsi_coefficientからRSIの信用度を計算
        # rsiはボックス相場に強いためtrend_rateと反比例の関係にある
        rsi_credit = (1 - trend_rate) * self.rsi_coefficient
        
        # MACDの信用度を計算
        # MACDはボックス相場の際にダマシが多いためtrend_rateが高いほど信用度が高くなる
        macd_credit = trend_rate * self.macd_coefficient

        # ストキャスティクスの信用度を計算
        # ストキャスティクスはボックス相場で有効なためtrend_rateと反比例の関係にある
        stochastic_oscillator_credit = (1 - trend_rate) * self.stochastic_oscillator_coefficient

        return rsi_credit, macd_credit, stochastic_oscillator_credit
    
    # 各種の解析によるトレンドの方向を示す
    def trend_direction(self, df) -> tuple:
        """
        各種の解析によるトレンドの方向を示す     
        return: トレンドの方向と強さを示す   
        """
        # ----------------------------
        # adxによるトレンドの方向を計算 1もしくは-1を越えると強いトレンドとされる
        adx, di = self.analyzer.adx(df)
        adx = adx[-self.trend_direction_window:]
        di = di[-self.trend_direction_window:]
        adx_mean = adx.mean()
        di_mean = di.mean()
        trend_direction = di_mean / abs(adx_mean)
        trend = adx_mean * trend_direction

        return trend

    def rsi_trend_reversal(self, df) -> int:
        """
        トレンド転換を判断するメソッド   
        returns: (トレンドの転換上昇下降, 
        """
        
        # ----------------------------
        # rsiによるトレンドの反転を計算
        self.rsi = self.analyzer.rsi(df)
        rsi = self.rsi
        rsi_fast_trend = rsi[-self.rsi_fast_trend_window:]
        downtrend_ratio, sideways_ratio, uptrend_ratio = ca.trend_ratios(rsi_fast_trend, 0.01)
        rsi_mean = rsi[-self.rsi_trend_reversal_window:]
        rsi_mean = rsi.mean()
        # rsiの平均が70%以上つまり買われすぎの場合
        if rsi_mean > self.rsi_buy_threshold:
            if downtrend_ratio > uptrend_ratio and downtrend_ratio > sideways_ratio:
                # rsiの下降が始まった場合
                return -1
            else:
                # rsiが横ばいか上昇している場合
                return 0
        # rsiの平均が30%以下つまり売られすぎの場合
        elif rsi_mean < self.rsi_sell_threshold:
            if uptrend_ratio > downtrend_ratio and uptrend_ratio > sideways_ratio:
                # rsiの上昇が始まった場合
                return 1
            else:
                # rsiが横ばいか下降している場合
                return 0
        else:
            # rsiが閾値内の場合
            return 0
    
    def macd_trend_reversal(self, df) -> int:
        """
        MACDによるトレンドの反転を計算するメソッド   
        """
        output = {}

        # MACDによるトレンドの反転を計算
        macd, macdsignal = self.analyzer.macd(df)
        self.macd, self.macdsignal = macd, macdsignal
        macd = macd[-self.macd_trend_reversal_window:]
        macd_signal = macdsignal[-self.macd_trend_reversal_window:]
        if macd[0] < macd_signal[0] and macd[-1] > macd_signal[-1]:
            # MACDがsignalを超えた場合
            return 1
        elif macd[0] > macd_signal[0] and macd[-1] < macd_signal[-1]:
            # MACDがsignalを下回った場合
            return -1
        else:
            # MACDがsignalとクロスしていない場合
            return 0
    
    def stochastic_oscillator_trend_reversal(self, df) -> dict:
        """
        ストキャスティクスによるトレンドの反転を計算するメソッド   
        return: 各種の解析によるトレンドの転換上昇下降を返す valueはこれからの予測が-1(下降) 0(横ばい) 1(上昇)
        """
        output = {}

        # ストキャスティクスによるトレンドの反転を計算
        k, d = self.analyzer.stochastic_oscillator(df)
        self.stochastic_k, self.stochastik_d = k, d
        self.stochastic_k = self.stochastic_k/100
        self.stochastik_d = self.stochastik_d/100
        k = k[-self.stochastic_oscillator_trend_reversal_window:]
        d = d[-self.stochastic_oscillator_trend_reversal_window:]
        is_golden_cross = k[0] < d[0] and k[-1] > d[-1]
        is_dead_cross = k[0] > d[0] and k[-1] < d[-1]   
        output["golden_dead_cross"] = int(is_golden_cross) - int(is_dead_cross)

        values = df.iloc[-self.stochastic_oscillator_divergence_window:]["Close"].values
        is_higher_value = ca.is_higher_value(values)
        is_lower_value = ca.is_lower_value(values)

        fast_stochastic_trends = self.__stochastic_oscillator_trends(k, is_higher_value, is_lower_value, "fast")
        slow_stochastic_trends = self.__stochastic_oscillator_trends(d, is_higher_value, is_lower_value, "slow")
        output = output | fast_stochastic_trends | slow_stochastic_trends
        
        is_strong_golden_cross = is_golden_cross and k[-1] < self.stochastic_oscillator_sell_threshold
        is_strong_dead_cross = is_dead_cross and k[-1] > self.stochastic_oscillator_buy_threshold
        output["strong_golden_dead_cross"] = int(is_strong_golden_cross) - int(is_strong_dead_cross)

        return output
    
    def __stochastic_oscillator_trends(self, arr: np.array, is_higher_value, is_lower_value, add_word) -> dict:
        """
        ストキャスティクスでのトレンドを判断するメソッド   
        arr: ストキャスティクスの値 %Kは敏感な判断 %Dは滑らかな判断   
        is_higher_value: 現在の価格が最高値を更新しているか   
        is_lower_value: 現在の価格が最安値を更新しているか   
        add_word: 出力のdictのキーに追加する文字列
        """
        output = {}

        overbought = arr[-1] > self.stochastic_oscillator_buy_threshold
        oversold = arr[-1] < self.stochastic_oscillator_sell_threshold

        output[f"{add_word} oversold_bought"] = int(oversold) - int(overbought)
        # ダイバージェンス
        downtrend_ratio, sideways_ratio, uptrend_ratio = ca.trend_ratios(arr, 0.01)
        is_uptrend = uptrend_ratio > downtrend_ratio and uptrend_ratio > sideways_ratio
        is_downtrend = downtrend_ratio > uptrend_ratio and downtrend_ratio > sideways_ratio
        output[f"{add_word} divergence_trend"] = int(is_lower_value and is_uptrend) - int(is_higher_value and is_downtrend)
        # トレンドフォロー 中立域でのトレンド方向
        output[f"{add_word} trendfollow_trend"] = 0
        mean = arr.mean()
        if self.stochastic_oscillator_sell_threshold < mean and mean < self.stochastic_oscillator_buy_threshold:
            # 中立域でのトレンド方向
            if mean > 0.5:
                output[f"{add_word} trendfollow_trend"] = 1
            else:
                output[f"{add_word} trendfollow_trend"] = -1
        
        return output

    def get_support_line(self, df, position_time: pd.Timestamp, window=14) -> np.array:
        """
        サポートラインを計算するメソッド   
        """
        start_window_time = position_time - pd.Timedelta(minutes=window-1)
        min_value = df.loc[start_window_time:position_time, "Low"].min()
        return min_value

    def get_resistance_line(self, df, position_time: pd.Timestamp, window=14) -> np.array:
        """
        レジスタンスラインを計算するメソッド   
        """
        start_window_time = position_time - pd.Timedelta(minutes=window-1)
        max_value = df.loc[start_window_time:position_time, "High"].max()
        return max_value

    def manual_judge(self, df, position: Position) -> Action:
        """
        各種の解析による売買の判断を手動で行うメソッド     
        df: 解析するデータフレーム   
        position: 現在のポジション   

        """

        # 各解析のトレンドの方向を計算
        trend = self.trend_direction(df)
        # 各解析のトレンドの転換を計算
        rsi_trend_reversal = self.rsi_trend_reversal(df)
        macd_trend_reversal = self.macd_trend_reversal(df)
        stochastic_oscillator_trend_reversal_dict = self.stochastic_oscillator_trend_reversal(df)
        
        # 各分析のトレンド転換の方向を判断
        # 重み付き平均値を計算
        # 本来であればcredit計算機能を使わずに各種解析から導き出される最適な行動を機械学習で学習させる
        # その際には各解析の信用度を計算する機能は不要
        rsi_credit, macd_credit, stochastic_oscillator_credit = self.credits(df)
        rsi_trend_reversal *= rsi_credit
        macd_trend_reversal *= macd_credit
        stochastic_oscillator_trend_reversal_dict = {key: int(value) * stochastic_oscillator_credit for key, value in stochastic_oscillator_trend_reversal_dict.items()}
        # ストキャスティクスでのトレンド方向
        st_fast_trend_follow = stochastic_oscillator_trend_reversal_dict.pop('fast trendfollow_trend')
        st_slow_trend_follow = stochastic_oscillator_trend_reversal_dict.pop('slow trendfollow_trend')

        analyzed_values = np.array([rsi_trend_reversal, macd_trend_reversal] + list(stochastic_oscillator_trend_reversal_dict.values()))
        trend_reversal = np.mean(analyzed_values)

        if position is None:
            # ポジションがない場合
            if abs(trend_reversal) > self.trend_reversal_threshold and abs(trend_reversal) < self.trend_reversal_expect_threshold:
                # 現在のトレンドからの反転の可能性がある状態
                return Action.NONE
            elif abs(trend_reversal) > self.trend_reversal_expect_threshold:
                # 現在のトレンドからの反転が高く期待される状態
                return Action.BUY if trend_reversal > 0 else Action.SELL
            else:
                # 強いトレンドが続いている状態
                if abs(trend) > 1:
                    return Action.BUY if trend > 0 else Action.SELL
                else:
                    return Action.NONE
        else:
            # positionにstop_loss_lineが設定されている場合の損切
            if position.stop_loss_line != 0:
                if position.order_type == Action.BUY:
                    if df.iloc[-1]["Low"] < position.stop_loss_line:
                        return Action.TAKE_PROFIT
                elif position.order_type == Action.SELL:
                    if df.iloc[-1]["High"] > position.stop_loss_line:
                        return Action.TAKE_PROFIT

            current_profit = position.calculate_profit(df.index[-1], df.iloc[-1]["Close"], df.iloc[-1]["Spread"])
            if current_profit < 0:
                return Action.NONE

            # ポジションに対して強いトレンドが続いている場合
            if trend > 1 and position.order_type == Action.BUY:
                return Action.NONE
            elif trend < -1 and position.order_type == Action.SELL:
                return Action.NONE

            # ポジションに対してトレンドの反転が起こりうる場合
            if abs(trend_reversal) > self.trend_reversal_threshold:
                return Action.NONE
            elif trend_reversal > self.trend_reversal_expect_threshold:
                if position.order_type == Action.BUY and trend_reversal < 0:
                    # 買いポジションで売り方向へのトレンド転換が起こりうる場合
                    return Action.TAKE_PROFIT
                elif position.order_type == Action.SELL and trend_reversal > 0:
                    # 売りポジションで買い方向へのトレンド転換が起こりうる場合
                    return Action.TAKE_PROFIT
                else:
                    return Action.NONE
            else:
                return Action.NONE


class AutoJudge:
    """
    Judgeを機械学習を使用して行うクラス   
    """

class Trader:
    """
    ポジションを管理するクラス   
    """
    judge: Judge
    # 現在のポジション ポジションは1つしか保持しない
    position = None
    # 現在の利益　(pip)
    total_profit = 0
    # 取引終了したポジションのリスト
    done_positions = []

    def __init__(self):
        self.judge = Judge()
            
    
    def update(self, df):
        """
        売買を各フレームごとに更新するメソッド   
        """
        action = self.judge.manual_judge(df, self.position)

        if action == Action.BUY:
            support_line = self.judge.get_support_line(df, df.index[-1], 30)
            self.position = Position(df.index[-1], df.iloc[-1]["Close"], Action.BUY, support_line)
            print(self.position)
        elif action == action.SELL:
            resistance_line = self.judge.get_resistance_line(df, df.index[-1], 30)
            self.position = Position(df.index[-1], df.iloc[-1]["Close"], Action.SELL, resistance_line)
            print(self.position)
        elif action == Action.TAKE_PROFIT:
            value = df.iloc[-1]["Close"]
            self.total_profit += self.position.checkout(df.index[-1], value, df.iloc[-1]["Spread"])
            self.done_positions.append(self.position)
            print(f"{self.position.checkout_time}, {value}, Take Profit: {self.position.profit_pips}")
            self.position = None

class TraderDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot([], [], label="Close")  
        self.ax2 = self.ax.twinx()
        self.tranings = []
        self.paused = False
        keyboard.on_press(self.toggle_pause)

    def toggle_pause(self, event): 
        if event.name == 'space':
            self.paused = not self.paused 

    def update_graph(self, closes: np.array, times, values: dict, order: Position, profits, traning:pd.Timestamp=None, show_length=120):
        """
        状態をグラフに描画する   
        return: Falseを返すとグラフが閉じられたことを示す   
        """
        if not plt.fignum_exists(self.fig.number):
            return False

        # ポーズに対応
        while self.paused:
            # matplotlibのウィンドウが閉じられた場合
            if not plt.fignum_exists(self.fig.number):
                return False
            plt.pause(0.1)

        self.fig.delaxes(self.ax)
        self.fig.delaxes(self.ax2)

        times = times[times>=times[-1] - pd.Timedelta(minutes=show_length)]
        # timesをdatetimeに変換
        times = [pd.Timestamp(t) for t in times]
        closes = closes[-len(times):]

        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        # ax2のy軸は右側に表示
        self.ax2.yaxis.tick_right()
        self.ax.plot(times, closes, label="Close")

        # if order is not None:
        #     if order.order_type == Action.BUY:
        #         self.ax.axhline(order.raw_value, color='r')
        #     elif order.order_type == Action.SELL:
        #         self.ax.axhline(order.raw_value, color='g')

        if values is not None:
            
            for (key, value) in values.items():
                value = value[-len(times):]
                self.ax2.plot(times, value, label=key)
            self.ax2.legend(loc='upper left')

        
        # for traning in self.tranings:
        #     self.ax.axvline(traning, color='b')

        # profitをax1に描画
        # self.ax1 = self.fig.add_subplot(212)
        # profit_times = profits[:, 0]
        # profit_values = profits[:, 1]
        # self.ax1.plot(profit_times, profit_values, label="Profit")
        
        plt.pause(0.0001)

        return True

    def save_graph(self, path):
        plt.savefig(path)
        
# 最終的なModelに入力するデータは
# dataframeの形式で各時間ごとの (buy, sell) のパーセントの値
#        buy  sell
# 12:00  0.3   0.7
# 12:01  0.2   0.8
# 12:02  0.1   0.9
# という感じで
# 現在のポジションに対するtakeprofitはsellやbuyとはまた違った別のアルゴリズムで計算する
# ポジション設定よりも損切りのほうが重要
# ポジション設定後の損切りラインの設定の手法が多すぎてどれがいいかわからない
# 損切りネットワークが必要
# すなわちポジションの設定理由とこれを否定する材料
# ネットワークでなくても手動のアルゴリズムでもとりあえずは入力と出力を決める必要がある
# 出力はTakeProfitするべきかどうかのBool値
# とりあえず決定木で実装してみる
# Positionのクラス設定時にSupportLineやResistanceLineなど損切りラインをあらかじめ設定しておく
# judge関数でactionを取得する際に同時に損切りラインを設定しておく
# サポートラインやレジスタンスラインは何を根拠にしてという情報なしで、ポジションの設定値のみで計算できる
# またフィボナッチリトレースメントも同様に何を根拠にしたかという情報は必要なく、ポジションの設定値のみで計算できる
# しかし、シグナル否定での損切ラインに関しては、何を根拠にしているかの情報が必要
# そのため、シグナル否定は各種解析時点での情報を保持しておく必要があり、それと同時に損切ラインをjudge関数に渡す必要がある
# 初期の決定木ではシグナル否定の損切ラインは考慮しないことにする


if __name__ == "__main__":
    path = "D:/TestAdvisor/Analyzers/Data/RawData/USDJPY/2015.01.02_2015.12.30.json"
    df = pd.read_json(path)
    df.columns = map(str.capitalize, df.columns)
    df['Current'] = pd.to_datetime(df['Current'])
    df = df.set_index('Current')

    # 入力の長さ
    input_sequence_length = 512
    # データを切り取る長さ 解析の際に戦闘のパディングが必要なためinput_sequence_lengthより長い必要がある
    sequence_length = int(input_sequence_length * 1.5)
    # ラベルが考慮する値動きの長さ これを調整するとmodelがどの程度までの値動きを考慮するか変わる
    label_sequence_length = 30


    trader = Trader()
    trader_drawer = TraderDrawer()

    limit_time = 60 * 24 * 16

    max_value = -np.inf
    min_value = np.inf

    for i in range(0, len(df) - sequence_length, 1):
        if i > limit_time:
            break

        sequence = df[i:i+sequence_length]
        close = sequence.iloc[-1]["Close"]
        trader.update(sequence)
        values = {"macd": trader.judge.macd,
                  "macdsignal": trader.judge.macdsignal,
                  "rsi": trader.judge.rsi,
                  "stochastic_k": trader.judge.stochastic_k,
                  "stochastic_d": trader.judge.stochastik_d}

        graph_state = trader_drawer.update_graph(sequence["Close"].values, sequence.index, values, trader.position, None)
        if not graph_state:
            break
    print(close)
    print(len(trader.done_positions), trader.total_profit)
