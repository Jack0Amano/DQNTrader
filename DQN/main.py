import datetime
import pandas as pd
import numpy as np
import enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from model import DQNAgent
from technicalAnalyzer import TechnicalAnalyzer
import torch

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

    
class Trader:
    orders = []
    current_order: Order = None
    first_time = None
    # 利確圧力を加えていくまでの時間
    limit_time_after_checkout = pd.Timedelta(minutes=120)

    state_channels = {
        "Close": True,
        "Profit": True,
        "OrderType": True,
        "Volume": True
    }

    def __init__(self):
        self.tec_analy = TechnicalAnalyzer()

    def take_action(self, value: float, spread: float, time: pd.Timestamp, order_type: OrderType) -> tuple:
        """
        現在の価格と行う場合は注文を受け取り、注文を行う   
        ここの報酬を変えることによって行動の評価を変えることができる   
        つまりここはルールと報酬を決める部分である   
        value: 現在の価格   
        spread: スプレッド   
        order_time: 現在の時間   
        order_type: 注文の種類   
        returns: (行動の報酬, 利益)
        """

        if self.first_time is None:
            self.first_time = time
        
        if self.current_order is None:
            if order_type == OrderType.BUY:
                self.current_order = Order(time, value, OrderType.BUY)
                print("Buy order is taken at", time, "Value is", value)
                return 0, 0
            elif order_type == OrderType.SELL:
                self.current_order = Order(time, value, OrderType.SELL)
                print("Sell order is taken at", time, "Value is", value)
                return 0, 0
            elif order_type == OrderType.TAKE_PROFIT:
                print("There is no order to checkout")
                return -10, 0
            elif order_type == OrderType.NOTHING:
                # 最後の注文から一定時間経過している場合は報酬を減らしていく
                if len(self.orders) > 0:
                    last_order: Order = self.orders[-1]
                    order_time_delta: pd.Timedelta = time - last_order.checkout_time
                    if order_time_delta > self.limit_time_after_checkout:
                        return -10, 0
                else:
                    # first_timeから一定時間経過している場合は報酬を減らしていく
                    # 何も買わずに終えるのを防ぐため
                    order_time_delta: pd.Timedelta = time - self.first_time
                    if order_time_delta > self.limit_time_after_checkout:
                        return -10, 0
        else:
            if order_type == OrderType.TAKE_PROFIT:
                profit = self.current_order.checkout(time, value, spread)
                print("Checkout order is taken at", time, "Value is", value, "Profit is", profit)
                self.orders.append(self.current_order)
                pip = self.current_order.pip
                self.current_order = None
                return profit / pip * 100, profit
            elif order_type == OrderType.BUY or order_type == OrderType.SELL:
                # ポジションは１つしか許容しないため、報酬を下げる
                return -10, 0
            elif order_type == OrderType.NOTHING:
                # limit_timeを超えた場合で、注文の現在の利益がマイナスの場合はこれと時間に応じて報酬を減らしていく
                order_time_delta: pd.Timedelta = time - self.current_order.time
                if order_time_delta > self.limit_time_after_checkout:
                    profit = self.current_order.calculate_profit(value, spread)
                    if profit < 0:
                        # 一定時間経過しても利益が出ない場合は報酬を減らしていく
                        reward = (-10 * -profit) -10 * order_time_delta.seconds / 60
                        return reward, 0
                    else:
                        # 利益が出ている場合も一定時間経過しているため報酬を減らしていく
                        return - (10 * order_time_delta.seconds / 60), 0
                else:
                    # 一定時間経過していない場合の処理はとりあえずなし
                    pass
                    
        return 0, 0

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

        output.append(times)

        if self.state_channels["Close"]:
            closes_normaled = (closes - closes.min()) / (closes.max() - closes.min()) 
            closes_normaled = np.array(closes_normaled)
            output.append(closes_normaled)

        self.close_intervals.append((datetime.datetime.now() - start_time).total_seconds())
        start_time = datetime.datetime.now()

        
        if self.state_channels["Profit"]:
            # # 利益は 0,0,0,... ,(order), profit(0), profit(1)となる
            # if self.current_order is not None:
            #     order_time = self.current_order.time.to_datetime64()
            #     profits = [self.current_order.calculate_profit(t, c, s) if t >= order_time else 0 for c, s, t in zip(closes, spreads, times)]
            
            # # TODO: これがすごく遅いので改善する
            # # 過去の注文がある場合もそれを埋める
            # for order in self.orders_history:
            #     order_time = order.time.to_datetime64()
            #     checkout_time = order.checkout_time.to_datetime64()
            #     if times[0] > checkout_time:
            #         continue
            #     old_profits = [order.calculate_profit(t, c, s) if t >= order_time and t <= checkout_time else p for c, s, t, p in zip(closes, spreads, times, profits)]
            #     profits = [op if abs(op) > abs(p) else p for p, op in zip(profits, old_profits)]

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
                    reward = (current_price - self.current_order.value) - current_spread
                elif self.current_order.order_type == OrderType.SELL:
                    reward = (self.current_order.value - current_price) - current_spread
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
                    current_profit = current_price - self.current_order.value
                elif self.current_order.order_type == OrderType.SELL:
                    current_profit = self.current_order.value - current_price
                else:
                    current_profit = 0

                # 将来の期待利益を計算
                buy_profit = max(self.__calculate_feature_profits(OrderType.BUY, self.current_order.value, future_prices, future_spread))
                sell_profit = max(self.__calculate_feature_profits(OrderType.SELL, self.current_order.value, future_prices, future_spread))
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

    def update_graph(self, state: np.array, order: Order, profits, show_length=120):
        """
        状態をグラフに描画する
        """
        self.fig.delaxes(self.ax)
        self.fig.delaxes(self.ax1)

        times = state[0][-show_length:]
        # timesをdatetimeに変換
        times = [pd.Timestamp(t) for t in times]
        closes = state[1][-show_length:]

        self.ax = self.fig.add_subplot(211)
        self.ax.plot(times, closes, label="Close")

        if order is not None:
            if order.order_type == OrderType.BUY:
                self.ax.axhline(order.value, color='r')
            elif order.order_type == OrderType.SELL:
                self.ax.axhline(order.value, color='g')

        # profitをax1に描画
        self.ax1 = self.fig.add_subplot(212)
        profit_times = profits[:, 0]
        profit_values = profits[:, 1]
        self.ax1.plot(profit_times, profit_values, label="Profit")

        plt.pause(0.0001)

    def save_graph(self, path):
        plt.savefig(path)

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
    draw_trader = TraderDrawer()

    agent_model = DQNAgent(sequence_length=input_sequence_length, state_channel_size=state_channels, batch_size=180)

    total_profit = 0
    min_limit_profit = -10000
    done = False

    profit_histries = np.array([[interpolated_data.index[sequence_length], 0]])

    limit_time = 60 * 24 * 7

    for i in range(0, len(interpolated_data) - sequence_length, 1):
        sequence = interpolated_data[i:i+sequence_length]

        if (i > limit_time):
            draw_trader.save_graph("models/withFullStateLSTM.png")
            break

        now = sequence.index[-1]
        close = sequence.iloc[-1]['Close']
        spread = sequence.iloc[-1]['Spread']
        g_state = trader.get_state(sequence, input_sequence_length)
        # index=0がgraph用のtimeなので削除
        state = g_state[1:].astype(np.float32)
        # stateのshapeが (input_size, sequence_length) なので(input_size, sequence_length)に変換
        state = np.array(np.split(state, state.shape[1], axis=1)).squeeze(2).astype(np.float32)

        order_raw_value = agent_model.action(state)
        order_type = OrderType(order_raw_value)
        

        reward, profit = trader.take_action(close, spread, now, order_type)
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

        agent_model.remember(state, order_raw_value, reward, label_q_values, profit, done)
        agent_model.replay()

        draw_trader.update_graph(g_state, trader.current_order, profit_histries)


