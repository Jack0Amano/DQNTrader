import datetime
import pandas as pd
import numpy as np
import enum
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from model import DQNAgent

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
    NONE = 0
    BUY = 1
    SELL = 2
    CHECKOUT = 3

class Order:
    time: pd.Timestamp
    value: float
    order_type: OrderType
    checkout_time: pd.Timestamp

    # 講座基本通貨はJPYとする

    # 講座の1ロットあたりの通貨量
    ONE_LOT_CURRENCY = 100000
    # レバレッジ
    leverage = 15
    # 注文のロット
    order_lot = 0.01
    # 今回の注文での通貨量
    order_currency = 0
    # 決済時のスプレッドを考慮するか 実際はスプレッドがあるためTrueにする
    spread_active = False
    # 1pipあたりの基本通貨における価格
    pip = 0.01

    def __init__(self, time: pd.Timestamp, value: float, order_type: OrderType, order_lot=0.01):
        """
        注文を作成する"""
        self.time = time
        self.value = value
        self.order_type = order_type
        self.order_lot = order_lot
        self.order_currency = self.ONE_LOT_CURRENCY * self.order_lot
    
    def __str__(self):
        return f"Time: {self.time}, Value: {self.value}, OrderType: {self.order_type}"
    
    # 現在の価格と比較しての損益を計算する
    def calculate_profit(self, current_price, spread) -> float:
        """
        現在の価格と比較しての損益を計算する   
        current_price: 現在の価格   
        spread: スプレッド   
        return: 損益（pip)
        """
        if self.order_type == OrderType.BUY:
            return (current_price - self.value) * self.pip - (spread if self.spread_active else 0)
        elif self.order_type == OrderType.SELL:
            return (self.value - current_price) * self.pip - (spread if self.spread_active else 0)
        else:
            return 0
        
    def checkout(self, time: pd.Timestamp, current_price, spread) -> float:
        """
        ポジションの決済   
        time: 決済時間   
        return: 損益(pip) 
        """
        self.checkout_time = time
        return self.calculate_profit(current_price, spread)
    
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

    state_channels = {
        "Close": True,
        "Profit": True,
        "OrderType": True,
        "Volume": False
    }

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
        
        # 注文をして一定時間放置しているときrewardを減らしていく、つまり利確圧力を加えていく
        limit_time = pd.Timedelta(minutes=180)
        # 最後の注文のCheckoutから一定時間すぎるときの報酬を減らしていく
        limit_time_after_checkout = pd.Timedelta(minutes=120)
        
        if self.current_order is None:
            if order_type == OrderType.BUY:
                self.current_order = Order(time, value, OrderType.BUY)
                print("Buy order is taken at", time, "Value is", value)
                return 0, 0
            elif order_type == OrderType.SELL:
                self.current_order = Order(time, value, OrderType.SELL)
                print("Sell order is taken at", time, "Value is", value)
                return 0, 0
            elif order_type == OrderType.CHECKOUT:
                print("There is no order to checkout")
                return -100, 0
            elif order_type == OrderType.NONE:
                # 最後の注文から一定時間経過している場合は報酬を減らしていく
                if len(self.orders) > 0:
                    last_order: Order = self.orders[-1]
                    order_time_delta: pd.Timedelta = time - last_order.checkout_time
                    if order_time_delta > limit_time_after_checkout:
                        return -10, 0
                else:
                    # first_timeから一定時間経過している場合は報酬を減らしていく
                    # 何も買わずに終えるのを防ぐため
                    order_time_delta: pd.Timedelta = time - self.first_time
                    if order_time_delta > limit_time_after_checkout:
                        return -10, 0
        else:
            if order_type == OrderType.CHECKOUT:
                profit = self.current_order.checkout(time, value, spread)
                print("Checkout order is taken at", time, "Value is", value, "Profit is", profit)
                self.orders.append(self.current_order)
                self.current_order = None
                return profit, profit
            elif order_type == OrderType.BUY or order_type == OrderType.SELL:
                # ポジションは１つしか許容しないため、報酬を下げる
                return -100, 0
            elif order_type == OrderType.NONE:
                # limit_timeを超えた場合で、注文の現在の利益がマイナスの場合はこれと時間に応じて報酬を減らしていく
                order_time_delta: pd.Timedelta = time - self.current_order.time
                if order_time_delta > limit_time:
                    profit = self.current_order.calculate_profit(value, spread)
                    if profit < 0:
                        # 一定時間経過しても利益が出ない場合は報酬を減らしていく
                        return (-100 * -profit) - (10 * order_time_delta.seconds / 60), 0
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

        times = np.array(sequence_data.index[-sequence_length:])
        closes = np.array(sequence_data['Close'].values[-sequence_length:])
        spreads = np.array(sequence_data['Spread'].values[-sequence_length:])
        profits = np.zeros(sequence_length)
        order_type = np.zeros(sequence_length)

        output.append(times)

        if self.state_channels["Close"]:
            output.append(closes)
        
        if self.state_channels["Profit"]:
            # 利益は 0,0,0,... ,(order), profit(0), profit(1)となる
            if self.current_order is not None:
                order_time = self.current_order.time.to_datetime64()
                profits = [self.current_order.calculate_profit(c, s) if t >= order_time else 0 for c, s, t in zip(closes, spreads, times)]
                
            # 過去の注文がある場合もそれを埋める
            for order in self.orders:
                order_time = order.time.to_datetime64()
                checkout_time = order.checkout_time.to_datetime64()
                old_profits = [order.calculate_profit(c, s) if t >= order_time and t <= checkout_time else p for c, s, t, p in zip(closes, spreads, times, profits)]
                profits = [op if abs(op) > abs(p) else p for p, op in zip(profits, old_profits)]

            output.append(profits)

        if self.state_channels["OrderType"]:
            # 注文の種類は 0,0,0,... ,(order), order_type(0), order_type(1)となる
            if self.current_order is not None:
                order_time = self.current_order.time.to_datetime64()
                order_type = [self.current_order.order_type.value if t >= order_time else 0 for t in times]
            
            # 過去の注文がある場合もそれを埋める
            for order in self.orders:
                order_time = order.time.to_datetime64()
                checkout_time = order.checkout_time.to_datetime64()
                old_order_type = [order.order_type.value if t >= order_time and t <= checkout_time else ot for t, ot in zip(times, order_type)]
                order_type = [max(ot, oot) for ot, oot in zip(order_type, old_order_type)]

            output.append(order_type)

        if self.state_channels["Volume"]:
            volumes = np.array(sequence_data['Volume'].values[-sequence_length:])
            output.append(volumes)

        return np.array([times, closes, profits, order_type])
    
    def get_state_channel_size(self):
        """
        get_stateが返すのチャンネル数を取得する  
        """
        return self.state_channels.count(True)

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

        plt.pause(0.1)

    def save_graph(self, path):
        plt.savefig(path)

if __name__ == "__main__":

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
    
    # データを切り取る長さ 解析の際に戦闘のパディングが必要なためinput_sequence_lengthより長い必要がある
    sequence_length = 1440
    # 入力の長さ
    input_sequence_length = 512

    trader = Trader()
    state_channels = trader.get_state_channel_size()
    draw_trader = TraderDrawer()

    agent_model = DQNAgent(sequence_length=input_sequence_length, state_channel_size=state_channels)

    total_profit = 0
    min_limit_profit = -10000
    done = False

    profit_histries = np.array([[interpolated_data.index[sequence_length], 0]])

    limit_time = 60 * 12

    for i in range(0, len(interpolated_data) - sequence_length, 1):
        sequence = interpolated_data[i:i+sequence_length]

        if (i > limit_time):
            draw_trader.save_graph("models/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
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
        next_state = trader.get_state(sequence, input_sequence_length)
        next_state = next_state[1:].astype(np.float32)
        next_state = np.array(np.split(next_state, next_state.shape[1], axis=1)).squeeze(2).astype(np.float32)
        total_profit += profit
        if profit != 0:
            profit_histries = np.append(profit_histries, [[now, total_profit]], axis=0)

        if total_profit < min_limit_profit:
            done = True
        done = total_profit < min_limit_profit

        agent_model.remember(state, order_raw_value, reward, next_state, profit, done)
        agent_model.replay()

        draw_trader.update_graph(g_state, trader.current_order, profit_histries)


        
        