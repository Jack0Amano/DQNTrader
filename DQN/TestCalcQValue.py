import enum

import pandas as pd
import numpy as np

class OrderType(enum.Enum):
    NOTHING = 0
    BUY = 1
    SELL = 2
    TAKE_PROFIT = 3

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
    

class TradingEnvironment:
    def __init__(self, spread):
        self.current_order: Order = None  # 現在のポジション: {"type": "Buy" or "Sell", "price": float}
        self.spread = spread  # スプレッド（取引コスト）

    def __calculate_feature_profits(self, action, target_price, future_prices) -> list:
        """
        target_priceでの取引を行った場合の将来の利益を計算   
        action: str, "Buy" or "Sell"
        target_price: float, 未来の価格を予測するための基準価格 (ここでポジションを設定する)
        future_prices: list[float], 未来の価格のリスト
        """
        if action == OrderType.BUY:
            # Buy の場合の将来の利益
            future_profits = []
            for fp in future_prices:
                future_profit = (fp - target_price) - self.spread
                future_profits.append(future_profit)
            return future_profits
        elif action == OrderType.SELL:
            # Sell の場合の将来の利益
            future_profits = []
            for fp in future_prices:
                future_profit = (target_price - fp) - self.spread
                future_profits.append(future_profit)
            return future_profits
    
    def calculate_reward(self, action, current_price, future_prices):
        """アクションに応じた報酬を計算"""
        rewards = []
        if action == OrderType.BUY:
            if not self.current_order:
                feature_profits = self.__calculate_feature_profits(OrderType.BUY, current_price, future_prices)
                if max(feature_profits) > 0:
                    # 将来の価格変動から利益が出せる場合
                    rewards = feature_profits
                else:
                    rewards.append(-1)
            else:
                rewards.append(0)
        elif action == OrderType.SELL:
            if not self.current_order:
                feature_profits = self.__calculate_feature_profits(OrderType.SELL, current_price, future_prices)
                if max(feature_profits) > 0:
                    rewards = feature_profits
                else:
                    rewards.append(-1)
            else:
                rewards.append(0)
        elif action == OrderType.TAKE_PROFIT:
            if self.current_order:
                if self.current_order.order_type == OrderType.BUY:
                    reward = (current_price - self.current_order["price"]) - self.spread
                elif self.current_order.order_type == OrderType.SELL:
                    reward = (self.current_order.value - current_price) - self.spread
                rewards.append(reward)
            else:
                rewards.append(0)  # ポジションがない場合の報酬は 0
        elif action == OrderType.NOTHING:
            if self.current_order is None:
                # ポジションがない場合、将来の価格変動から利益が出せるかチェック
                buy_profitable = any((fp - current_price) - self.spread > 0 for fp in future_prices)
                sell_profitable = any((current_price - fp) - self.spread > 0 for fp in future_prices)
                if buy_profitable or sell_profitable:
                    # 利益を出せる可能性があるため、機会損失としてマイナス報酬
                    # 最大利益の期待値をマイナスとして設定
                    possible_rewards = []
                    for fp in future_prices:
                        possible_buy = (fp - current_price) - self.spread
                        possible_sell = (current_price - fp) - self.spread
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
                buy_profit = max(self.__calculate_feature_profit(OrderType.BUY, self.current_order.value, future_prices))
                sell_profit = max(self.__calculate_feature_profit(OrderType.SELL, self.current_order.value, future_prices))
                max_future_profit = max([buy_profit, sell_profit], default=0)
                
                if current_profit > max_future_profit:
                    # 現在が最大の利益の場合、現在利確すべきであるためNothingの報酬は -current_profit
                    rewards.append(-current_profit)
                else:
                    # 現在の利益より期待される未来の利益が大きい場合、ポジションを保持するためにNothingの報酬はmax_future_profit
                    rewards.append(max_future_profit)
        return rewards

    def calculate_q_values(self, current_price, future_prices):
        """各アクションのQ値を計算"""
        actions = [OrderType.NOTHING, OrderType.BUY, OrderType.SELL, OrderType.TAKE_PROFIT]
        q_values = {}
        for action in actions:
            if action == OrderType.TAKE_PROFIT and not self.current_order:
                # ポジションがない場合、TakeProfitは無効
                q_values[action] = -np.inf  # 無効な選択肢として極小値を設定
            elif (action == OrderType.BUY or action == OrderType.SELL) and self.current_order:
                # ポジションがある場合、Buy と Sell は無効
                q_values[action] = -np.inf
            else:
                rewards = self.calculate_reward(action, current_price, future_prices)
                q_values[action] = np.mean(rewards)  # 平均で期待値を計算
        return q_values


# サンプルデータ
price_data = [100.3, 100.2, 100.4, 100.5, 101.4, 101.7, 101.6]
n = 6  # 未来を見るステップ数

# 環境を初期化してラベルを生成
env = TradingEnvironment(20*0.01)
#env.position = {"type": "Buy", "price": 103}
q_values = env.calculate_q_values(100, price_data)
print(q_values)