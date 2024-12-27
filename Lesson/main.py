import random

import pygame

from entities import Player, Item
from model import DQNAgent


# メインループ
def game_loop(model):
    # ゲームオブジェクト
    player = Player(screen_height, lane_count, lane_width)
    items = []
    run = True
    counter = 0
    life = 0
    while run:
        action = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # アイテムの生成
        if max_items > len(items):
            if random.randint(1, 10) == 1:
                items.append(Item(lane_count, lane_width))

        state = get_state(player, items)  # 状態を取得
        action = model.action(state)  # AIによる行動選択
        reward, point = take_action(player, items, action)  # 行動の結果を反映
        life += point
        done = False
        # lifeがなくなれば終了
        if life < 0:
            done = True
        elif 10 < life:
            life = 10
        next_state = get_state(player, items)  # 次の状態を取得
        model.remember(state, action, reward, next_state, done)  # 経験を記憶

        model.replay()  # 経験を用いて学習

        # 描画
        screen.fill(white)
        player.draw(screen, red)
        for obs in items:
            obs.draw(screen, black)

        # テキストをレンダリング
        text = font.render(
            f"time{int(counter / 30)}::life{life}", True, black
        )  # 黒色で描画
        # テキストを左上に描画
        screen.blit(text, (10, 10))

        # 画面を更新
        pygame.display.update()

        clock.tick(30)

        # カウンターを更新
        counter += 1

        # プレイヤーのHPが0より下
        if done:
            player = Player(screen_height, lane_count, lane_width)
            items = []
            counter = 0
            life = 0

    pygame.quit()


# 行動する
def take_action(car, items, action):
    reward = 0
    if action == 0:
        car.move("left")
    elif action == 2:
        car.move("right")

    point = 0
    for obs in items:
        # アイテムの移動
        obs.move()

        # プレイヤーに当たった場合
        if (
            car.lane == obs.lane
            and car.y < obs.y + obs.height
            and car.height + car.y > obs.y
        ):
            items.remove(obs)
            point += 1
            reward = 10

        # アイテムの削除
        if obs.y > screen_height - 10:
            items.remove(obs)
            point -= 1
            reward = -100

    return reward, point


# 状態を取得する関数
def get_state(player, items):
    state = [float(player.lane)]  # player.laneをfloatにキャストしておく
    # 最新のアイテム max_items 個まで考慮
    for obs in items[:max_items]:  # 最新 max_items 個のアイテム
        state.extend([float(obs.lane), float(obs.y)])

    # アイテムが max_items より少ない場合、デフォルト値で埋める
    while len(state) < 2 * max_items + 1:  # プレイヤーのレーン + 2 * max_items
        state.extend([0.0, 0.0])  # デフォルト値で埋める

    return state


# Pygameの初期化
pygame.init()

# フォントの設定
font = pygame.font.Font(None, 36)  # デフォルトフォントをサイズ36で使用

# ゲームウィンドウのサイズ
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 色の定義
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# フレームレート調整用
clock = pygame.time.Clock()

# レーンの設定
lane_count = 11  # レーンの数
lane_width = screen_width // lane_count

# アイテムの最大数
max_items = 10

# エージェントの作成
state_size = 1 + (max_items * 2)  # 状態空間のサイズ
action_size = 3  # 行動の数（左、そのまま、右）
agent_model = DQNAgent(state_size, action_size)

# ゲームの開始
game_loop(agent_model)