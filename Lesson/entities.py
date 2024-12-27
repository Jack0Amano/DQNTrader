import random
import pygame

# プレイヤー
class Player:
    def __init__(self, screen_height, lane_count, lane_width):
        self.width = 50
        self.height = 70
        self.lane = lane_count // 2  # 初期にいるレーンは中央
        self.y = screen_height - self.height - 10
        self.speed = 5
        self.lane_count = lane_count
        self.lane_width = lane_width

    def draw(self, screen, color):
        """
        描画
        :param screen: 描画先
        :param color: 色
        :return:
        """
        x = self.lane * self.lane_width + (self.lane_width - self.width) // 2
        pygame.draw.rect(screen, color, (x, self.y, self.width, self.height))

    # 移動
    def move(self, direction):
        """
        車を移動させる
        :param direction: 移動方向
        :return: None
        """
        if direction == "left" and self.lane > 0:
            self.lane -= 1
        elif direction == "right" and self.lane < self.lane_count - 1:
            self.lane += 1

# アイテム
class Item:
    def __init__(self, lane_count, lane_width):
        self.width = 50
        self.height = 70
        self.lane = random.randint(0, lane_count - 1)
        self.y = -self.height
        self.speed = 5
        self.lane_count = lane_count
        self.lane_width = lane_width

    def draw(self, screen, color):
        """
        描画
        :param screen: 描画先
        :param color: 色
        :return:
        """
        x = self.lane * self.lane_width + (self.lane_width - self.width) // 2
        pygame.draw.rect(screen, color, (x, self.y, self.width, self.height))

    def move(self):
        """
        アイテムを移動させる
        :return: None
        """
        self.y += self.speed