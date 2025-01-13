import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import HybridModel


# 優先度付き経験再生バッファの定義
class PrioritizedReplayBuffer:
    def __init__(self):
        self.memory = []  # 経験を保存するリスト

    # 経験をバッファに追加
    def add(self, classifier_x, box_x, trend_x, label):
        experience = (classifier_x, box_x, trend_x, label)
        self.memory.append(experience)

    def pop(self, batch_size):
        """
        メモリの先頭からバッチサイズ分の経験を取り出しメモリから消す   
        """
        classifier_x, box_x, trend_x, labels = zip(*self.memory)
        # 古い経験から優先してサンプリング
        classifier_x = classifier_x[:batch_size]
        box_x = box_x[:batch_size]
        trend_x = trend_x[:batch_size]
        labels = np.concatenate(labels[:batch_size], axis=0)
        # クラスインデックスに変換 損失関数がCrossEntropyLossの場合
        # labels = np.argmax(labels, axis=1)

        self.memory = self.memory[batch_size:]

        return classifier_x, box_x, trend_x, labels


    def __len__(self):
        return len(self.memory)
    
# DQNエージェントの定義
class RunTimeAgent:
    def __init__(
        self,
        classifier_input_dim,
        box_input_dim,
        trend_input_dim,
        label_sequence_length,
        batch_size=2048,
        learning_rate=0.001,
        
    ):
        self.memory = PrioritizedReplayBuffer()
        self.device = torch.device("cuda")
        self.model= HybridModel(classifier_input_dim, box_input_dim, trend_input_dim).to("cuda")
        # Adamオプティマイザを使用してネットワークのパラメータを最適化
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # modelのhidden_stateを保持するための変数
        self.batch_size = batch_size
        self.label_sequence_length = label_sequence_length
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.frame = 0

        self.classifier_hidden_state = None
        self.box_hidden_state = None
        self.trend_hidden_state = None

    # 経験をメモリに追加
    def remember(self, classifier_x, box_x, trend_x, label):
        self.memory.add(classifier_x, box_x, trend_x, label)

    # 行動を選択
    def action(self, classifier_x, box_x, trend_x) -> np.ndarray:
        """
        行動を選択するメソッド   
        return トレンド方向の[横這い・上昇・下降確率]   
        """
        self.model.eval()
        classifier_x = torch.from_numpy(classifier_x).unsqueeze(0).to(self.device)
        box_x = torch.from_numpy(box_x).unsqueeze(0).to(self.device)
        trend_x = torch.from_numpy(trend_x).unsqueeze(0).to(self.device)
        with torch.no_grad():  # 勾配計算を無効化
            model_out, logit, classifier_hc, box_hc, trend_hc = self.model(classifier_x, 
                                                                            box_x, 
                                                                            trend_x, 
                                                                            self.classifier_hidden_state, 
                                                                            self.box_hidden_state, 
                                                                            self.trend_hidden_state)
        (classifier_h, classifier_c) = classifier_hc.to("cup")
        (box_h, box_c) = box_hc.to("cup")
        (trend_h, trend_c) = trend_hc.to("cup")
        
        if classifier_c is not None:
            self.classifier_hidden_state = (classifier_h.detach(), classifier_c.detach())
        if box_c is not None:
            self.box_hidden_state = (box_h.detach(), box_c.detach())
        if trend_c is not None:
            self.trend_hidden_state = (trend_h.detach(), trend_c.detach())
        
        return model_out, logit
    
    def calculate_loss(self, model_logit: torch.Tensor, label: np.ndarray) -> torch.Tensor:
        """
        損失関数の計算   
        """
        tensor_label = torch.from_numpy(label).to(self.device)
        return self.criterion(model_logit, tensor_label)


    # 経験をリプレイしてネットワークを訓練
    def replay(self, train:bool) -> float:
        self.frame += 1
        if len(self.memory) < self.batch_size + self.label_sequence_length:
            return None # メモリが十分に溜まるまではリプレイを実行しない
        
        # メモリから優先度に基づいたサンプリングを実行
        classifier_x, box_x, trend_x, labels = self.memory.pop(self.batch_size)
        classifier_x = torch.from_numpy(np.array(classifier_x)).float().to(self.device)
        box_x = torch.from_numpy(np.array(box_x)).float().to(self.device)
        trend_x = torch.from_numpy(np.array(trend_x)).float().to(self.device)

        if train:
            # ネットワークを訓練モードに切り替え
            self.model.train()
            _, logit, classifier_hc, box_hc, trend_hc = self.model(classifier_x, 
                                                                    box_x, 
                                                                    trend_x,
                                                                    self.classifier_hidden_state, 
                                                                    self.box_hidden_state, 
                                                                    self.trend_hidden_state)
            self.__hold_hidden_state(classifier_hc, box_hc, trend_hc)
            loss = self.calculate_loss(logit, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        else:
            # ネットワークを評価モードに切り替え
            self.model.eval()
            with torch.no_grad():
                _, logit, classifier_hc, box_hc, trend_hc = self.model(classifier_x, 
                                                                        box_x, 
                                                                        trend_x, 
                                                                        self.classifier_hidden_state, 
                                                                        self.box_hidden_state, 
                                                                        self.trend_hidden_state)
            self.__hold_hidden_state(classifier_hc, box_hc, trend_hc)
            loss = self.calculate_loss(logit, labels)

        return loss.item()
    
    def __hold_hidden_state(self, classifier_hc, box_hc, trend_hc):
        """
        Modelから出力されるhidden_stateを保持するメソッド   
        """
        (classifier_h, classifier_c) = classifier_hc if classifier_hc is not None else (None, None)
        (box_h, box_c) = box_hc if box_hc is not None else (None, None)
        (trend_h, trend_c) = trend_hc if trend_hc is not None else (None, None)

        if classifier_c is not None:
            self.classifier_hidden_state = (classifier_h.to("cpu").detach(), classifier_c.to("cpu").detach())
        if box_c is not None:
            self.box_hidden_state = (box_h.to("cpu").detach(), box_c.to("cpu").detach())
        if trend_c is not None:
            self.trend_hidden_state = (trend_h.to("cpu").detach(), trend_c.to("cpu").detach())
        
    
    def clear_hidden_state(self):
        self.classifier_hidden_state = None
        self.box_hidden_state = None
        self.trend_hidden_state = None

    def clear_memory(self):
        self.memory = PrioritizedReplayBuffer()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

if __name__ == "__main__":

    agent = RunTimeAgent(box_input_dim=9, sequence_length=30)
    model = agent.model
    
    output, _ = model.test(9, 30, 1)
    print(output.shape)