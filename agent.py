import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import HybridModel
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 優先度付き経験再生バッファの定義
class PrioritizedReplayBuffer:
    def __init__(self):
        self.cursor_index = 0
        self.classifier_memory = []
        self.box_memory = []
        self.trend_memory = []
        self.label_memory = []
        self.time_memory = []

    # 経験をバッファに追加
    def add(self, classifier_x, box_x, trend_x, label, time):
        """
        経験をメモリに追加する
        """
        self.classifier_memory.append(classifier_x)
        self.box_memory.append(box_x)
        self.trend_memory.append(trend_x)
        self.label_memory.append(label)
        self.time_memory.append(time)
        
        # if self.classifier_memory is None:
        #     self.classifier_memory = [classifier_x]
        #     self.box_memory = [box_x]
        #     self.trend_memory = [trend_x]
        #     self.label_memory = np.array([label])
        # else:
        #     self.classifier_memory = np.append(self.classifier_memory, np.array([classifier_x]), axis=0)
        #     self.box_memory = np.append(self.box_memory, np.array([box_x]), axis=0)
        #     self.trend_memory = np.append(self.trend_memory, np.array([trend_x]), axis=0)
        #     self.label_memory = np.append(self.label_memory, np.array([label]), axis=0)


    def get(self, batch_size):
        """
        メモリの先頭からバッチサイズ分の経験を取り出しカーソルを動かす
        """
        # 古い経験から優先してサンプリング
        classifier_x = np.array(self.classifier_memory[self.cursor_index:batch_size+self.cursor_index])
        box_x = np.array(self.box_memory[self.cursor_index:batch_size+self.cursor_index])
        trend_x = np.array(self.trend_memory[self.cursor_index:batch_size+self.cursor_index])
        labels = self.label_memory[self.cursor_index:batch_size+self.cursor_index]
        labels = np.concatenate(labels, axis=0)
        times = self.time_memory[self.cursor_index:batch_size+self.cursor_index]
        # クラスインデックスに変換 損失関数がCrossEntropyLossの場合
        # labels = np.argmax(labels, axis=1)

        # print("Cursor", self.cursor_index, "BatchSize", batch_size, "labels", labels.shape)

        self.cursor_index += batch_size
        
        return classifier_x, box_x, trend_x, labels, times
    
    def clear(self):    
        """
        メモリーに保存した経験を全て消去する
        """
        self.cursor_index = 0
        self.classifier_memory = []
        self.box_memory = []
        self.trend_memory = []
        self.label_memory = []
        self.time_memory = []

    def clear_cursor(self):
        """
        カーソルを初期化する
        """
        self.cursor_index = 0

    def is_gettable(self, batch_size):
        """
        メモリからバッチサイズ分の経験を取り出せるか判定する
        """
        return self.cursor_index + batch_size < len(self.label_memory)
    
    def get_memory_gbytes(self):
        """
        メモリの使用量を取得する (GB)
        """
        if len(self.label_memory) == 0:
            return 0
        classifier_bytes = self.classifier_memory[0].nbytes * len(self.classifier_memory)
        box_bytes = self.box_memory[0].nbytes * len(self.box_memory)
        trend_bytes = self.trend_memory[0].nbytes * len(self.trend_memory)
        label_bytes = self.label_memory[0].nbytes * len(self.label_memory)
        bytes = classifier_bytes + box_bytes + trend_bytes + label_bytes
        return bytes / 1024 / 1024 / 1024
    
    def get_memory(self):
        """
        メモリをnumpy配列に変換して返す
        """
        self.classifier_memory = np.array(self.classifier_memory)
        self.box_memory = np.array(self.box_memory)
        self.trend_memory = np.array(self.trend_memory)
        self.label_memory = np.array(self.label_memory)
        self.time_memory = np.array(self.time_memory)
        output_dic = {"classifier": self.classifier_memory, "box": self.box_memory, "trend": self.trend_memory, "label": self.label_memory, "time": self.time_memory}
        return output_dic
    
    def set_memory(self, memory_dic):
        """
        numpy配列をメモリにセットする
        """
        self.cursor_index = 0
        self.classifier_memory = memory_dic["classifier"]
        self.box_memory = memory_dic["box"]
        self.trend_memory = memory_dic["trend"]
        self.label_memory = memory_dic["label"]
        self.time_memory = memory_dic["time"]

    
# DQNエージェントの定義
class RunTimeAgent:
    def __init__(
        self,
        classifier_input_dim,
        box_input_dim,
        trend_input_dim,
        label_sequence_length,
        batch_size=2048,
        learning_rate=0.005,
        
    ):
        self.memory = PrioritizedReplayBuffer()
        self.device = torch.device("cuda")
        self.model= HybridModel(classifier_input_dim, box_input_dim, trend_input_dim).to("cuda")
        # Adamオプティマイザを使用してネットワークのパラメータを最適化
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5, verbose=True)
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
    def remember(self, classifier_x, box_x, trend_x, label, time):
        self.memory.add(classifier_x, box_x, trend_x, label, time)

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
        if not self.memory.is_gettable(self.batch_size + self.label_sequence_length):
            return None, None # メモリが十分に溜まるまではリプレイを実行しない
        
        # メモリから優先度に基づいたサンプリングを実行
        classifier_x, box_x, trend_x, labels, times = self.memory.get(self.batch_size)
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

        return loss.item(), times
    
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
        self.memory.clear()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def step_scheduler(self, loss):
        """
        learning_rateを可変するスケジューラーを1ステップ進める   
        """
        self.scheduler.step(loss)

    def get_learning_rate(self) -> float:
        """
        スケジューラーによって設定された学習率を取得するメソッド   
        return: 学習率 float
        """
        return self.scheduler.get_last_lr()[0]

if __name__ == "__main__":

    agent = RunTimeAgent(box_input_dim=9, sequence_length=30)
    model = agent.model
    
    output, _ = model.test(9, 30, 1)
    print(output.shape)