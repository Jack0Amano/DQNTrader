# DQN Trader

概要
----
Deep Q Networkでのゲーム（FXでの通貨トレーディング）の自動化EA   
過去に行ったLSTMを利用した未来予測では15分以上では予測精度が6割など実用に耐えない物だったので、ポジション設定や損切などの全体をゲーム方式で行うことで実用可能なEAの作成を目指す。   
DQNにLSTMを組み込んだDRQN等を実装し時系列上での学習を目指す。   

参考
----
[PyTorchで深層強化学習（DQN、DoubleDQN）を実装してみた](https://ie110704.net/2017/10/15/pytorch%E3%81%A7%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%EF%BC%88dqn%E3%80%81doubledqn%EF%BC%89%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F/#toc4)   
[【強化学習】2018年度最強と噂のR2D2を実装/解説してみた](https://qiita.com/pocokhc/items/3b64d747a2f36da559c3)   
[PyTorchでDQNをやってみた](https://zenn.dev/viceinc/articles/e78fee3a0c73e1)   
[考察や学習ノート](WIKI/THINKING.md)


LICENCE
----
This project is licensed under the MIT License, see the LICENSE.txt file for details