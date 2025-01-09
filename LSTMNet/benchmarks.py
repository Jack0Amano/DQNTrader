import timeit

# 1000件程度の辞書を作成
dict1 = {i: i for i in range(1000)}
dict2 = {i: i * 2 for i in range(500, 1500)}

# 方法1: | 演算子
def method1():
    return dict1 | dict2

# 方法2: ** アンパッキング
def method2():
    return {**dict1, **dict2}

# 方法3: update メソッド
def method3():
    d = dict1.copy()
    d.update(dict2)
    return d

# 方法4: ChainMap
from collections import ChainMap
def method4():
    return dict(ChainMap(dict2, dict1))

# ベンチマーク実行
print("Method 1 (|):", timeit.timeit(method1, number=100000))
print("Method 2 (**):", timeit.timeit(method2, number=100000))
print("Method 3 (update):", timeit.timeit(method3, number=100000))
print("Method 4 (ChainMap):", timeit.timeit(method4, number=100000))