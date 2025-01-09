# Description: このファイルには、共通の分析関数が含まれています。

import numpy as np

def trend_ratios(arr:np.array, threshold:float) -> tuple:
    """
    arrayが上昇傾向にあるか下降傾向にあるかを判断するメソッド   
    returns: (下降傾向、横這い、上昇傾向）の割合を表す
    """
    if len(arr) < 2:
        raise ValueError("配列には少なくとも2つ以上の要素が必要です。")

    changes = np.diff(arr)
    # 上昇、下降、横ばいのカウント
    uptrend_count = np.sum(changes > threshold)
    downtrend_count = np.sum(changes < -threshold)
    sideways_count = np.sum((changes >= -threshold) & (changes <= threshold))

    # 全体の数
    total_changes = len(changes)

    # 割合を計算
    uptrend_ratio = (uptrend_count / total_changes) * 100
    downtrend_ratio = (downtrend_count / total_changes) * 100
    sideways_ratio = (sideways_count / total_changes) * 100

    return downtrend_ratio, sideways_ratio, uptrend_ratio

# 現在の価格が上昇トレンドで、かつ最高値を更新している
def is_higher_value(values: np.array):
    if (len(values) < 5):
        raise ValueError("valuesは5つ以上の要素が必要です。")

    downtrend_ratio, sideways_ratio, uptrend_ratio = trend_ratios(values, 0.01)
    is_uptrend = uptrend_ratio > downtrend_ratio and uptrend_ratio > sideways_ratio
    is_higher_value = np.median(values) <  values[-1]
    return is_uptrend and is_higher_value

# 現在の価格が下降トレンドで、かつ最安値を更新している
def is_lower_value(values: np.array):
    if (len(values) < 5):
        raise ValueError("valuesは5つ以上の要素が必要です。")

    downtrend_ratio, sideways_ratio, uptrend_ratio = trend_ratios(values, 0.01)
    is_downtrend = downtrend_ratio > uptrend_ratio and downtrend_ratio > sideways_ratio
    is_lower_value = np.median(values) > values[-1]
    return is_downtrend and is_lower_value

def is_crossing(arr1, arr2):
    """
    2つのNumPy配列が線で結ばれたときにクロスしているかを判定する。

    Parameters:
        arr1 (np.ndarray): 1つ目の配列。
        arr2 (np.ndarray): 2つ目の配列。

    Returns:
        bool: クロスしている場合はTrue、していない場合はFalse。
    """
    if len(arr1) != len(arr2):
        raise ValueError("2つの配列は同じ長さでなければなりません。")
    
    for i in range(len(arr1) - 1):
        # 線分1の点 (x1, y1) と (x2, y2)
        x1, y1 = i, arr1[i]
        x2, y2 = i + 1, arr1[i + 1]
        
        # 線分2の点 (x3, y3) と (x4, y4)
        x3, y3 = i, arr2[i]
        x4, y4 = i + 1, arr2[i + 1]

        # 線分の交差を判定
        if do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            return True

    return False

def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    2つの線分が交差しているかを判定する。

    Parameters:
        x1, y1, x2, y2: 線分1の端点。
        x3, y3, x4, y4: 線分2の端点。

    Returns:
        bool: 線分が交差している場合はTrue、していない場合はFalse。
    """
    # ベクトルを用いた計算
    def ccw(xa, ya, xb, yb, xc, yc):
        return (yc - ya) * (xb - xa) > (yb - ya) * (xc - xa)

    # 線分1と線分2の交差判定
    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))