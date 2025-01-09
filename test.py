import matplotlib.pyplot as plt
import numpy as np

# サンプルデータの作成
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# データがない期間を設定
mask = (x < 3) | (x > 7)
x = x[mask]
y = y[mask]

fig, ax = plt.subplots()
ax.plot(x, y)

# データがない期間を縮める
ax.set_xlim(0, 10)
ax.set_xticks([0, 3, 7, 10])
ax.set_xticklabels(['0', '3', '7', '10'])

plt.show()
