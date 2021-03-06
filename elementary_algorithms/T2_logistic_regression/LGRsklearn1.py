# -*- coding: UTF-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

df_X = pd.read_csv('./logistic_x.txt', sep='\ +', header=None, engine='python')  # 读取X值
ys = pd.read_csv('./logistic_y.txt', sep='\ +', header=None, engine='python')  # 读取y值
ys = ys.astype(int)  # 转换ys的数据类型为整型
df_X['label'] = ys[0].values  # 将X按照y值的结果一一打标签
ax = plt.axes()
# 在二维图中描绘X点所处位置，直观查看数据点的分布情况
df_X.query('label == 0').plot.scatter(x=0, y=1, ax=ax, color='blue')
df_X.query('label == 1').plot.scatter(x=0, y=1, ax=ax, color='red')
plt.show()