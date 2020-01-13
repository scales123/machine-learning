# -*- coding: UTF-8 -*-
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

df_X = pd.read_csv('./logistic_x.txt', sep='\ +',header=None, engine='python')  # 读取X值
ys = pd.read_csv('./logistic_y.txt', sep='\ +',header=None, engine='python')  # 读取y值
ys = ys.astype(int)
df_X['label'] = ys[0].values  # 将X按照y值的结果一一打标签
# 提取用于学习的数据
Xs = df_X[[0, 1]].values
Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
"""
【拼接数组的方法】
np.hstack——Stack arrays in sequence horizontally (column wise).   
np.vstack——Stack arrays in sequence vertically (row wise).
https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
"""
ys = df_X['label'].values

lr = LogisticRegression(fit_intercept=False)  # 因为前面已经将截距项的值合并到变量中，此处参数设置不需要截距项
lr.fit(Xs, ys)  # 拟合
score = lr.score(Xs, ys)  # 结果评价
print("Coefficient: %s" % lr.coef_)
print("Score: %s" % score)

ax = plt.axes()
df_X.query('label == 0').plot.scatter(x=0, y=1, ax=ax, color='blue')
df_X.query('label == 1').plot.scatter(x=0, y=1, ax=ax, color='red')

_xs = np.array([np.min(Xs[:, 1]), np.max(Xs[:, 1])])
# 将数据以二维图形式描点，并用学习得出的参数结果作为阈值，划分数据区域
_ys = (lr.coef_[0][0] + lr.coef_[0][1] * _xs) / (- lr.coef_[0][2])
plt.plot(_xs, _ys, lw=1)