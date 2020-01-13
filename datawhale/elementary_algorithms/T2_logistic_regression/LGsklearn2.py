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
ys = df_X['label'].values
