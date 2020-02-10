# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_X = pd.read_csv('./logistic_x.txt', sep='\ +', header=None, engine='python')  # 读取X值
ys = pd.read_csv('./logistic_y.txt', sep='\ +', header=None, engine='python')  # 读取y值
ys = ys.astype(int)  # 转换ys的数据类型为整型
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

class LGR_GD():
    def __init__(self):
        self.w = None
        self.n_iters = None

    def fit(self, X, y, alpha=0.03, loss=1e-10):  # 设定步长为0.02，判断是否收敛的条件为1e-10
        y = y.reshape(-1, 1)  # 重塑y值的维度以便矩阵运算
        [m, d] = np.shape(X)  # 自变量的维度
        self.w = np.zeros((1, d))  # 将参数的初始值定为0
        tol = 1e5
        self.n_iters = 0
        # ============================= show me your code =======================
        while tol > loss:  # 设置收敛条件
            # 计算Sigmoid函数结果
            zs = X.dot(self.w.T)
            h_f = 1 / (1 + np.exp(-zs))

            theta = self.w + alpha * np.mean(X*(y - h_f), axis=0)  # 计算迭代的参数值
            # axis= 0 对**横轴操作**，在运算的过程中其运算的方向表现为**纵向运算**

            tol = np.sum(np.abs(theta - self.w))  # tol本身是一个差值？ 平均绝对误差

            self.w = theta  # 更新参数值
            self.n_iters += 1  # 更新迭代次数
        # ============================= show me your code =======================

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == "__main__":
    LGR_GD = LGR_GD()
    LGR_GD.fit(Xs, ys)

    ax = plt.axes()

    df_X.query('label == 0').plot.scatter(x=0, y=1, ax=ax, color='blue')
    df_X.query('label == 1').plot.scatter(x=0, y=1, ax=ax, color='red')

    _xs = np.array([np.min(Xs[:, 1]), np.max(Xs[:, 1])])
    _ys = (LGR_GD.w[0][0] + LGR_GD.w[0][1] * _xs) / (- LGR_GD.w[0][2])
    plt.plot(_xs, _ys, lw=1)
    plt.show()
    print("梯度下降法结果参数：%s;梯度下降法迭代次数：%s" % (LGR_GD.w, LGR_GD.n_iters))
