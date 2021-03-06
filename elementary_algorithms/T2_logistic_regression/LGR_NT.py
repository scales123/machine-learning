# -*- coding: UTF-8 -*-
import pandas as pd
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


class LGR_NT():
    def __init__(self):
        self.w = None
        self.n_iters = None

    def fit(self, X, y, loss=1e-10):  # 判断是否收敛的条件为1e-10
        y = y.reshape(-1, 1)  # 重塑y值的维度以便矩阵运算
        [m, d] = np.shape(X)  # 自变量的维度
        self.w = np.zeros((1, d))  # 将参数的初始值定为0
        tol = 1e5
        n_iters = 0
        Hessian = np.zeros((d, d))
        # ============================= show me your code =======================
        while tol > loss:
            zs = X.dot(self.w.T)
            h_f = 1 / (1 + np.exp(-zs))
            grad = np.mean(X*(y - h_f), axis=0)
            # axis= 0 对**横轴操作**，在运算的过程中其运算的方向表现为**纵向运算**

            for i in range(d):
                for j in range(d):
                    if j >= i:
                        Hessian[i][j] = np.mean(h_f*(h_f-1)*X[:, i]*X[:, j])  # 更新海森矩阵中的值
                    else:
                        Hessian[i][j] = Hessian[j][i]  # 按海森矩阵的性质，对称点可直接得出结果
            theta = self.w - np.linalg.inv(Hessian).dot(grad)  # 迭代公式
            tol = np.sum(np.abs(theta - self.w))  # loss 平均绝对误差
            self.w = theta  # 更新迭代值
            n_iters += 1  # 更新迭代次数
        # ============================= show me your code =======================
        self.w = theta
        self.n_iters = n_iters

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == "__main__":
    LGR_NT = LGR_NT()
    LGR_NT.fit(Xs, ys)

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred

    print("牛顿法结果参数：%s;牛顿法迭代次数：%s" % (LGR_NT.w, LGR_NT.n_iters))
