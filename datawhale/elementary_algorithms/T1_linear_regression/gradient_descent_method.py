# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

# 生成数据
import numpy as np
# 生成随机数
np.random.seed(1234)
x = np.random.rand(500, 3)
# 构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
y = x.dot(np.array([4.2, 5.7, 10.8]))

# 梯度下降法
class LR_GD():
    def __init__(self):
        self.w = None

    def fit(self, X, y, alpha=0.02, loss=1e-10):  # 设定步长为0.02,判断是否收敛的条件为1e-10
        y = y.reshape(-1, 1)  # 重塑y值的维度以便矩阵运算
        [m, d] = np.shape(X)  # 自变量的维度 (m为样本数，d为属性个数)
        self.w = np.ones((d))  # 将参数的初始值定为0
        tol = 1e5
        # ============================= show me your code =======================
        while tol > loss:
            # here
            for i in range(m - 1):
                tol = np.square(y[i] - x[i].dot(np.array(self.w)))  # 损失值
                J_w = (y[i] - x[i].dot(np.array(self.w))) * x[i] / m  # 代价函数
            self.w = self.w + alpha * J_w  # 梯度下降迭代
        print('*********训练结束**********')
        # ============================= show me your code =======================

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred


if __name__ == "__main__":
    lr_gd = LR_GD()
    lr_gd.fit(x, y)
    print("估计的参数值为：%s" % (lr_gd.w))
    x_test = np.array([2, 4, 5]).reshape(1, -1)
    print("预测值为：%s" % (lr_gd.predict(x_test)))
