# -*- coding: UTF-8 -*-

# 生成数据
import numpy as np
# 生成随机数
np.random.seed(1234)
x = np.random.rand(500, 3)
# 构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
y = x.dot(np.array([4.2, 5.7, 10.8]))

# 最小二乘法
class LR_LS():
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        # 最小二乘法矩阵求解
        # ============================= show me your code =======================
        self.w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        # ============================= show me your code =======================

    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        # ============================= show me your code =======================
        y_pred = X.dot(self.w)
        # ============================= show me your code =======================
        return y_pred


if __name__ == "__main__":
    lr_ls = LR_LS()
    lr_ls.fit(x, y)
    print("估计的参数值：%s" % (lr_ls.w))
    x_test = np.array([2, 4, 5]).reshape(1, -1)
    print("预测值为: %s" % (lr_ls.predict(x_test)))