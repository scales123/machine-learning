# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
sklearn.linear_model参数详解：

fit_intercept : 默认为True,是否计算该模型的截距。如果使用中心化的数据，可以考虑设置为False,不考虑截距。
注意这里是考虑，一般还是要考虑截距

normalize: 默认为false. 当fit_intercept设置为false的时候，这个参数会被自动忽略。
如果为True,回归器会标准化输入参数：减去平均值，并且除以相应的二范数。
在这里还是建议将标准化的工作放在训练模型之前。通过设置sklearn.preprocessing.StandardScaler来实现，而在此处设置为false

copy_X : 默认为True, 否则X会被改写

n_jobs: int 默认为1. 当-1时默认使用全部CPUs ??(这个参数有待尝试)

可用属性：

coef_:训练后的输入端模型系数，如果label有两个，即y值有两列。那么是一个2D的array

intercept_: 截距

可用的methods:

fit(X,y,sample_weight=None): X: array, 稀疏矩阵 [n_samples,n_features] y: array [n_samples, n_targets] 
sample_weight: 权重 array [n_samples] 在版本0.17后添加了sample_weight

get_params(deep=True)： 返回对regressor 的设置值

predict(X): 预测 基于 R^2值

score： 评估

参考https://blog.csdn.net/weixin_39175124/article/details/79465558
"""
# 生成数据
np.random.seed(1000)  # 当我们设置相同的seed时，每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样
x = np.random.rand(500,3)  # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
y = x.dot(np.array([5,6,7]))  # 构建映射关系，模拟真实的数据待预测值,映射关系为y = 5 + 6*x1 + 7*x2

# 调用模型
lr = LinearRegression(fit_intercept=True)
# 训练模型
lr.fit(x,y)
print("估计的参数值为：%s" %(lr.coef_))
plt.show(lr.coef_)
# 计算R平方
print("R2：%s" %(lr.score(x,y)))
plt.show(lr.score())
# 任意设定变量，预测目标值
x_test = np.array([2,4,5]).reshape(1,-1)
y_hat = lr.predict(x_test)
print("预测值为：%s" %(y_hat))
plt.show(y_hat)

