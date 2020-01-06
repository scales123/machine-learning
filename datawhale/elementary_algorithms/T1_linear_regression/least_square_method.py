# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(1000)  # 当我们设置相同的seed时，每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样
x = np.random.rand(500,3)  # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1
y = x.dot(np.array([5,6,7]))  # 构建映射关系，模拟真实的数据待预测值,映射关系为y = 5 + 6*x1 + 7*x2

class LR_LS():
    def __init__(self):
        self.w = None
        
