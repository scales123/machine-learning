# -*- coding: UTF-8 -*-

# 生成数据
import numpy as np
# 生成随机数
np.random.seed(1234)
x = np.random.rand(500, 3)
# 构建映射关系，模拟真实的数据待预测值,映射关系为y = 4.2 + 5.7*x1 + 10.8*x2，可自行设置值进行尝试
y = x.dot(np.array([4.2, 5.7, 10.8]))

