# -*- coding: UTF-8 -*-
import math

def func():
    for x1 in range(-4, 4):
        for x2 in range(-4, 4):
            y = x1 * math.exp(-x1 ** x1 - x2 ** x2)
    return y

def large(y):  # 定义一个large函数，函数的参数为可变参数
     ma = y[0]  # 初始化最大值
     for ma in y:
         if ma < y:
             ma = y
     return ma  # 返回最大值

if __name__ == '__main__':
    func()


