# -*-coding:utf-8 -*-

import numpy as np

'''
函数img2vector，将图像转化为向量，该函数创建1x1024的数组，然后打开给定的文件，循环读出文件的前32行，
并将每行的头32个字值存储在NumPy数组种，最后返回数组。
'''

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

