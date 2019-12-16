# -*-coding:utf-8 -*-

import numpy as np
import operator
from os import listdir  # 从os 模块导入listdir，可以给出指定目录文件名
from sklearn.neighbors import KNeighborsClassifier as kNN  # 载入sklearn库

'''
函数img2vector，将图像转化为向量，该函数创建1x1024的数组，然后打开给定的文件，循环读出文件的前32行，
并将每行的头32个字值存储在NumPy数组种，最后返回数组。
'''
def classify0(inX, dataSet, labels, k):
    # ①距离计算：已知类别数据集与当前点的距离（欧氏距离公式）
    dataSetSize = dataSet.shape[0]  # 读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]读取列数
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet  # tile(inX，(dataSetSize,1))复制比较向量inX,tile的功能是告诉inX需要复制多少遍，
    # 这里复制(dataSetSize行,1列)，目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，形成的插值矩阵放到diffMat里
    sqDiffMat = diffMat**2  # 注意这里是把矩阵中的每一个元素进行乘方
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  #开根号
    # 按照距离递增次序排序
    sortedDistIndicies = distances.argsort()  # 使用argsort进行排序，返回从小到大的顺序值，注意是顺序值！！
    # 如[2,4,1]返回[1,2,0],依次为其顺序的索引
    classCount = {}  # 新建一个字典，用于计数
    # ②选取与当前点距离最小的k个点
    for i in range(k):  # 按照顺序对标签进行计数
        voteIlabel = labels[sortedDistIndicies[i]]  # 按照之前的排序值，对标签依次进行计数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 对字典进行抓取，此时字典是空的
    # 所以没有标签，现在将一个标签作为key,value就是label出现的次数，因为从数组0开始，但计数从1开始，故需加1
    # ③排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回一个列表按照第二个元素进行降序排列
    return sortedClassCount[0][0]  # 返回出现次数最多到label值，即为当前的预测分类

def img2vector(filename):  # image to Vector
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)  # trainingDigits返回文件夹下的文件个数
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])  # get image_class_labels
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  # 将每一个文件的1x1024数据存储到trainingMat中
    testFileList = listdir('testDigits')        # 返回testDigits目录下的文件列表
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))

