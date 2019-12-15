# -*-coding:utf-8 -*-
# classify函数的参数：
# inX：用于分类的输入向量
# dataSet：训练样本集合
# labels：标签向量
# k：K-近邻算法中的k
# shape：是array的属性，描述一个多维数组的维度
# tile（inX, (dataSetSize,1)）：把inX二维数组化，dataSetSize表示生成数组后的行数，1表示列的倍数。整个这一行代码表示前一个二维
# 数组矩阵的每一个元素减去后一个数组对应的元素值，这样就实现了矩阵之间的减法，简单方便得不让你佩服不行！
# axis=1：参数等于1的时候，表示矩阵中行之间的数的求和，等于0的时候表示列之间数的求和。
# argsort()：对一个数组进行非降序排序
# classCount.get(numOflabel,0) + 1：get()：该方法是访问字典项的方法，即访问下标键为numOflabel
# 的项，如果没有这一项，那么初始值为0。然后把这一项的值加1。所以Python中实现这样的操作就只需要一行代码，实在是很简洁高效。

import numpy as np
import operator

def createDataSet():  # 建立数据集
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# 在这个数据集里，group代表了其具体坐标位置，labels代表了其标签（属性）
# 下面基于该数据集建立了一个基本分类器，该分类器將通过坐标，对其可能的属性是‘A’还是‘B’进行预测

# 该函数为简单的knn分类器
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

if __name__ == '__main__':
    group, labels = createDataSet()
    print(group,labels)
    classify_ = classify0([0, 0], group, labels, 3)
    print(classify_)

