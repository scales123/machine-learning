# -*-coding:utf-8 -*-

import numpy as np
import operator

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

def file2matrix(filename):  # file to numpy_matrix
    fr = open(filename)  # open file and save to fr
    arrayOLines = fr.readlines()  # read by line and save to arrayOLines
    numberOfLines = len(arrayOLines)  # get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 3))  # build a zero matrix (numberOfLines, 3)
    classLabelVector = []  # Create a single-column matrix and store its classes
    index = 0  # Index value reset
    for line in arrayOLines:
        line = line.strip()  # delete all 'enter', get 'one row' of data
        listFromLine = line.split('\t')  # use 'tab' spilt one row of data to an element list
        returnMat[index, :] = listFromLine[0:3]  # Assign the first three elements of each row in turn to returnMat
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector  # Return the feature matrix and the class matrix

''' 
Data normalization formula: newValue = (oldValue - min)/(max - min)
'''

def autoNorm(dataSet):
    min_values = dataSet.min(0)  # Minimum value of each column
    print(min_values)
    max_values = dataSet.max(0)  # Maximum value of each column
    print(max_values)
    ranges = max_values - min_values
    normdataSet = np.zeros(np.shape(dataSet))  # matrix: (1000,3)
    m = dataSet.shape[0]  # m = 1000
    normdataSet = dataSet - np.tile(min_values, (m, 1))  # oldValue - min
    normdataSet = normdataSet/np.tile(ranges, (m, 1))  # newValue = (oldValue - min)/(max - min)
    return normdataSet, ranges, min_values

'''
classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], classLabelsVector[numTestVecs:m, 3])
上面这行也可以这么写：
print(normMat[i, :])    # 第i行的向量，i从0到99的整数，闭区间
print(normMat[numTestVecs:m, :])    # numTestVecs = 100，m = 1000，这个表示后900行的向量
print(datingLabels[numTestVecs:m])  # 所对应的标签
'''

def datingClassTest():
    hoRatio = 0.10  # Extract 10% from dataset
    returnMat, classLabelsVector = file2matrix('datingTestSet.txt')
    normMat, ranges, min_values = autoNorm(returnMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :],classLabelsVector[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"\
            % (classifierResult, classLabelsVector[i]))
        if (classifierResult != classLabelsVector[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("pencentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    returnMat, classLabelsVector = file2matrix('datingTestSet2.txt')
    normMat, ranges, min_values = autoNorm(returnMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - min_values)/(ranges, normMat, classLabelsVector, 3))
    print("You will probably like this person: ", resultList[classifierResult - 1])


