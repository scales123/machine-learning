# -*-coding:utf-8 -*-

import numpy as np
import operator
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from kNNclassify import createDataSet


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
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
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
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector  # Return the feature matrix and the class matrix

''' 
Data normalization formula: newValue = (oldValue - min)/(max - min)
'''
'''
在matplotlib中，整个图像为一个Figure对象。在Figure对象中可以包含一个或者多个Axes对象。每个Axes(ax)对象都是一个拥有自己坐标系统的绘图区域。所属关系如下：
def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,         
subplot_kw=None, gridspec_kw=None, **fig_kw):
参数：
nrows，ncols：子图的行列数。
sharex, sharey：
设置为 True 或者 ‘all’ 时，所有子图共享 x 轴或者 y 轴，
设置为 False or ‘none’ 时，所有子图的 x，y 轴均为独立，
设置为 ‘row’ 时，每一行的子图会共享 x 或者 y 轴，
设置为 ‘col’ 时，每一列的子图会共享 x 或者 y 轴。
返回值
fig： matplotlib.figure.Figure 对象
ax：子图对象（ matplotlib.axes.Axes）或者是他的数组
'''

def showdatas(datingDataMat, datingLabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

def autoNorm(dataSet):
    min_values = dataSet.min(0)  # Minimum value of each column
    print(min_values)
    max_values = dataSet.max(0)  # Maximum value of each column
    print(max_values)
    ranges = max_values - min_values
    m = dataSet.shape[0]  # m = 1000
    fenzi = dataSet - np.tile(min_values, (m, 1))  # oldValue - min
    normdataSet = fenzi/np.tile(ranges, (m, 1))  # newValue = (oldValue - min)/(max - min)
    return normdataSet, ranges, min_values

'''
classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], classLabelsVector[numTestVecs:m, 3])
上面这行也可以这么写：
print(normMat[i, :])    # 第i行的向量，i从0到99的整数，闭区间
print(normMat[numTestVecs:m, :])    # numTestVecs = 100，m = 1000，这个表示后900行的向量
print(datingLabels[numTestVecs:m])  # 所对应的标签
'''

def datingClassTest():
    hoRatio = 0.10  # #hold out 10%
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
    returnMat, classLabelVector = file2matrix('datingTestSet2.txt')
    normMat, ranges, min_values = autoNorm(returnMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - min_values)/ranges, normMat, classLabelVector, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

if __name__ == '__main__':
    group, labels = createDataSet()
    print(group,labels)
    classify_ = classify0([0, 0], group, labels, 3)
    print(classify_)

    returnMat, classLabelVector = file2matrix('datingTestSet2.txt')
    print(returnMat)
    print(classLabelVector[0:20])

    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)

    normdataSet, ranges, min_values = autoNorm(returnMat)
    print(normdataSet)
    print(ranges)
    print(min_values)

    datingClassTest()






