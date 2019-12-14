# -*-coding:utf-8 -*-

import numpy as np

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


if __name__ == '__main__':
    '''haha buqueindg '''
    datingDataSetMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingLabels)
