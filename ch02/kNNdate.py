# -*-coding:utf-8 -*-

import numpy as np

def filematrix(filename):  # file to numpy_matrix
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

if __name__ == '__main__':
    normdataSet, ranges, min_values = kNNdate.autonorm(datingdataSetMat)
