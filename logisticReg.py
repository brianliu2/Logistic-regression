from numpy import *
import numpy as np
# define the function to calculate sigmoid
def sigmoid(valX):
    return 1.0/(1+exp(-valX))

# define the function for estimating the weights of
# logistic regression, the modified stochastic gradient ascent
# is used here as optimization
def stoGradientAscent(dataMat, classLabels, numIter):
    rowNum, colNum = shape(dataMat)
    weights = ones(colNum)
    for i in range(numIter):
        dataIndex = list(range(rowNum))
        for m in range(rowNum):
            # alpha decreases with iteration, won't be zero as it is a constant
            alpha = 4 / (1.0 + m + i) + 0.0001
            randIndex = random.randint(0, colNum)
            sigmoidVal = sigmoid(sum(dataMat.ix[randIndex, :] * weights))
            err = classLabels[randIndex] - sigmoidVal
            weights = weights + alpha * err * dataMat.ix[randIndex]
            #del (dataIndex[randIndex])
        print(i)
    return weights

def classifyVector(testData, weights):
    testDataArr = np.array(testData)
    rowNumTest = shape(testData)[0]
    preTestLabels = []
    for n in range(rowNumTest):
        prob = sigmoid(sum(testDataArr[n, :] * weights))
        if prob > 0.5:
            curLabel = 1.0
        else:
            curLabel = 0.0
        preTestLabels.append(curLabel)
    return preTestLabels
