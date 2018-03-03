#special GBDT based on residual and minus gradient
import numpy as np 

def loadData():
    dataArray = [[1, 2], [3, 5], [2.4, 3.4], [7, 2], [8, 1], [6, 5], [4, 6]]
    dataLabel = [10, 2, 4, 2.3, 3, 5, 7]
    return dataArray, dataLabel

def classify(dataMatrix, dim, threVal, value1, value2):
    m, n = np.shape(dataMatrix)
    classSplit = np.zeros((m ,1))
    classSplit[dataMatrix[:, dim] < threVal] = value1
    classSplit[dataMatrix[:, dim] >= threVal] = value2
    return classSplit

def classify_select(dataArray, dataLabel):
    m, n = np.shape(dataArray)
    dataMatrix = np.mat(dataArray)
    labelMatrix = np.mat(dataLabel).T
    numStep = 20
    classStump = {}
    minLoss = 100000
    for i in range(n):
        max_ = dataMatrix[:, i].max()
        min_ = dataMatrix[:, i].min()
        step_len = (max_ - min_)/numStep
        for j in range(-1, int(step_len)+1):
            threVal = min_ + j * step_len
            value1 = labelMatrix[dataMatrix[:, i] < threVal]
            value1 = value1.sum()/len(value1)
            value2 = labelMatrix[dataMatrix[:, i] >= threVal]
            value2 = value2.sum()/len(value2)
            classSplit = classify(dataMatrix, i, threVal, value1, value2)
            sum = np.square(classSplit - labelMatrix).sum()
            if sum < minLoss:
                minLoss = sum
                classStump['dim'] = i
                classStump['thraval'] = threVal
                classStump['value1'] = value1
                classStump['value2'] = value2
                new_label = classSplit - labelMatrix
    return classStump, new_label

def generate_boost(dataArray, dataLabel, Iter = 40):
    classifyArray = []
    m, n = np.shape(dataArray)
    dataMatrix = np.mat(dataArray)
    left_label = dataLabel
    f = np.zeros((m, 1))
    for i in range(Iter):
        classStump, left_label = classify_select(dataArray, left_label)
        f += classify(dataMatrix, classStump['dim'], classStump['threval'],\
        classStump['value1'], classStump['value2'])
        classifyArray.append(classStump)
        loss = np.square(dataLabel - f).sum()
        if loss <= 0.1:
            break
    return classStump, f




    

        









            
