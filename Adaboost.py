import numpy as np

def loadSimpleData():
    datMat = matrix([[1. , 2.1], [2. ,1.1], [1.3, 1.], [1. , 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0]
    return dataMat, classLabels

def classify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones(shape(dataMatrix)[0], 1)
    if threshIneq = 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def dt_select(dataArray, classLabels, D):
    minError = 1000
    dataMatrix = np.mat(daraArray)
    labelMat = np.mat(classLabels).T
    m, n = shape(daraMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m , 1)))
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in {'lt', 'gt'}:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = classify(dataMatrix, i, threshVal, inequal)
                errArr = np.ones((m, 1))
                errArr[predictedVals == labelMat] = 0
                weightedError = np.dot(D, errArr)
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"\
                %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst =  predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D", D.T)
        alpha = float(0.5*log((1 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst", classEst.T)
        aggErrors = np.multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate =  aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0:
            break
    return weakClassArr

#predict
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = classify(dataMatrix, classifierArr['i']['dim'], classifierArr[i]['thresh'], classifierArr[i]['Ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)




















