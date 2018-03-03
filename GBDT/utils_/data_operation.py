from __future__ import division
import numpy as np
import math
import sys

def calculate_entropy(y):
    """Calculate the entropy of label array y"""
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def mean_squared_error(y_true, y_pred):
    """Return the mean squared error between y_true and y_pred"""
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def calculate_variance(X):
    """Return the variance of the features in dataset X"""
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance

def accuracy_score(y_true, y_pred):
    """Compare y_true and y_pred and return the accuracy"""
    accuracy = np.sum(y_true == y_pred, axis = 0) / len(y_true)
    return accuracy

