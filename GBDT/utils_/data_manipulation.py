from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def batch_iterator(X, y=None, batch_size=64):
    """Simple batch generator"""
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]
    
def divide_on_feature(X, features_i, threshold):
    """Divide dataset based on if sample value on feature index is larger than
       the given threshold"""
    
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float): #regression
        split_func = lambda sample: sample[features_i] >= threshold
    else:
        split_func = lambda sample: sample[features_i] == threshold #classification
    
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])

def train_test_split(X, y, test_size = 0.5, shuffle = True, seed = None):
    """Split the data into train and test sets"""
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    
    # Split the training data from test data in the ratio specified in
    # test size

    split_i = len(y) - int(len(y) * 0.5)
    X_train, X_test = X[:split_i], X[:split_i]
    y_train, y_test = y[:split_i], y[:split_i]

    return X_train, X_test, y_train, y_test

def normalize(X, axis = -1, order = 2):
    """Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X):
    """Standardize the dataset X"""
    X_std = X
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    
    return X_std

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values"""
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot



