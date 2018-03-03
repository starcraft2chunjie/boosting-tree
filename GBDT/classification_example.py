from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# Import helper functions
from utils_.data_manipulation import *
from utils_.data_operation import *
from utils_.loss_function import *
from gbdt_model import GBDTClassifier

def main():

    print ("-- Gradient Boosting Classification --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(y_train)

    clf = GBDTClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)



if __name__ == "__main__":
    main()
