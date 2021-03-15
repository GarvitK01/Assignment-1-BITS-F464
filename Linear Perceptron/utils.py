import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle


def accuracy(y_true, y_preds):
    accuracy = np.sum(y_true==y_preds)/len(y_true)
    return accuracy

# Dataset 1
def make_dataset(number):

    if(number==1):
        filename = 'C:/Users/garvi/Desktop/BITS/ML/ML Assignment/ML_Assignment/Linear Perceptron/dataset_LP_' + str(number) + '.txt'
    if(number==2):
        filename = 'C:/Users/garvi/Desktop/BITS/ML/ML Assignment/ML_Assignment/Linear Perceptron/dataset_LP_' + str(number) + '.csv'
    dataset = pd.read_csv(filename, header = None)
    X = dataset.to_numpy()
    m, d = X.shape
    y = X[:, d-1]
    X = X[:, :d-1]

    indices = [i for i in range(len(y))]
    shuffle(indices)
    X = X[indices, :]
    y = y[indices]

    X_train = X[0: int(len(y) * 0.7), :]
    X_test = X[int(len(y) * 0.7):, :]

    y_train = y[0: int(len(y) * 0.7)]
    y_test = y[int(len(y) * 0.7):]

    return X_train, X_test, y_train, y_test