import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import make_dataset
from utils import accuracy
import random

class Perceptron_Model:
    
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = self.activation_function
        self.w = None
        self.wo = None
        
    def train(self, X, y):
        n_examples, n_features = X.shape
        
        self.w = np.random.rand(n_features);
        self.wo = random.random()

        #Multiplied with loss to get misclassified examples
        t = np.array([1 if i > 0 else -1 for i in y])
        
        for _ in range(self.epochs):
            
            
            count = 0
            #Stochastic Gradient Descent
            for idx, x in enumerate(X):
            
                linear_result = np.dot(x, self.w) + self.wo
                
                #Get the prediction for the current example
                prediction = self.activation(linear_result)
                
                if(prediction == t[idx]):
                    count += 1

                # If the prediction is incorrect, update the weights
                if(prediction != y[idx]):
                    self.w += (self.lr * (t[idx]) * x)
                    self.wo += self.lr * (t[idx])
            
            if(count==X.shape[0]):
                break

                
                
    
    def predict(self, X):
        output = np.dot(X, self.w) + self.wo
        predictions = self.activation(output)
        return predictions
        
        
    def activation_function(self, x):
        return np.where(x>=0, 1, 0)


X_train, X_test, y_train, y_test = make_dataset(1)
print("Dataset - 1:")
perceptron = Perceptron_Model(learning_rate=0.01, epochs=5000)
perceptron.train(X_train, y_train)
predictions = perceptron.predict(X_train)
print("Train Set Accuracy", accuracy(y_train, predictions))

predictions = perceptron.predict(X_test)
print("Test Set Accuracy", accuracy(y_test, predictions))

X_train, X_test, y_train, y_test = make_dataset(2)

perceptron = Perceptron_Model(learning_rate=0.01, epochs=5000)
perceptron.train(X_train, y_train)
predictions = perceptron.predict(X_train)
print("\nDataset - 2:")
print("Train Set Accuracy", accuracy(y_train, predictions))
predictions = perceptron.predict(X_test)
print("Test Set Accuracy", accuracy(y_test, predictions))
