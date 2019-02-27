# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:39:31 2019

@author: Erwan
"""

import random
import matplotlib.pyplot as plt
from numpy import dot


class LIBSVMFile:
    
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        
    def load_data(self):
        f = open(self.filename)
        
        self.data = {'labels': [], 'values': []}
        
        for line in f.readlines():
            if line:
                split_line = line.split()
                label = float(split_line[0])
                values = [float(pair.split(':')[1]) for pair in split_line[1:]]
                
                self.data['labels'].append(label)
                self.data['values'].append(values)
                
        f.close()
        
        return self
            
    def save_data(self, data=None):
        """
        Save data in the LIBSVM format.
        
        Args:
            data: list of elements, each element being a list of values,
            preceded by a label (which must be an integer for classification
            tasks, or any real number for regression tasks).
            
            For example,
            
            data = [(1, [12, 62, -3]),
                    (2, [6, 41, 0]),
                    (2, [8, 22, 2]),
                    (1, [15, 30, -6])]
        """
        self.data = data
        
        f = open(self.filename, 'w')
        
        for label, values in zip(data['labels'], data['values']):
            f.write(str(label))
            for i, value in enumerate(values):
                f.write(' ')
                f.write(str(i))
                f.write(':')
                f.write(str(value))
            f.write('\n')
            
        f.close()
        
        return self
        
class LinearClassifier:
    
    def __init__(self, alpha=1, max_steps=1000, err_crit=2):
        if callable(alpha):
            # If alpha is a function
            self.learning_rate = alpha
        else:
            # Otherwise a constant function is created
            self.learning_rate = lambda t: alpha
        self.max_steps = max_steps  # Maximum number of iterations before stopping
        self.err_crit = err_crit    # Error threshold: when the number of misclassified
                                    # examples passes below, the training stops
        
        self.weights = None
        self.data = None
        
        
    def fit(self, data):
        self.data = data
        
        # Labels list
        Y = self.data['labels']
        
        # Values list, with extra 1 to account for the intercept term
        X = [[1.0] + values for values in self.data['values']]
        
        # Number of examples
        n = len(Y)
        
        # Initialization at 1
        self.weights = [1] * len(X[0])
        
        step = 0
        m = n
        
        while step < self.max_steps and m > self.err_crit:
            r = random.randrange(n)
            y, x = Y[r], X[r]
            
            for i, w in enumerate(self.weights):
                w += self.learning_rate(step) * (y - self._classifier(x)) * x[i]
                self.weights[i] = w
            
            # Number of missclassified examples
            
            m = sum([abs(true_y - pred_y) for true_y, pred_y in zip(Y, self.predict(X))])
            step += 1
            
    def predict(self, X):
        return [self._classifier(x) for x in X]
                
    def _classifier(self, x):
        return self._threshold(dot(self.weights, x))
        
    def _threshold(self, a):
        if a > 0:
            return 1
        else:
            return 0
        
if __name__ == '__main__':
    data = LIBSVMFile('data.libsvm').load_data().data
    X1 = [e[0] for e in data['values']]
    X2 = [e[1] for e in data['values']]
    Y = data['labels']
    
    clf = LinearClassifier(alpha=0.0000001)
    clf.fit(data)
    w = clf.weights
    
    Xg = range(10000, 80000)
    Yg = [(-w[0] - w[1]*x) / w[2] for x in Xg]
    
    plt.scatter(X1, X2, c=Y)
    plt.plot(Xg, Yg)
    plt.show()

    
        