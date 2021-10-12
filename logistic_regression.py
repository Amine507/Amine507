"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")


        self.theta_t = np.ones(X.shape[1] + 1)
        
        for i in range(self.n_iter) :
            x_theta_prod = np.matmul(X, self.theta_t[1:]) - self.theta_t[0]
            prob_know_x_thet = np.array([sigmoid(xi) for xi in x_theta_prod]) - y
            x_prime = np.c_[np.ones(X.shape[0]),X]
            p_mult_x_prime = np.multiply(np.reshape(prob_know_x_thet, [X.shape[0], 1]), x_prime)
            grad_theta = np.mean(p_mult_x_prime, axis=0)
            self.theta_t -= grad_theta
        

        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        
        x_theta_prod = np.matmul(X, self.theta_t[1:]) - self.theta_t[0]
        prob_know_x_thet = np.array([sigmoid(xi) for xi in x_theta_prod])
        predict = lambda p : 1 if (p > 0.5) else 0
        
        
        return np.array([predict(pi) for pi in prob_know_x_thet])

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        
        x_theta_prod = np.matmul(X, self.theta_t[1:]) - self.theta_t[0]
        one_prob_know_x_thet = np.array([sigmoid(xi) for xi in x_theta_prod])
        zero_prob_know_x_thet = 1 - one_prob_know_x_thet
        
        return np.c_[one_prob_know_x_thet, zero_prob_know_x_thet]

if __name__ == "__main__":
    pass
