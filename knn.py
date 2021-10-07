"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 2)

#je n'ai pas testé donc ca reste un code brouillon pour le moment

if __name__ == "__main__":

    n_points = 3000  # We want 3000 samples
    min_samples = [2, 8, 32, 64, 128, 500]
    acc_scores = np.zeros(len(min_samples))
    X, y = make_unbalanced_dataset(n_points, 50)
    #crée 3000 points avec une valeur associée à chaque point
    #X matrice 3000x2 avec les coord de chaque point
    #y liste 3000 valeurs correspondant à chaque point

    X_l = X[:1000]
    X_t = X[1000:]
    y_l = y[:1000]
    y_t = y[1000:]

    for i, n in enumerate(n_neighbors):
        knn_classifier = KNeighborsClassifier(n_neighbors = n).fit(X_l, y_l) 
        y_p = knn_classifier.predict(X_t) 
        acc_scores[i] = accuracy_score(y_t, y_p)  
        plot_boundary('n_neighbors={}'.format(value), estimator,X_t, y_t, mesh_step_size=0.1, title="")
