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

    n_points = 3000  
    n_neighbors = [1, 5, 50, 100 ,500] 
    scores = np.zeros((1, len(n_neighbors)))
    X, y = make_unbalanced_dataset(n_points, 50)

    X_l, X_t, y_l, y_t = train_test_split(X, y, test_size=2/3)  


    for i, n in enumerate(n_neighbors):
        knn_classifier = KNeighborsClassifier(n_neighbors = n).fit(X_l, y_l) 
        y_p = knn_classifier.predict(X_t) 
        scores[1, i] = accuracy_score(y_t, y_p)  
        plot_boundary('n_neighbors={}'.format(value), estimator,X_t, y_t, mesh_step_size=0.1, title="")
