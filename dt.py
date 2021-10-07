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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


if __name__ == "__main__":

    n_points = 3000  # We want 3000 samples
    min_samples = [2, 8, 32, 64, 128, 500]
    acc_scores = np.zeros(len(min_samples))
    gen_scores = np.zeros(5,6) #for the point 3 where we take the mean of 5 generated datasets
    
    for j in range(5)
        X, y = make_unbalanced_dataset(n_points, 50)
        #crée 3000 points avec une valeur associée à chaque point
        #X matrice 3000x2 avec les coord de chaque point
        #y liste 3000 valeurs correspondant à chaque point

        X_l = X[:1000]
        X_t = X[1000:]
        y_l = y[:1000]
        y_t = y[1000:]
    
        # 1000 samples for the training, 2000 for the test
        #X_l matrice 1000x2 des points qui ont servi à l'entrainement
        #X_t matrice 1000x2 des points qui ont servi au test
        #y_l liste 1000 des valeurs de points d'entrainement
        #y_t liste 2000 des valeurs de points de test

        for i, mss in enumerate(min_samples):
            DecisionTree = DecisionTreeClassifier(min_samples_split=mss).fit(X_l, y_l) #create the tree
            y_p = DecisionTree.predict(X_t) #test the tree
            acc_scores[i] = accuracy_score(y_t, y_p)  # table of scores
            plot_boundary('dt_min_samples={}'.format(value), estimator,X_t, y_t, mesh_step_size=0.1, title="")
        gen_scores[j,:] = acc_scores
    gen_mean = np.mean(gen_scores,0)
    gen_std = np.std(gen_scores,0)
        
