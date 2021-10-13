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
from sklearn import tree

def dt_score(seed, mss):
    X, y = make_unbalanced_dataset(n_points, seed)

    X_l = X[:1000]
    X_t = X[1000:]
    y_l = y[:1000]
    y_t = y[1000:]

    DecisionTree = DecisionTreeClassifier(min_samples_split=mss).fit(X_l, y_l) #create the tree
    y_p = DecisionTree.predict(X_t) #test the tree
    return (accuracy_score(y_t, y_p), DecisionTree, X_t, y_t)  # table of scores

if __name__ == "__main__":

    n_points = 3000  # We want 3000 samples
    min_samples_split = [2, 8, 32, 64, 128, 500]
    acc_scores = np.zeros((len(min_samples_split)))
    
    
    #crée 3000 points avec une valeur associée à chaque point
    #X matrice 3000x2 avec les coord de chaque point
    #y liste 3000 valeurs correspondant à chaque point
    

    for i, mss in enumerate(min_samples_split):
        acc_scores[i], DecisionTree, X_t, y_t = dt_score(40, mss)  # table of scores
        plot_boundary('dt_min_samples'+str(i), DecisionTree,X_t, y_t, mesh_step_size=0.1, title="Decision Tree with min_samples_split="+str(mss))

    for j in range(5):
        for i, mss in enumerate(min_samples_split):
            acc_scores[i], DecisionTree, X_t, y_t = dt_score(40+j, mss)  # table of scores
            plot_boundary('seed_='+ str(40+j) + 'min_sample_split_=' + str(mss) + str(i), DecisionTree,X_t, y_t, mesh_step_size=0.1, title="Decision Tree with min_samples_split="+str(mss))