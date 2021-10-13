"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.fixes import linspace

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# (Question 2)

def q1(seed, n):
    X, y = make_unbalanced_dataset(n_points, seed)

    X_l = X[:1000]
    X_t = X[1000:]
    y_l = y[:1000]
    y_t = y[1000:]

    knn = KNeighborsClassifier(n_neighbors=n).fit(X_l, y_l)
    y_p = knn.predict(X_t)
    return (accuracy_score(y_t, y_p), knn, X_l, y_l)

def q2 (seed, n):
    X, y = make_unbalanced_dataset(n_points, seed)
    knn_classifier = KNeighborsClassifier(n_neighbors = n)
    cross_scores = cross_val_score(knn_classifier, X, y, cv = 10)
    return np.mean(cross_scores)

def q3 (seed, n_neigh, n_fit):
    sum_accuracy = 0
    n_iter = 10
    for i in range(n_iter):
        X_f, y_f = make_unbalanced_dataset(n_fit, seed+i)
        knn_classifier = KNeighborsClassifier(n_neighbors = n_neigh).fit(X_f,y_f)
        X_t, y_t = make_unbalanced_dataset(test_set_size, seed+10+i)
        y_p = knn_classifier.predict(X_t)
        sum_accuracy = sum_accuracy + accuracy_score(y_t, y_p)

    return sum_accuracy/10


if __name__ == "__main__":

    n_points = 3000
    n_neighbors = [1, 5, 50, 100 ,500]
    acc_scores = np.zeros(len(n_neighbors))

    for i, n in enumerate(n_neighbors):
        acc_scores[i], knn_classifier, X_l, y_l = q1(50, n)
        plot_boundary('n_neighbors={}'.format(n_neighbors[i]), knn_classifier,X_l, y_l, mesh_step_size=0.1, title="k-NN with min_samples_split="+str(n))
    print("\n scores :\n \n", acc_scores)

    # ten-fold cross validation
   
    subset_scores = np.zeros(len(n_neighbors))
   
    for k, n in enumerate(n_neighbors):
        subset_scores[k] = q2(50, n)

    optimal_n = np.argmax(subset_scores)
    print("\n optimal indice :\n \n",optimal_n)
    print("\n scores :\n \n", subset_scores[optimal_n])

    #plot 3 a
    test_set_size = 500
    training_set_sizes = [50, 150, 250, 350, 450, 500]
    optimal_n = []

    for k, i in enumerate(training_set_sizes):
        subset_neighbors = [(j+1) for j in range(i)]
        results = np.zeros(len(subset_neighbors))

        for l, n in enumerate(subset_neighbors):
            results[l] = q3(40, n, i)

        plt.plot(subset_neighbors, results)
        optimal_n.append(np.argmax(results))
        print("\n Optimal n : ", np.argmax(results), "for training set size of : ", i)
        plt.savefig('accuracy_' + str(i) + '.pdf')

    plt.plot(training_set_sizes, optimal_n)
    plt.savefig('optimal_n.pdf')