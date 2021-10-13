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

#je n'ai pas testé donc ca reste un code brouillon pour le moment

if __name__ == "__main__":

    n_points = 3000  
    n_neighbors = [1, 5, 50, 100 ,500] 
    acc_scores = np.zeros(len(n_neighbors))
    X, y = make_unbalanced_dataset(n_points, 50)

    X_l = X[:1000]
    X_t = X[1000:]
    y_l = y[:1000]
    y_t = y[1000:]

    for i, n in enumerate(n_neighbors):
        knn_classifier = KNeighborsClassifier(n_neighbors = n).fit(X_l, y_l) 
        y_p = knn_classifier.predict(X_t) 
        acc_scores[i] = accuracy_score(y_t, y_p)  # table of scores
        plot_boundary('n_neighbors={}'.format(n_neighbors[i]), knn_classifier,X_l, y_l, mesh_step_size=0.1, title="k-NN with min_samples_split="+str(n))
    print("\n scores :\n \n", acc_scores)

    # ten-fold cross validation
   
    subset_scores = np.zeros(100)

    #on teste pour un nombre de voisins qui va de 5 à 500
<<<<<<< Updated upstream
    for k, n in enumerate(subset_neighbors):
=======
   
    for k, n in enumerate(n_neighbors):
>>>>>>> Stashed changes
        knn_classifier = KNeighborsClassifier(n_neighbors = n)
        cross_scores = cross_val_score(knn_classifier, X, y, cv = 10)
        subset_scores[k] = np.mean(cross_scores)
    optimal_n = np.argmax(subset_scores)
    print("\n optimal indice :\n \n",optimal_n)
    print("\n scores :\n \n", subset_scores[optimal_n])

    #plot 3 a
    test_set_size = 500
    training_set_sizes = [50, 150, 250, 350, 450, 500]
    optimal_n = []
    fig, axs = plt.subplots(len(training_set_sizes))

    for k, i in enumerate(training_set_sizes):
        subset_neighbors = [(j+1) for j in range(i)]
        results = np.zeros(len(subset_neighbors))
        for l, n in enumerate(subset_neighbors):
            sum_accuracy = 0
            for j in range(10):
                X_f, y_f = make_unbalanced_dataset(i, 42+j)
                knn_classifier = KNeighborsClassifier(n_neighbors = n).fit(X_f,y_f)
                X_t, y_t = make_unbalanced_dataset(test_set_size, 69+j)
                y_p = knn_classifier.predict(X_t)
                sum_accuracy = sum_accuracy + accuracy_score(y_t, y_p)

            sum_accuracy = sum_accuracy/10
            results[l] = sum_accuracy
        # plot sur le subplot k results
        axs[k].plot(subset_neighbors, results) 
        # changer l'échelle sur les axes.
        optimal_n.append(np.argmax(results))
        print("\n Optimal n : ", np.argmax(results), "for training set size of : ", i)
    plt.show()
