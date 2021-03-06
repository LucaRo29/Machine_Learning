import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':

    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)

        knn = KNearestNeighborsClassifier()

        parameters = {'k': [1, 3, 5, 9, 25, 49]}
        clf = GridSearchCV(knn, parameters, return_train_score=True)

        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"Test Score: {test_score}")
        print(f"Dataset {idx}: {clf.best_params_}")

        plt.figure()
        plotting.plot_decision_boundary(X_train, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.title(f'Dataset{idx}')
        plt.savefig(f'images/Dataset{idx}.png')

        plt.show()
        plt.figure()
        plt.plot(clf.cv_results_['mean_train_score'], label="mean_train_score")
        plt.plot(clf.cv_results_['mean_test_score'], label="mean_test_score")
        plt.title(f'Dataset{idx}')
        plt.legend()
        plt.savefig(f'images/Dataset{idx}_cvresults.png')
        plt.show()
