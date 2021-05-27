import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    for k in [1, 30, 100]:

        clf = KNearestNeighborsClassifier(k=k)
        clf.fit(X_train, y_train)
        clf.predict(X_test)

        scores = cross_val_score(clf, X_train, y_train, cv=5)
        meanscores = np.mean(scores)
        print(f"Mean score of cross validation: {meanscores}")

        test_score = clf.score(X_test, y_test)
        print(f"Test Score for k={k}: {test_score}")
        plt.figure()
        plotting.plot_decision_boundary(X_train, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.title(f'DecisionBoundariesK={k}')
        plt.savefig(f'images/task1_3DecisionBoundariesK={k}.png')
        plt.show()



    knn = KNearestNeighborsClassifier()
    parameters = {'k': [3, 5, 7, 9, 17, 25, 33, 49, 99]}
    clf = GridSearchCV(knn, parameters, return_train_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(clf.best_params_)
    print(clf.best_score_)
    Karray = [3, 5, 7, 9, 17, 25, 33, 49, 99]
    plt.plot(Karray, clf.cv_results_['mean_train_score'], label="mean_train_score",marker="o")
    plt.plot(Karray, clf.cv_results_['mean_test_score'], label="mean_test_score",marker="o")
    plt.legend()
    plt.xlabel('K')
    plt.savefig('images/task1_3result.png')
    plt.show()


