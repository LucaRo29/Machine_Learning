import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        n_estimators = 1
        parameters = {
            'max_depth': [2, 5, 10, 25, 100,None]
        }
        rf = RandomForestClassifier(n_estimators=n_estimators)
        clf = GridSearchCV(rf, parameters, n_jobs=-1)
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"Dataset {idx}: {clf.best_params_}")
        print("Test Score:", test_score)
        plt.figure()
        plotting.plot_decision_boundary(X_train, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.title(f'RandomForestDataset{idx}&n_estimators= {n_estimators}')
        plt.savefig(f'images/RandomForestDataset{idx}&n_estimators= {n_estimators}.png')
