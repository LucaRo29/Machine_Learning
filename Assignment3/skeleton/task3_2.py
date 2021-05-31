import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import get_heart_dataset, get_toy_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle as pkl

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    # for i in range(10):
    #   n_estimators = 100
    #   parameters = {
    #     'max_depth': [2, 5, 10, 25,50, 100,200,None]
    #   }
    #   rf = RandomForestClassifier(n_estimators=n_estimators)
    #   clf = GridSearchCV(rf, parameters, n_jobs=-1)
    #   clf.fit(X_train, y_train)
    #   test_score = clf.score(X_test, y_test)
    #   print(f"Dataset: {clf.best_params_}")
    #   print(f'n_estimators: {n_estimators}')
    #   print("Test Score:", test_score)
    # # report your results
    rf = RandomForestClassifier(n_estimators=100, max_depth=None)
    rf.fit(X_train, y_train)
    print("Test Score:", rf.score(X_test, y_test))

    plt.bar(np.array(range(25)), rf.feature_importances_)
    plt.title('feature_importances')
    plt.savefig(f'images/feature_importances.png')
    plt.show()


    svc = SVC()

    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Best SVC params: {clf.best_params_}")
    print("Test Score:", test_score)



    rf = RandomForestClassifier()
    rfecv = RFECV(rf, scoring='accuracy')
    reducedX_train = rfecv.fit_transform(X_train, y_train)
    reducedX_test = rfecv.transform(X_test)


    svc = SVC()
    svc.fit(reducedX_train, y_train)
    print("Test Score:", svc.score(reducedX_test, y_test))
