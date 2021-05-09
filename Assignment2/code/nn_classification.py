import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    pca = PCA(svd_solver='randomized', whiten=True, n_components=n_components)

    X_reduced = pca.fit_transform(features)

    # PCA(n_components)

    explained_var = np.cumsum(pca.explained_variance_ratio_)

    # print(np.sum(pca.explained_variance_ratio_))
    print(f'Explained variance: {explained_var}')
    return X_reduced


def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [10, 100, 200]  # TODO create a list #Train accuracy: 0.9054. Test accuracy: 0.7724 Loss: 0.3208
    # Train accuracy: 1.0000. Test accuracy: 0.8329 Loss: 0.0083
    # Train accuracy: 1.0000. Test accuracy: 0.8402 Loss: 0.0065
    # 1000 Train accuracy: 1.0000. Test accuracy: 0.8741 Loss: 0.0039

    classifier = MLPClassifier(solver='adam', random_state=1, max_iter=500, hidden_layer_sizes=n_hidden_neurons).fit(
        X_train, y_train)
    # Set parameters (some of them are specified in the HW2 sheet).

    train_acc = classifier.score(X_train, y_train)

    test_acc = classifier.score(X_test, y_test)
    loss = classifier.loss_
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')


def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [10, 100, 200]
    # Train accuracy: 1.0000. Test accuracy: 0.8862 Loss: 0.2817
    # Train accuracy: 0.9751. Test accuracy: 0.8378 Loss: 0.0218
    # Train accuracy: 0.9830. Test accuracy: 0.8620 Loss: 0.3784

    classifier = MLPClassifier(solver='adam', random_state=1, max_iter=500, hidden_layer_sizes=n_hidden_neurons,
                               alpha=1.0, early_stopping=True).fit(X_train, y_train)

    train_acc = classifier.score(X_train, y_train)

    test_acc = classifier.score(X_test, y_test)
    loss = classifier.loss_

    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')


def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [1, 2, 8, 32, 64]  # TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    n_hidden_neurons = [10, 100, 200]

    for i in range(len(seeds)):
        classifier = MLPClassifier(solver='adam', random_state=seeds[i], max_iter=500,
                                   hidden_layer_sizes=n_hidden_neurons,
                                   alpha=1.0, early_stopping=True).fit(X_train, y_train)

        train_acc = classifier.score(X_train, y_train)
        test_acc = classifier.score(X_test, y_test)
        loss = classifier.loss_
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        train_acc_arr[i] = classifier.score(X_train, y_train)
        test_acc_arr[i] = classifier.score(X_test, y_test)

    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    print(f'Max accuracy: {max(test_acc_arr):.4f}')
    print(f'Min accuracy: {min(test_acc_arr):.4f}')
    # TODO: print min and max accuracy as well

    # TODO: Confusion matrix and classification report (for one classifier that performs well)

    print("Predicting on the test set")
    y_pred = classifier.predict(X_test)  # TODO calculate predictions
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):
    """
    Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    parameters = None  # TODO create a dictionary of params

    nn = None  # TODO create an instance of MLPClassifier. Do not forget to set parameters as specified in the HW2 sheet.
    grid_search = None  # TODO create an instance of GridSearchCV from sklearn.model_selection (already imported) with
    # appropriate params. Set: n_jobs=-1, this is another parameter of GridSearchCV, in order to get faster execution of the code.

    # TODO call fit on the train data
    # TODO print the best score
    # TODO print the best parameters found by grid_search
