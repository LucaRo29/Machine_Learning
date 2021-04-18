import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv


def pearson_coeff(x, y):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional)
    :return: Pearson coefficient of correlation
    """
    # Implement it yourself, you are allowed to use np.mean, np.sqrt, np.sum.
    r = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sqrt(
        np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))
    return r


def design_matrix(x, degree=1):  # Simple, and polynomial
    """
    :param x: Feature vector (one-dimensional)
    :param degree: Degree in the polynomial expansion (in the simplest case, degree 1)
    :return: Design matrix of shape (n_samples,  degree + 1). e.g., for degree 1, shape is (n_samples, 2)
    """
    # Hint: use np.power and np.arange

    ones = np.ones(shape=x.shape)
    Xtemp = np.vstack((ones, x))

    X = Xtemp.transpose()

    x_helper = x

    if degree > 1:
        for i in range(degree - 1):
            x_helper = np.power(x_helper, i + 2)
            X = np.hstack((X, x_helper.reshape(-1, 1)))

    return X


def design_matrix_multilinear(x):  # Multilinear
    """
    :param x: Features (MATRIX), shape (n_samples, n_features)
    :return: Design matrix of shape (n_samples, n_features)
    """
    # Hint: Use np.concatenate or np.stack
    x = x.transpose()
    dim = x.shape
    ones = np.ones((dim[0], 1))

    X = np.hstack((ones, x))

    return X


def scatterplot_and_line(x, y, theta):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # Theta will be an array with two coefficients, representing slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.

    pass


def scatterplot_and_curve(x, y, theta):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # Theta will be an array with coefficients.
    # In which format is it stored in the theta array? Take care of that when plotting.
    # Hint: use np.polyval
    # TODO
    pass


def fit_predict_mse(x, y, degree=1):
    """
    Use this function for solving the tasks Meaningful relations, No linear relations, and Polynomial regression!!!

    :param x: Variable 1 (Feature vector (one-dimensional))
    :param y: Variable_2 (one-dimensional), dependent variable
    :param degree: Degree in the polynomial expansion (in the simplest case, degree 1)
    :return: Theta - optimal parameters found; mse - Mean Squared Error
    """

    X = design_matrix(x, degree)
    theta = pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    y_pred = X.dot(theta)
    mse = np.sum((y - y_pred) ** 2) / len(y)

    plt.scatter(x, y)
    plt.plot(x, X.dot(theta), color="red")
    plt.show()

    return theta, mse


def multilinear_fit_predict_mse(x, y):
    """
    Use this function for solving the task Multilinear regression!!!

    :param x: Features (MATRIX), shape (n_samples, n_features)
    :param y: Dependent variable (one-dimensional)
    :return: Theta - optimal parameters found; mse - Mean Squared Error
    """

    X = design_matrix_multilinear(x)

    theta = pinv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    y_pred = X.dot(theta)
    mse = np.sum((y - y_pred) ** 2) / len(y)

    return theta, mse
