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
    r = None# TODO
    return r

def design_matrix(x, degree=1): # Simple, and polynomial
    """
    :param x: Feature vector (one-dimensional)
    :param degree: Degree in the polynomial expansion (in the simplest case, degree 1)
    :return: Design matrix of shape (n_samples,  degree + 1). e.g., for degree 1, shape is (n_samples, 2)
    """
    # Hint: use np.power and np.arange
    X = None# TODO
    return X

def design_matrix_multilinear(x): # Multilinear
    """
    :param x: Features (MATRIX), shape (n_samples, n_features)
    :return: Design matrix of shape (n_samples, n_features)
    """
    # Hint: Use np.concatenate or np.stack
    X = None# TODO
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
    # TODO
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

    X = None# TODO create a design matrix (use design_matrix function)
    theta = None# TODO calculate theta using pinv from numpy.linalg (already imported)

    y_pred = None# TODO predict the value of y
    mse = None# TODO calculate MSE
    return theta, mse

def multilinear_fit_predict_mse(x, y):
    """
    Use this function for solving the task Multilinear regression!!!

    :param x: Features (MATRIX), shape (n_samples, n_features)
    :param y: Dependent variable (one-dimensional)
    :return: Theta - optimal parameters found; mse - Mean Squared Error
    """
    X = None # TODO create a design matrix (use design_matrix_multilinear function)
    theta = None# TODO calculate theta using pinv from numpy.linalg (already imported)

    y_pred = None# TODO  Predict the value of y
    mse = None# TODO calculate MSE
    return theta, mse
