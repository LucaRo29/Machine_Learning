import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ackley_function(f):
    """
    Plotting the 3D surface for a given cost function f.
    :param f: The function to optimize
    :return:
    """
    n = 200
    bounds = [-2, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f(XX, YY)

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x0, y0, learning_rate, lr_decay, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the parameter x by the gradient_x times the learning_rate,
    and y by the gradient_y times the learning_rate
    The function should return the point (x, y) and the list of errors in each iteration in a numpy array.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate:
    :param lr_decay: A number to multiply learning_rate with, in each iteration (choose a value between 0.75 to 1.0)
    :param max_iter: maximum number of iterations
    :return: x, y (solution), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    x, y = x0, y0

    # TODO Implement a gradient descent algorithm, with a decaying learning rate

    return x, y, E_list


def ackley(x, y):
    # TODO: Implement the cost function specified in the HW1 sheet
    z = # TODO
    return z


def gradient_ackey(x, y):
    # TODO: Implement gradients of Ackley function w.r.t. x and y
    grad_x = # TODO
    grad_y = # TODO
    return grad_x, grad_y