from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    return mse

def visualize_3D_data(x, y):
    """
    :param x: Datapoints - (x, y) coordinates. Shape: (n_samples, 2)
    :param y: Datapoints y. Shape: (n_samples, ). y = f(x1, x2)
    :return:
    """
    # TODO: 3D plot to illustrate data cloud (we want to see data points, not surface)
    pass

def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons_list = # TODO (try at least 3 different numbers of neurons)

    # TODO: MLPRegressor, choose the model yourself

    # Calculate predictions
    y_pred_train = # TODO
    y_pred_test = # TODO
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
