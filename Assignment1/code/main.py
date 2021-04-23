import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing

from lin_reg_ecg import test_fit_line, find_new_peak, check_if_improved
from lin_reg_smartwatch import pearson_coeff, fit_predict_mse, multilinear_fit_predict_mse, scatterplot_and_line, \
    scatterplot_and_curve, design_matrix_multilinear
from gradient_descent import ackley, gradient_ackey, gradient_descent, plot_ackley_function
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def task_1_1():
    test_fit_line()

    # Load ecg signal from 'data/ecg.npy' using np.load
    ecg = np.load("data/ecg.npy")

    # Load indices of peaks from 'indices_peaks.npy' using np.load. There are 83 peaks.
    peaks = np.load("data/indices_peaks.npy")

    # Create a "timeline". The ecg signal was sampled at sampling rate of 180 Hz, and in total 50 seconds.
    # Datapoints are evenly spaced. Hint: shape of time signal should be the same as the shape of ecg signal.
    time = np.arange(0, 50, 1 / 180)
    print(f'time shape: {time.shape}, ecg signal shape: {ecg.shape}')
    print(f'First peak: ({time[peaks[0]]:.3f}, {ecg[peaks[0]]:.3f})')  # (0.133, 1.965)

    # Plot of ecg signal (should be similar to the plot in Fig. 1A of HW1, but shown for 50s, not 8s)
    plt.plot(time, ecg)
    plt.plot(time[peaks], ecg[peaks], "x")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.show()

    new_peaks = np.zeros(peaks.size)
    new_sig = np.zeros(peaks.size)
    improved = np.zeros(peaks.size)

    for i, peak in enumerate(peaks):
        x_new, y_new = find_new_peak(peak, time, ecg)
        new_peaks[i] = x_new
        new_sig[i] = y_new
        improved[i] = check_if_improved(x_new, y_new, peak, time, ecg)

    print(f'Improved peaks: {np.sum(improved)}, total peaks: {peaks.size}')
    print(f'Percentage of peaks improved: {np.sum(improved) / peaks.size :.4f}')


def task_1_2():
    # COLUMN NAMES: hours_sleep, hours_work, avg_pulse, max_pulse, duration, exercise_intensity, fitness_level, calories
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, "duration": 4,
                    "exercise_intensity": 5, "fitness_level": 6, "calories": 7}
    # Load the data from 'data/smartwatch_data.npy' using np.load
    smartwatch_data = np.load("data/smartwatch_data.npy")

    # Now you can access it, for example,  smartwatch_data[:, column_to_id["hours_sleep"]]

    # Meaningful relations

    print("duration -> calories")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("max_pulse -> avg_pulse")
    x = smartwatch_data[:, column_to_id["max_pulse"]]
    y = smartwatch_data[:, column_to_id["avg_pulse"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("duration -> exercise_intensity")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["exercise_intensity"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("duration -> fitness_level")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["fitness_level"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)

    # No linear relations
    print()
    print()
    print()
    print("No linear relations")
    print()

    print("hours_sleep -> hours_work")
    x = smartwatch_data[:, column_to_id["hours_sleep"]]
    y = smartwatch_data[:, column_to_id["hours_work"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("hours_sleep -> exercise_intensity ")
    x = smartwatch_data[:, column_to_id["hours_sleep"]]
    y = smartwatch_data[:, column_to_id["exercise_intensity"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)

    print()
    print("avg_pulse -> exercise_intensity ")
    x = smartwatch_data[:, column_to_id["avg_pulse"]]
    y = smartwatch_data[:, column_to_id["exercise_intensity"]]
    print("pearson_coeff")
    print(pearson_coeff(x, y))
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    # Polynomial regression
    print("Polynomial: max_pulse -> avg_pulse")
    x = smartwatch_data[:, column_to_id["max_pulse"]]
    y = smartwatch_data[:, column_to_id["avg_pulse"]]
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: max_pulse -> avg_pulse")
    x = smartwatch_data[:, column_to_id["max_pulse"]]
    y = smartwatch_data[:, column_to_id["avg_pulse"]]
    t, m = fit_predict_mse(x, y, 2)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: max_pulse -> avg_pulse")
    x = smartwatch_data[:, column_to_id["max_pulse"]]
    y = smartwatch_data[:, column_to_id["avg_pulse"]]
    t, m = fit_predict_mse(x, y, 3)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: max_pulse -> avg_pulse")
    x = smartwatch_data[:, column_to_id["max_pulse"]]
    y = smartwatch_data[:, column_to_id["avg_pulse"]]
    t, m = fit_predict_mse(x, y, 4)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    # degree = 2 was best

    print("Polynomial: duration -> calories")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    t, m = fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: duration -> calories")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    t, m = fit_predict_mse(x, y, 2)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: duration -> calories")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    t, m = fit_predict_mse(x, y, 3)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    print("Polynomial: duration -> calories")
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    t, m = fit_predict_mse(x, y, 4)
    print("theta")
    print(t)
    print("mse")
    print(m)
    print()

    # degree 3 was best

    # Multilinear

    print("duration & exercise_intensity & avg_pulse -> calories")
    x1 = smartwatch_data[:, column_to_id["duration"]]
    x2 = smartwatch_data[:, column_to_id["exercise_intensity"]]
    x3 = smartwatch_data[:, column_to_id["avg_pulse"]]
    x = np.vstack((x1, x2))
    x = np.vstack((x, x3))
    y = smartwatch_data[:, column_to_id["calories"]]

    t, m = multilinear_fit_predict_mse(x, y)
    print("theta")
    print(t)
    print("mse")
    print(m)

    # When choosing two variables for polynomial regression, use a pair that you used for Meaningful relations, so you can check if the MSE decreases.
    # When choosing a few variables for multilinear regression, use a pair that you used for Meaningful relations, so you can check if the MSE decreases.


def task_2():
    heart_data = np.load("data/heart_data.npy")
    heart_data_targets = np.load("data/heart_data_targets.npy")

    sc = StandardScaler()  # TODO normalize data using StandardScaler from sklearn.preprocessing (already imported)
    X_normalized = sc.fit_transform(heart_data)

    # Spilit data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X_normalized, heart_data_targets,
                                                        test_size=0.2, random_state=0)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Create a classifier
    #clf = LogisticRegression(solver='lbfgs',penalty="l2").fit(x_train,y_train) #Train accuracy: 0.8430. Test accuracy: 0.8525.
    #clf = LogisticRegression(solver='lbfgs', penalty='none').fit(x_train, y_train) #Train accuracy: 0.8347. Test accuracy: 0.8525.

    #clf = LogisticRegression(solver='liblinear').fit(x_train, y_train) #Train accuracy: 0.8430. Test accuracy: 0.8361.
    #clf = LogisticRegression(solver='liblinear',penalty="l1").fit(x_train, y_train)#Train accuracy: 0.8430. Test accuracy: 0.8361.
    #clf = LogisticRegression(solver='liblinear', penalty="l2").fit(x_train, y_train)#Train accuracy: 0.8430. Test accuracy: 0.8361.

    #clf = LogisticRegression(solver='newton-cg', penalty='none').fit(x_train, y_train)    #Train accuracy: 0.8347. Test accuracy: 0.8525.
    #clf = LogisticRegression(solver='newton-cg', penalty='l2').fit(x_train, y_train) #Train accuracy: 0.8430. Test accuracy: 0.8525.

    #clf = LogisticRegression(solver='lbfgs', penalty='none',C=10).fit(x_train, y_train) #Train accuracy: 0.8347. Test accuracy: 0.8525.
    #clf = LogisticRegression(solver='lbfgs', penalty='none', C=0.1).fit(x_train, y_train) #Train accuracy: 0.8347. Test accuracy: 0.8525.

    #clf = LogisticRegression(solver='newton-cg', penalty='none',C=10).fit(x_train, y_train) #Train accuracy: 0.8347. Test accuracy: 0.8525.
    #clf = LogisticRegression(solver='newton-cg', penalty='none', C=0.1).fit(x_train, y_train) #Train accuracy: 0.8347. Test accuracy: 0.8525.

    clf = LogisticRegression().fit(x_train, y_train) #Train accuracy: 0.8430. Test accuracy: 0.8525.

    acc_train = clf.score(x_train, y_train)
    acc_test = clf.score(x_test, y_test)

    print(f'Train accuracy: {acc_train:.4f}. Test accuracy: {acc_test:.4f}.')

    # Calculate predictions and log_loss
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    loss_train, loss_test = log_loss(y_train,y_train_pred), log_loss(y_test,y_test_pred)
    print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

    print()
    print("Theta:")
    print(clf.coef_)
    print()
    print("Intercept:")
    print(clf.intercept_)





def task_3():
    # Plot the Function, to see how it looks like
    plot_ackley_function(ackley)

    # Choose a random starting point
    x0 = np.random  # TODO choose a random starting x-coordinate, use rand function from np.random
    y0 = np.random  # TODO choose a random starting y-coordinate, use rand function from np.random
    print(x0, y0)

    # Call the function gradient_descent
    # Choose max_iter, learning_rate, lr_decay (first see what happens with lr_decay=1, then change it to a lower value)
    x, y, E_list = gradient_descent(ackley, gradient_ackey, x0, y0, learning_rate = 0.001, lr_decay = 1 , max_iter = 1000)

    # Print the point that is the best found solution
    print(f'{x:.4f}, {y:.4f}')

    # TODO Make a plot of the cost over iteration. Do not forget to label the plot (xlabel, ylabel, title)

    print(f'Solution found: f({x:.4f}, {y:.4f})= {ackley(x, y):.4f}')
    print(f'Global optimum: f(0, 0)= {ackley(0, 0):.4f}')


def main():
    # task_1_1()
    # task_1_2()
    #task_2()
    task_3()


if __name__ == '__main__':
    main()
