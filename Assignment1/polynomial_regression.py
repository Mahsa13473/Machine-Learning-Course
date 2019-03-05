#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
tr_err = np.zeros(6)
te_err = np.zeros(6)

# Repeat for degree 1 to degree 6 polynomial
for degree in range(1, 7):

    # linear_regression(x, t, basis, reg_lambda=0, degree=0):
    (w, tr_err[degree - 1]) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degree)
    #evaluate_regression(x, t, w, basis, degree)
    (t_est, te_err[degree - 1]) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree)


#Produce a plot of results.
degree = [1,2,3,4,5,6]
plt.plot(degree, te_err)
plt.plot(degree, tr_err)
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials, no regularization, un-normalized training set')
plt.xlabel('Polynomial degree')
plt.show()
