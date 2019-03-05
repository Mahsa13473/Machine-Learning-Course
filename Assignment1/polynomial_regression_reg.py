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

tr_err = np.zeros(8)
te_err = np.zeros(8)

lambdaa = np.array([0, 0.01, 0.1, 1, 10, 100, 1000, 10000])
degree = 2
Best_Lambdaa = 0

#mean_error = np.zeros(8)

(Best_Lambdaa, mean_error) = a1.linear_regression_reg_Which_Lambda_Cross_Validation(x_train, t_train, lambdaa, 'polynomial', 2)

print("Best Lambda = ", Best_Lambdaa)
print("unregularized result =", mean_error[0])
print(mean_error)

plt.matplotlib.pyplot.semilogx(lambdaa, mean_error)

plt.xlabel('Lambda')
plt.ylabel('average validation set error')
plt.legend(['mean validation error'])

plt.title('polynomial degree 2 with regularization and 10-fold cross validation')
plt.show()


