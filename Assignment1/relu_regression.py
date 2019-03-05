#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]

N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN, 3] #feature 11
t_train = targets[0:N_TRAIN]

x_test = x[N_TRAIN:, 3] #feature 11
t_test = targets[N_TRAIN:]

(w, tr_err) = a1.linear_regression(x_train, t_train, 'ReLU', 0, 0)
(t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'ReLU', 0)

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num =1000)
(y_ev, error) = a1.evaluate_regression(np.matrix(x_ev).transpose(), np.zeros(1000), w, 'ReLU', 0)

print("training error for ReLU =" , tr_err,)
print("testing error for ReLU =", te_err)

plt.plot(np.matrix(x_ev).transpose(),y_ev,'g.-')
plt.plot(x_train,t_train,'ro')

plt.legend(['learned polynomial', 'training data'])
plt.xlabel('x')
plt.ylabel('t')
plt.title('ﬁt for feature 11 (GNI) using ReLU ')
plt.show()
