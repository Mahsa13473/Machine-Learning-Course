#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]

N_TRAIN = 100;
# Select a single feature.
feature_index = 11
x_train = x[0:N_TRAIN, feature_index - 8]
t_train = targets[0:N_TRAIN]

x_test = x[N_TRAIN:, feature_index - 8]
t_test = targets[N_TRAIN:]

(w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3)

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
(y_ev, error) = a1.evaluate_regression(np.matrix(x_ev).transpose(), np.zeros(500), w, 'polynomial', 3)

plt.plot(np.matrix(x_ev).transpose(),y_ev,'g.-')
plt.plot(x_train,t_train,'ro')
plt.plot(x_test,t_test,'bo')
plt.legend(['learned polynomial', 'training data','test data'])
plt.xlabel('x')
plt.ylabel('t')
plt.title('ﬁts for degree 3 polynomials for feature 11 (GNI)')
plt.show()
