#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta_range = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')

# Randomly shuffle data set
data = np.random.permutation(data)

# Data matrix, with column of ones at end.
X = data[:, 0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]
# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

plt.figure()
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
epsilon = 1e-16

for eta in eta_range:
    # Initialize w.
    w = np.array([0.1, 0, 0])

    # Error values over all iterations.
    e_all = []

    for iter in range(0, max_iter):
        # repeat process on each input
        for i in range(0, len(X)):
            y = sps.expit(np.dot(X[i], w))
            grad_e = np.dot((y-t[i]),X[i])
            w = w - eta*grad_e

        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X, w))

        #solve problem of log(0)
        y = np.clip(y, epsilon, 1 - epsilon)

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

         # Add this error to the end of error vector.
        e_all.append(e)

        # Stop iterating if error doesn't change more than tol.
        if iter > 0:
            if np.absolute(e - e_all[iter - 1]) < tol:
                break

    plt.plot(e_all)


plt.legend(eta_range)
plt.show()
