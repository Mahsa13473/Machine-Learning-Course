#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py
tr_err = np.zeros(8)
te_err = np.zeros(8)

#for feature 8 to 15
for i in range(8):
    (w, tr_err[i]) = a1.linear_regression(x_train[:,i], t_train, 'polynomial', 0, 3)
    (t_est, te_err[i]) = a1.evaluate_regression(x_test[:,i], t_test, w, 'polynomial', 3)


# plot (using https://pythonspot.com/matplotlib-bar-chart/)

# create plot
fig, ax = plt.subplots()
index = np.arange(8)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, tr_err, bar_width,
                 alpha=opacity,
                 color='b',
                 label='traing error')

rects2 = plt.bar(index + bar_width, te_err, bar_width,
                 alpha=opacity,
                 color='r',
                 label='test error')

plt.xlabel('Features')
plt.ylabel('RMS')
plt.title('training error and testing error for features 8 to 15')
plt.xticks(index + bar_width, ('8', '9', '10', '11', '12', '13', '14', '15'))
plt.legend()

plt.tight_layout()
plt.show()

