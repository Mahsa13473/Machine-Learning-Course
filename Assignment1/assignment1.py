"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding="ISO-8859-1")
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    # e.g. phi = design_matrix(x,basis, degree)
    phi = design_matrix(x, basis, degree)
    #print(phi)

    # TO DO:: Compute coefficients using phi matrix
    # Compute Moore - Penrose Pseudoinverse
    if reg_lambda == 0:
        pinv_phi = np.linalg.pinv(phi)
    else:
        #Normal Equation when we have L2 regularization
        pinv_phi = np.dot(np.linalg.inv(np.dot(phi.transpose(), phi)+ (reg_lambda * np.identity(phi.shape[1]))), phi.transpose())

    w = np.dot(pinv_phi, t)

    # Measure root mean squared error on training data.
    y_train = np.dot(phi, w)
    delta = y_train - t
    train_err = np.sqrt(np.sum(np.power(delta, 2)) / x.shape[0])
    return (w, train_err)

def design_matrix(x, basis, degree=0):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':

        phi = np.ones(shape=(x.shape[0], x.shape[1] * degree))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for d in range(degree):
                    phi[i][j * degree + d] = x[i, j]**(d + 1)

        phi = np.append(np.ones(shape=(x.shape[0], 1)), phi, axis=1)


    elif basis == 'ReLU':
        phi = np.ones(shape=(x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                phi[i][j] = max(-1*x[i][j]+5000, 0)
        phi = np.append(np.ones(shape=(x.shape[0], 1)), phi, axis=1)
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
  	# TO DO:: Compute t_est and err
    phi_test = design_matrix(x, basis, degree)
    t_est = phi_test * w
    delta_test = t - t_est
    err = np.sqrt(np.sum(np.power(delta_test, 2)) / x.shape[0])

    return (t_est, err)


def linear_regression_reg_Which_Lambda_Cross_Validation(x, t, lambdaa, basis, degree):
    tr_err = np.zeros(10)
    val_err = np.zeros(10)
    mean_error = np.zeros(8)

    for i in range (8):
        for j in range (10):

            #Seperate data for 10-fold cross validation
            x_train = np.concatenate((x[0:j*10,:], x[(j+1)*10:100,:]), axis=0)
            t_train = np.concatenate((t[0:j*10,:], t[(j+1)*10:100,:]), axis=0)

            x_val = x[j*10: (j+1)*10]
            t_val = t[j*10: (j+1)*10]

            (w, tr_err[j]) = linear_regression(x_train, t_train, basis, lambdaa[i], degree)
            (t_est, val_err[j]) = evaluate_regression(x_val, t_val, w, basis, degree)

        mean_error[i] = val_err.mean(axis = 0)


    Best_Lambdaa = lambdaa[np.where(mean_error == min(mean_error))]

    return (Best_Lambdaa, mean_error)
