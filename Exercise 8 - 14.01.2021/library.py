import time

import numpy as np
from numpy.linalg import eigvals, inv
from numpy.random import normal
import matplotlib.pyplot as plt


def train_model(X, Y):
    A = ((inv(X.T @ X) @ X.T) @ Y).T

    return A


def estimate_y(A, x):
    return A @ x


def get_chi2(y_true, y_model):
    return np.sum((y_true - y_model)**2)


def generate_data(N, x_true, y_true, sigma_x, sigma_y):
    X = normal(loc=x_true, scale=sigma_x, size=(len(x_true), N)).T
    Y = normal(loc=y_true, scale=sigma_y, size=(len(y_true), N)).T

    return X, Y


def get_eigenvalues(A):
    return np.sort(eigvals(A.T @ A))


def run_experiment(N, x_true, y_true, sigma_x, sigma_y):
    X, Y = generate_data(N, x_true, y_true, sigma_x, sigma_y)
    A_estimate = train_model(X, Y)
    y_estimate = estimate_y(A_estimate, x_true)
    chi2 = get_chi2(y_estimate, y_true)

    return A_estimate, y_estimate, X, Y, chi2

# def plot(x, means):
#     fig, ax = plt.subplots()
#     ax.plot(x, means[0], label=fr'$\chi^2$')
#     ax.set(
#         title=r'Mean $\chi^2$',
#         xlabel='Sample Size $N$',
#         ylabel='Mean $\chi^2$')
#     ax.grid()
#
#     fig, ax = plt.subplots()
#     ax.plot(x, means[1], color='tab:blue', label=fr'Largest Eigenvalue of A')
#     ax.tick_params(axis='y', labelcolor='tab:blue')
#     ax.set(ylabel=fr'Largest Eigenvalue of A')
#     ax = ax.twinx()
#     ax.tick_params(axis='y', labelcolor='tab:orange')
#     ax.plot(x, means[2], color='tab:orange',
#             label=fr'Largest Eigenvalue of X')
#     ax.set(ylabel=fr'Largest Eigenvalue of X')
#     ax.set(
#         title=fr'Largest Eigenvalues',
#         xlabel=fr'Sample Size $N$')
#     ax.grid()
# N, sigma_x, sigma_y = 100, 0.1, 0.01
# chi_2 = estimate(N, x_true, y_true, sigma_x, sigma_y)

# y_true_check = A_true @ x_true
# A_eigenvalues_check = eigvals(A_true.T @ A_true)
# x_inv_check = (inv(A_true.T @ A_true) @ A_true.T) @ y_true
#
#
# y_estimate_check = A_estimate @ x_true
# dy_estimate_check = y_estimate_check - y_true
# X_eigenvalues_check = eigvals(X.T @ X)
