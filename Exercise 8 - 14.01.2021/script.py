import time

import numpy as np
from numpy.linalg import eigvals, inv

from library import *

x_true = np.array([[2, 0.5, -0.25]], dtype=np.float64).T
y_true = np.array([[2.1875, 2.25, 2.1875, 2]], dtype=np.float64).T
A_true = np.array([[1, 0.5, 0.25], [1, 1, 1], [1, 1.5, 2.25], [1, 2, 4]],
                  dtype=np.float64)

# start = time.time()
# num, sigma_x, sigma_y = 100, 0.1, 0.01
# means, stds = [], []
# A_eigenvals = []
# X_eigenvals = []
# Ns = list(range(10, 301))
# for N in Ns:
#     temp = []
#     for i in range(num):
#         chi2, A, X = estimate(N, x_true, y_true, sigma_x, sigma_y)
#         temp.append((chi2, A[-1], X[-1]))
#
#     means.append(np.mean(temp, axis=0))
#
# means = np.transpose(means)
# plot(Ns, means)
# print(time.time() - start)
#
# start = time.time()
# num, N, sigma_y = 50, 100, 0.01
# means, stds = [], []
# A_eigenvals = []
# X_eigenvals = []
# sigma_xs = np.array(range(1, 1000)) / 10000
# for sigma_x in sigma_xs:
#     temp = []
#     for i in range(num):
#         chi2, A, X = estimate(N, x_true, y_true, sigma_x, sigma_y)
#         temp.append((chi2, A[-1], X[-1]))
#
#     means.append(np.mean(temp, axis=0))
#
# means = np.transpose(means)
# plot(sigma_xs, means)
# print(time.time() - start)

plt.show()