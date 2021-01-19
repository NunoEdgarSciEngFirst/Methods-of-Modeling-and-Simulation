from library import *
import matplotlib.pyplot as plt

# True Model
x_true = np.array([[2, 0.5, -0.25]], dtype=np.float64).T
y_true = np.array([[2.1875, 2.25, 2.1875, 2]], dtype=np.float64).T
A_true = np.array([[1, 0.5, 0.25], [1, 1, 1], [1, 1.5, 2.25], [1, 2, 4]],
                  dtype=np.float64)

# Data Collection
Ns = list(range(10, 101))
num = 100
sigma_x, sigma_y = 0.1, 0.01
A_N, eigvals_X_N, chi2_N = [], [], []

for N in Ns:
    temp = [0, 0, 0]
    for i in range(num):
        A, _, X, _, chi2 = run_experiment(N, x_true,
                                          y_true, sigma_x, sigma_y)
        eigvals_X = get_eigenvalues(X)
        temp[0] += A
        temp[1] += eigvals_X
        temp[2] += chi2

    A_N.append(temp[0] / num)
    eigvals_X_N.append(temp[1] / num)
    chi2_N.append(temp[2] / num)

# Data Processing
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

# Eigenvalues of X
# eigvals_X_N = np.array(eigvals_X_N).T
# scale = 400
# labels = [fr'Smallest', fr'Middle', fr'Largest / {scale:d}']
#
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(3):
#     if i == 2:
#         ax.plot(Ns, eigvals_X_N[i] / scale, label=labels[i], color=colors[i],
#                 linestyle=linestyles[i])
#     else:
#         ax.plot(Ns, eigvals_X_N[i], label=labels[i], color=colors[i],
#                 linestyle=linestyles[i])
#
# ax.set(title=fr'Eigenvalues of $X$',
#        xlabel=fr'Sample Size $N$',
#        ylabel=fr'Eigenvalues $\lambda_i$')
# ax.grid()
# ax.legend()

# Accuracy of y
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(Ns, chi2_N, color=colors[0],
#         linestyle=linestyles[0])
#
# ax.set(title=fr'Accuracy of $y$',
#        xlabel=fr'Sample Size $N$',
#        ylabel=fr'Residual $\chi^2 = |y-y_{{true}}|^2$')
# ax.grid()

# Stability of A
# A_chi2_N = [np.sum(np.abs(A - A_true)) for A in A_N]
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(Ns, A_chi2_N, color=colors[0],
#         linestyle=linestyles[0])
#
# ax.set(title=fr'Stability of $A$',
#        xlabel=fr'Sample Size $N$',
#        ylabel=fr'Residual $\chi^2 = \sum_{{ij}} |A_{{ij}}-A_{{true;ij}}|$')
# ax.grid()

# A
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(A_N[-1], vmax=5, vmin=-1)
aux = ax[1].imshow(A_true, vmax=5, vmin=-1)

for i in range(A_true.shape[0]):
    for j in range(A_true.shape[1]):
        ax[0].text(j, i, fr'{A_true[i, j]:.2f}',
                   ha='center', va='center', color='w')
        ax[1].text(j, i, fr'{A_N[-1][i, j]:.2f}',
                   ha='center', va='center', color='w')

fig.colorbar(aux, ax=ax.ravel().tolist())
ax[0].set(title=fr'$A_{{true}}$')
ax[1].set(title=fr'$A_{{N={Ns[-1]}}}$')
ax[0].axis('off')
ax[1].axis('off')

plt.show()
