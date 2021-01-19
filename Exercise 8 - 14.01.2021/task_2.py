from library import *
import matplotlib.pyplot as plt

# True Model
x_true = np.array([[2, 0.5, -0.25]], dtype=np.float64).T
y_true = np.array([[2.1875, 2.25, 2.1875, 2]], dtype=np.float64).T
A_true = np.array([[1, 0.5, 0.25], [1, 1, 1], [1, 1.5, 2.25], [1, 2, 4]],
                  dtype=np.float64)

# Data Collection
N = 300
num = 100
sigma_xs = np.logspace(-6, 0, 100)
sigma_y = 0.01
A_s, eigvals_X_s, chi2_s = [], [], []

for sigma_x in sigma_xs:
    temp = [0, 0, 0]
    for i in range(num):
        A, _, X, _, chi2 = run_experiment(N, x_true,
                                          y_true, sigma_x, sigma_y)
        eigvals_X = get_eigenvalues(X)
        temp[0] += A
        temp[1] += eigvals_X
        temp[2] += chi2

    A_s.append(temp[0] / num)
    eigvals_X_s.append(temp[1] / num)
    chi2_s.append(temp[2] / num)

# Data Processing
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

# Eigenvalues of X
# eigvals_X_s = np.array(eigvals_X_s).T
# labels = [fr'Smallest', fr'Middle', fr'Largest']
#
# fig, ax = plt.subplots(figsize=(10, 6))
# plots = []
# for i in range(2):
#     plots.append(ax.semilogx(sigma_xs, eigvals_X_s[i], label=labels[i],
#                              color=colors[i], linestyle=linestyles[i])[0])
#
# ax2 = ax.twinx()
# plots.append(
#     ax2.semilogx(sigma_xs, eigvals_X_s[2], label=labels[2], color=colors[2],
#                  linestyle=linestyles[2])[0])
# ax.set(title=fr'Eigenvalues of $X$',
#        xlabel=fr'Standard Deviation Samples $\sigma_x$',
#        ylabel=r'Smallest/Middle Eigenvalues $\lambda_{1,2}$')
# ax2.set(ylabel=fr'Largest Eigenvalue $\lambda_3$')
# ax2.tick_params(axis='y', labelcolor=colors[2])
# ax.grid()
# ax.legend(handles=plots)

# Accuracy of y
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.semilogx(sigma_xs, chi2_s, color=colors[0],
#         linestyle=linestyles[0])
#
# ax.set(title=fr'Accuracy of $y$',
#        xlabel=fr'Standard Deviation Samples $\sigma_x$',
#        ylabel=fr'Residual $\chi^2 = |y-y_{{true}}|^2$')
# ax.grid()

# Stability of A
A_chi2_s = [np.sum(np.abs(A - A_true)) for A in A_s]

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(sigma_xs, A_chi2_s, color=colors[0],
            linestyle=linestyles[0])

ax.set(title=fr'Stability of $A$',
       xlabel=fr'Standard Deviation Samples $\sigma_x$',
       ylabel=fr'Residual $\chi^2 = \sum_{{ij}} |A_{{ij}}-A_{{true;ij}}|$')
ax.grid()

# A
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
vmax = np.ceil(max(np.max(A_s[0][:]), np.max(A_true[:])))
vmin = np.floor(min(np.min(A_s[0][:]), np.min(A_true[:])))
ax[1].imshow(A_s[0], vmax=vmax, vmin=vmin)
aux = ax[0].imshow(A_true, vmax=vmax, vmin=vmin)

for i in range(A_true.shape[0]):
    for j in range(A_true.shape[1]):
        ax[0].text(j, i, fr'{A_true[i, j]:.2f}',
                   ha='center', va='center', color='w')
        ax[1].text(j, i, fr'{A_s[0][i, j]:.2f}',
                   ha='center', va='center', color='w')

fig.colorbar(aux, ax=ax.ravel().tolist())
ax[0].set(title=fr'$A_{{true}}$')
ax[1].set(title=fr'$A_{{N={sigma_xs[0]}}}$')
ax[0].axis('off')
ax[1].axis('off')

plt.show()
