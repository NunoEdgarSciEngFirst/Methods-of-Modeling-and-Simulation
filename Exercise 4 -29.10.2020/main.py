# Third Party Import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Own Import
from fitting import (polynomial_fit, eval_result,
                     get_polynomial_funcs, param_distribution)

padding = 1.05

####################### Exercise 1 #######################

x = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
y = [81, 50, 35, 27, 26, 60, 106, 189, 318, 520]
sigma = np.sqrt(y)
orders = [4]

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, label=fr'Data', capsize=5, elinewidth=1,
            color='k', linewidth=0.75, linestyle='',
            marker='x', markersize=7)

for order in orders:
    parameters, p_val, C, R = polynomial_fit(x, y, list(sigma), order)
    funcs = get_polynomial_funcs(order)
    variance = np.diagonal(C)
    #print('Order {}:\np-value:\t{}\nChi^2:\t\t{}\n'.format(order,p_val[0],p_val[1]))
    print('Order {}:\t{}'.format(order, parameters))
    print('Error:\t\t{}\n'.format(variance))
    x_mod = np.linspace(min(x) * padding, max(x) * padding, 50)
    y_mod = eval_result(x_mod, parameters, funcs)
    y_np = eval_result(x_mod, parameters, funcs)
    y_min = eval_result(x_mod, parameters +
                        np.sign(parameters) * np.sqrt(variance), funcs)
    y_max = eval_result(x_mod, parameters -
                        np.sign(parameters) * np.sqrt(variance), funcs)

    ax.plot(x_mod, y_mod, label=r'{0:d}. Order: $p={1:.3f}$'.format(
        order, *p_val), linewidth=0.75)
    ax.plot(x_mod, y_min, label=r'$1\sigma Error', linewidth=0.75, color='k')
    ax.plot(x_mod, y_max, linewidth=0.75, color='k')


ax.legend()
ax.grid()
ax.set(xlabel=fr'$x$', ylabel=fr'$y$',
       title=fr'Polynomial Fit for Example 1')
'''

####################### Exercise 2 #######################

x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y = [2.2, 2.6, 3.6, 3.7, 6.1, 7.6, 8.3, 8.8, 9.4]
sigma = 0.5
order = 1

parameters, _, C, _ = polynomial_fit(x, y, sigma, order)
stds = np.sqrt(np.diagonal(C))
print(parameters, stds)
grids, f_a, levels = param_distribution(
    C, contour_levels=[1, .9973, .9545, .9, .6827], res=100)

level_names = [fr'$3\sigma$', fr'$2\sigma$', fr'$90\%$', fr'$1\sigma$', fr'$\Delta\chi^2=1$']
grids = [grid + parameter for grid, parameter in zip(grids, parameters)]
sorted_levels = sorted(levels)
sorted_names = [level_name for _,
                level_name in sorted(zip(levels, level_names))]

fig, ax = plt.subplots()
plot = ax.contour(*grids, f_a, levels=sorted_levels, colors=[
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])

fmt = {}
for l, s in zip(plot.levels, sorted_names):
    fmt[l] = s

p = plot.collections[-1].get_paths()[-1].vertices
x, y = p[:, 0], p[:, 1]
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)

# Contour
ax.axvline(x=parameters[0], color='k', linestyle='--', linewidth=1)
ax.axhline(y=parameters[1], color='k', linestyle='--', linewidth=1)
ax.hlines(y=y_min, xmin=x_min - 0.1, xmax=x_max + 0.1,
          color='k', linestyle=':', linewidth=1)
ax.hlines(y=y_max, xmin=x_min - 0.1, xmax=x_max + 0.1,
          color='k', linestyle=':', linewidth=1)
ax.vlines(x=x_min, ymin=y_min - 0.02, ymax=y_max + 0.02,
          color='k', linestyle=':', linewidth=1)
ax.vlines(x=x_max, ymin=y_min - 0.02, ymax=y_max + 0.02,
          color='k', linestyle=':', linewidth=1)
ax.text(0.45, 0.83, s=r'$d= {1:.1f}\pm{0:.1f}$'.format(
    stds[0], parameters[0]), color='k')
ax.text(-0.29, 0.98, s=r'$k= {1:.2f}\pm{0:.2f}$'.format(
    stds[1], parameters[1]), color='k')

ax.grid()
ax.set(xlabel=fr'd', ylabel=fr'k',
       title=fr'Contour Lines of the Density Distribution for Example 2')
# ax.clabel(plot, inline=1, fontsize=8, fmt=fmt,
#         use_clabeltext=True, manual=True)

# Surface
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(*grids, f_a, linewidth=0, cmap=cm.jet, antialiased=True)
plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0),useOffset=False)

fig.colorbar(surf, shrink=0.75)

ax.set(xlabel=fr'$d$', ylabel=fr'$k$', zlabel=r'$f~~\left(\Delta a\right)$',
       title=fr'Parameter Distribution for Example 2')
'''
plt.show()
