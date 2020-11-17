# Third Party Import
import matplotlib.pyplot as plt
import numpy as np

# Own Import
from fitting import polynomial_fit, eval_result, get_polynomial_funcs

padding = 1.1

####################### Exercise 1 #######################

x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y = [2.2, 2.6, 3.6, 3.7, 6.1, 7.6, 8.3, 8.8, 9.4]
sigma = 0.5
order = 1

parameters, p_val, C, R = polynomial_fit(x, y, sigma, order)
funcs = get_polynomial_funcs(order)
variance = np.diagonal(C)

x_mod = np.linspace(min(x) / padding, max(x) * padding, 50)
y_mod = eval_result(x_mod, parameters, funcs)
y_min = eval_result(x_mod, parameters + np.sqrt(variance), funcs)
y_max = eval_result(x_mod, parameters - np.sqrt(variance), funcs)

fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=sigma, label=fr'Data', capsize=6, elinewidth=1,
            color='k', linewidth=0.75, linestyle='',
            marker='x', markersize=10)
ax.plot(x_mod, y_mod, label=fr'Linear Fit', linewidth=0.75, color='tab:blue')
ax.plot(x_mod, y_min, label=fr'$1\sigma$', linewidth=0.75, color='tab:orange')
ax.plot(x_mod, y_max, linewidth=0.75, color='tab:orange')

ax.text(6.05, 1.5, s=r'$\chi^2={1:.3f};~p={0:.3f}$'.format(
    *p_val), color='tab:blue')
ax.text(6.05, 2.2, s=r'$y=({1:.2f}\pm{3:.2f})*x+({0:.1f}\pm{2:.1f})$'.format(*parameters, *np.sqrt(variance), p_val[0]), color='tab:blue')


ax.legend()
ax.grid()
ax.set(xlabel=fr'x [1]', ylabel=fr'y [1]',
       title=fr'Example 1')

####################### Exercise 2 #######################

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [7.3, 3.1, 11.7, 11.9, 15.5, 31.0, 33.9, 47.0, 61.5, 71.9]
sigma = 4.3
order = 2

parameters, p_val, C, R = polynomial_fit(x, y, sigma, order)
funcs = get_polynomial_funcs(order)
variance = np.diagonal(C)

parameters_lin, p_val_lin, C_lin, R_lin = polynomial_fit(x, y, sigma, 1)
funcs_lin = get_polynomial_funcs(order)
variance_lin = np.diagonal(C_lin)

x_mod = np.linspace(min(x) / padding, max(x) * padding, 50)
y_mod = eval_result(x_mod, parameters, funcs)
y_lin = eval_result(x_mod, parameters_lin, funcs_lin)

y_min = eval_result(x_mod, parameters + np.sqrt(variance), funcs)
y_max = eval_result(x_mod, parameters - np.sqrt(variance), funcs)

fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=sigma, label=fr'Data', capsize=6, elinewidth=1,
            color='k', linewidth=0.75, linestyle='',
            marker='x', markersize=10)
ax.plot(x_mod, y_mod, label=fr'Quadratic Fit', linewidth=0.75, color='tab:blue')
ax.plot(x_mod, y_lin, label=fr'Linear Fit', linewidth=0.75, color='tab:green')
ax.plot(x_mod, y_min, label=fr'$1\sigma$', linewidth=0.75, color='tab:orange')
ax.plot(x_mod, y_max, linewidth=0.75, color='tab:orange')

ax.text(0.5, 75, s=r'$\chi^2={1:.3f};~p={0:.3f}$'.format(*p_val_lin),
        color='tab:green')
ax.text(0.5, 82, s=r'$y=({1:.2f}\pm{3:.2f})*x+({0:.2f}\pm{2:.2f})$'.format(*parameters_lin, *np.sqrt(variance_lin)), color='tab:green')

ax.text(0.5, 55, s=r'$\chi^2={1:.3f};~p={0:.3f}$'.format(*p_val),
        color='tab:blue')
ax.text(0.5, 62, s=r'$y=({2:.2f}\pm{5:.2f})*x^2+(0\pm{4:.0f})*x+({0:.0f}\pm{3:.0f})$'.format(*parameters, *np.sqrt(variance)), color='tab:blue')

ax.legend()
ax.grid()
ax.set(xlabel=fr'x [1]', ylabel=fr'y [1]',
       title=fr'Example 2')

####################### Exercise 3 #######################

x = [0.002083, 0.001010, 0.000680, 0.000508, 0.000265, 0.000210,
     0.000099, 0.000072,
     0.000046, 0.000015]
y = [0.033333, 0.021053, 0.017544, 0.015385, 0.012821, 0.012195,
     0.011111, 0.010870,
     0.010526, 0.010256]
sigma = [0.003333, 0.001330, 0.000923, 0.000710, 0.000493, 0.000446,
         0.000370, 0.000354,
         0.000332, 0.000316]
order = 1

parameters, p_val, C, R = polynomial_fit(x, y, sigma, order)
parameters_no, *_ = polynomial_fit(x, y, 0, order)
funcs = get_polynomial_funcs(order)
variance = np.diagonal(C)
print(p_val,C,R)
x_mod = np.linspace(min(x) / padding, max(x) * padding, 50)
y_mod = eval_result(x_mod, parameters, funcs)
y_no = eval_result(x_mod, parameters_no, funcs)
y_min = eval_result(x_mod, parameters + np.sqrt(variance), funcs)
y_max = eval_result(x_mod, parameters - np.sqrt(variance), funcs)

fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=sigma, label=fr'Data', capsize=6, elinewidth=1,
            color='k', linewidth=0.75, linestyle='',
            marker='x', markersize=10)
ax.plot(x_mod, y_mod, label=fr'Linear Fit', linewidth=0.75, color='tab:blue')
ax.plot(x_mod, y_no, label=fr'No Weight', linewidth=0.75, color='tab:green')
ax.plot(x_mod, y_min, label=fr'$1\sigma$', linewidth=0.75, color='tab:orange')
ax.plot(x_mod, y_max, linewidth=0.75, color='tab:orange')


ax.text(0.0005, 0.031,
        s=r'$y={1:.1f}*x+{0:.3f}$'.format(*parameters_no), color='tab:green')

ax.text(0.0005, 0.0105, s=r'$\chi^2={1:.3f};~p=1$'.format(*p_val),
        color='tab:blue')
ax.text(0.0005, 0.012, s=r'$y=({1:.1f}\pm{3:.1f})*x+({0:.5f}\pm{2:.5f})$'.format(*parameters, *np.sqrt(variance)), color='tab:blue')
ax.legend()
ax.grid()
ax.set(xlabel=fr'x [1]', ylabel=fr'y [1]',
       title=fr'Example 3')

plt.show()
