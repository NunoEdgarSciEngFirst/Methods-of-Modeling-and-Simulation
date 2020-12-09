from library import diffusion_solver
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

D = 5E3
t_end = 172800
N_0 = 2E10
N_I = 20E10
N_max = 1E12
b_0 = 100000
b_I = 600000
b_max = 350000
delta_b = 1000
omega_N = 10000
b = np.array(list(range(b_0, b_I + delta_b, delta_b)))
N = N_max * np.exp(-((b - b_max) / omega_N)**2)
equi = np.linspace(0, 1, len(N), endpoint=True) * (N_I - N_0) + N_0
N += equi

result_standard = diffusion_solver(N, D, t_end=t_end, t_krit=1, dx=1000)
result_implicit = diffusion_solver(N, D, t_end=t_end, t_krit=1, dx=1000,
                                   type_='implicit')

# fig, ax = plt.subplots()
# line1, = ax.plot(b, result_standard[0][1],
#                  label=r'$t={0:.0f}$ s'.format(result_standard[0][0]),
#                  color='tab:blue')
# line2, = ax.plot(b, result_standard[0][1],
#                  label=r'$t={0:.0f}$ s'.format(result_standard[0][0]),
#                  color='tab:orange')
# line3, = ax.plot(b, result_implicit[0][1],
#                  label=r'$t={0:.0f}$ s'.format(result_implicit[0][0]),
#                  color='tab:red')
#
#
# def update(i):
#     line2.set_data(b, result_standard[i][1])
#     line3.set_data(b, result_implicit[i][1])
#     line2.set_label(r'$t={0:.0f}$ s'.format(result_standard[i][0]))
#     legend = ax.legend()
#
#     return [line2, line3, legend]
#
#
# ani = animation.FuncAnimation(fig, update, len(result_standard),
#                               interval=1, repeat_delay=1000, blit=True)
#
# ax.grid()
# plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
B, T = np.meshgrid(b / 1000, np.arange(0, t_end + 1000**2 / (2 * D), 1000**2 /
                                       (2 * D)) / 3600)

ax.plot_surface(B, T, np.array(result_standard) - np.array(result_implicit),
                cmap=cm.seismic,
                linewidth=0,
                antialiased=True)
#
# ax.plot(b/1000, result_implicit[0][1],
#         label=r'Initial Condition',
#         color='tab:blue')
# ax.plot(b/1000, result_implicit[-1][1],
#         label=r'Solution $\Delta t=\Delta t_{crit}$',
#         color='tab:green', linestyle='-')
# ax.plot(b/1000, result_implicit2[-1][1],
#         label=r'Solution $\Delta t=0.2\Delta t_{crit}$',
#         color='tab:red', linestyle='-.')
# ax.plot(b/1000, result_implicit3[-1][1],
#         label=r'Solution $\Delta t=2\Delta t_{crit}$',
#         color='tab:orange', linestyle='--')
# ax.plot(b/1000, equi,
#         label=r'Added Equilibrium',
#         color='k', linestyle='--')

ax.set(xlabel=r'$b_i~~[km]$', ylabel=r'$t~~[h]$', zlabel=r'$n_e~~[m^{-3}]$')
ax.grid()

plt.show()
