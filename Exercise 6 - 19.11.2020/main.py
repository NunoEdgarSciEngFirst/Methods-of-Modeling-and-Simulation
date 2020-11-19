from library import diffusion_solver
import matplotlib.pyplot as plt
import numpy as np

D = 5E3
t_end = 172800
N_max = 1E11
b_max = 350000
omega_N = 10000
b = 1000 * np.array(list(range(1, 600 + 1)))
N = N_max * np.exp(-(b - b_max) ** 2 / omega_N ** 2)
N[0] = 2E10
N[-1] = 20E10
t, result = diffusion_solver(N, D, t_end=t_end, dx=1000, dt=10)

_, ax = plt.subplots()
ax.plot(b, result[0], label=r'$t={}$'.format(t[0]))
ax.plot(b, result[-1], label=r'$t={}$'.format(t[-1]))

ax.grid()
ax.legend()
plt.show()
