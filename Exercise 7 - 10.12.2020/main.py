from library import euler, rk4
import numpy as np
import matplotlib.pyplot as plt

theta_0 = 0
phi_0s = np.deg2rad([0, 30, 60, 89.9])
N_0 = 315
H_n = 7000
r_0 = 6371000
h = 100
z = 650000

N = lambda r: N_0 * np.exp((r_0 - r) / H_N)
n = lambda N: 1 + 1E-6 * N
phi = lambda r, C: np.arcsin(C / (n(r) * r))
dtheta = lambda r, C: C / (n(r) * r**2 * np.cos(phi(r, C)))

ax = plt.subplot(111, projection='polar')

styles = ['-', '--', '-.', ':']
for phi_0, linestyle in zip(phi_0s, styles):
    C = n(r_0) * r_0 * np.sin(phi_0)
    func = lambda r: dtheta(r, C)
    result = euler(func, r_0 + z, t_0=r_0, dt=h)
    ax.plot(result[1], result[0] / 1000, linestyle=linestyle,
            label=r'$\Phi_0 = {0:.1f}°$'.format(np.rad2deg(phi_0)))

ax.grid(True)
ax.set_rmin(r_0 / 1000)
ax.set_rmax((r_0 + z) / 1000)
ax.set_thetamin(-5)
ax.set_thetamax(25)
ax.set_theta_offset(np.deg2rad(90))
ax.set_theta_direction(-1)
ax.set_rlabel_position(np.deg2rad(90))
ax.set_title('Euler Integration')
ax.set_xlabel(r'Angle $\Phi~~[°]$')
ax.set_ylabel(r'Radius $r~~[km]$')
plt.legend(loc='lower right')

plt.figure()
ax = plt.subplot(111, projection='polar')

styles = ['-', '--', '-.', ':']
for phi_0, linestyle in zip(phi_0s, styles):
    C = n(r_0) * r_0 * np.sin(phi_0)
    func = lambda r: dtheta(r, C)
    result = rk4(func, r_0 + z, t_0=r_0, dt=h)
    ax.plot(result[1], result[0] / 1000, linestyle=linestyle,
            label=r'$\Phi_0 = {0:.1f}°$'.format(np.rad2deg(phi_0)))

ax.grid(True)
ax.set_rmin(r_0 / 1000)
ax.set_rmax((r_0 + z) / 1000)
ax.set_thetamin(-5)
ax.set_thetamax(25)
ax.set_theta_offset(np.deg2rad(90))
ax.set_theta_direction(-1)
ax.set_rlabel_position(np.deg2rad(90))
ax.set_title('RK4 Integration')
ax.set_xlabel(r'Angle $\Phi~~[°]$')
ax.set_ylabel(r'Radius $r~~[km]$')
plt.legend(loc='lower right')
plt.show()
