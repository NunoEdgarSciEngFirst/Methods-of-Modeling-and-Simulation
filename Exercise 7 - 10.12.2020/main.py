from library import euler, rk4, ode
import numpy as np
import matplotlib.pyplot as plt

theta_0 = 0
phi_0s = np.deg2rad([0, 30, 60, 89.9])
N_0 = 315
H_n = 7000
r_0 = 6371000
h = 100
z = 650000

N = lambda r: N_0 * np.exp((r_0 - r) / H_n)
n = lambda r: 1  # + 1E-6 * N(r)
phi = lambda r, C: np.arcsin(C / (n(r) * r))
dtheta = lambda r, C: C / (n(r) * r**2 * np.cos(phi(r, C)))

# ax = plt.subplot(111, projection='polar')
#
# styles = ['-', '--', '-.', ':', '-']
# for phi_0, linestyle in zip(phi_0s, styles):
#     C = n(r_0) * r_0 * np.sin(phi_0)
#     func = lambda r: dtheta(r, C)
#     result = rk4(func, r_0 + z, t_0=r_0, dt=h)
#     print(np.rad2deg(result[1,-1]))
#     ax.plot(result[1], (result[0] - r_0) / 1000, linestyle=linestyle,
#             label=r'$\varphi_0 = {0:.1f}°$'.format(np.rad2deg(phi_0)))
#
# ax.grid(True)
# ax.set_rmin(0)
# ax.set_rmax(z / 1000)
# ax.set_thetamin(-5)
# ax.set_thetamax(30)
# ax.set_theta_offset(np.deg2rad(90))
# ax.set_theta_direction(-1)
# ax.set_rlabel_position(np.deg2rad(90))
# ax.set_title('Euler Integration')
# ax.set_xlabel(r'Angle $\theta$')
# ax.set_ylabel(r'Radius from Surface $r-r_0~~[km]$', labelpad=-25)
# plt.legend(loc='lower right')
#
#
# plt.figure()
# ax = plt.subplot(111, projection='polar')
#
# styles = ['-', '--', '-.', ':']
# for phi_0, linestyle in zip(phi_0s, styles):
#     C = n(r_0) * r_0 * np.sin(phi_0)
#     func = lambda r: dtheta(r, C)
#     result = rk4(func, r_0 + z, t_0=r_0, dt=h)
#     ax.plot(result[1], (result[0] - r_0) / 1000, linestyle=linestyle,
#             label=r'$\varphi_0 = {0:.1f}°$'.format(np.rad2deg(phi_0)))
#
# ax.grid(True)
# ax.set_rmin(0 / 1000)
# ax.set_rmax(z / 1000)
# ax.set_thetamin(-5)
# ax.set_thetamax(30)
# ax.set_theta_offset(np.deg2rad(90))
# ax.set_theta_direction(-1)
# ax.set_rlabel_position(np.deg2rad(90))
# ax.set_title('RK4 Integration')
# ax.set_xlabel(r'Angle $\theta$')
# ax.set_ylabel(r'Radius from Surface $r-r_0~~[km]$', labelpad=-25)
# plt.legend(loc='lower right')


# ax = plt.subplot()
# styles = ['-', '--', '-.', ':']
# C = n(r_0) * r_0 * np.sin(89.9)
# func = lambda r: dtheta(r, C)
#
# result = euler(func, r_0 + z, t_0=r_0, dt=100)
# ax.plot((result[0] - r_0) / 1000, np.rad2deg(result[1]), linestyle='-',
#         label=r'Euler: $\Delta h = 100~m$')
# result = euler(func, r_0 + z, t_0=r_0, dt=1000)
# ax.plot((result[0] - r_0) / 1000, np.rad2deg(result[1]), linestyle='--',
#         label=r'Euler: $\Delta h = 1~km$')
# result = rk4(func, r_0 + z, t_0=r_0, dt=100)
# ax.plot((result[0] - r_0) / 1000, np.rad2deg(result[1]), linestyle='-.',
#         label=r'RK4: $\Delta h = 100~m$')
# result = rk4(func, r_0 + z, t_0=r_0, dt=1000)
# ax.plot((result[0] - r_0) / 1000, np.rad2deg(result[1]), linestyle=':',
#         label=r'RK4: $\Delta h = 1~km$')
# result = ode(func, r_0 + z, t_0=r_0, dt=1)
# ax.plot((result[0] - r_0) / 1000, np.rad2deg(result[1]), linestyle='-',
#         dashes=[8, 4, 2, 4, 2, 4],
#         label=r'SciPy: $\Delta h = 1~m$')
#
# ax.grid(True)
# ax.set_title('RK4 Integration')
# ax.set_ylabel(r'Angle $\theta~[^\circ]$')
# ax.set_xlabel(r'Radius from Surface $r-r_0~~[km]$')
# #ax.set_xlim([620,650])
# #ax.set_ylim([-0.1,0.8])
# plt.legend(loc='lower right')

ax = plt.subplot(111, projection='polar')
scale = 1
styles = ['-', '--', '-.', ':']
C = n(r_0) * r_0 * np.sin(np.deg2rad(89.9))
func = lambda r: dtheta(r, C)

result = euler(func, r_0 + z, t_0=r_0, dt=100)
ax.plot(result[1] * scale, (result[0] - r_0) / 1000, linestyle='-',
        label=r'Euler: $\Delta h = 100~m$')
result = euler(func, r_0 + z, t_0=r_0, dt=1000)
ax.plot(result[1] * scale, (result[0] - r_0) / 1000, linestyle='--',
        label=r'Euler: $\Delta h = 1~km$')
result = rk4(func, r_0 + z, t_0=r_0, dt=100)
ax.plot(result[1] * scale, (result[0] - r_0) / 1000, linestyle='-.',
        label=r'RK4: $\Delta h = 100~m$')
result = rk4(func, r_0 + z, t_0=r_0, dt=1000)
ax.plot(result[1] * scale, (result[0] - r_0) / 1000, linestyle=':',
        label=r'RK4: $\Delta h = 1~km$')
result = ode(func, r_0 + z, t_0=r_0, dt=1)
print(np.rad2deg(result[1][0]), np.rad2deg(result[1][-1]))
ax.plot(result[1] * scale, (result[0] - r_0) / 1000, linestyle='-',
        dashes=[8, 4, 2, 4, 2, 4],
        label=r'SciPy: $\Delta h = 1~m$')

ax.grid(True)
# ax.set_rmin(0)
# ax.set_rmax(30)
ax.set_thetamin(90)
ax.set_thetamax(270)
# ax.set_theta_offset(np.deg2rad(270))
# ax.set_theta_direction(-1)
# ax.set_xlabel(r'Angle $\theta$')
# ax.set_ylabel(r'Radius from Surface $r-r_0~~[km]$')
# ax.set_xticks(np.deg2rad(np.linspace(0, 180, 7, endpoint=True)))
# ax.set_xticklabels([r'${0:.0f}\degree$'.format(x) for x in
#                     np.linspace(0, 6, 7, endpoint=True)])
plt.legend()
# ax.set_rlabel_position(90)
plt.axis('off')
#plt.savefig('legend.png', dpi=600)
plt.show()
