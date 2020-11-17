# Third Party Import
import matplotlib.pyplot as plt
import numpy as np

# Own Import
from models import LotkaVolterraModel, StrogatzModel

'''
####################### First Model #######################

model_1 = LotkaVolterraModel(0.5, 0.008, 0.008, 0.8, 50, 25)

print(model_1.get_fix_points())
print(model_1.is_stable())

t, population = model_1.integrate(0, 50)

fig, ax = plt.subplots()
fig.suptitle(fr'Time Diagram - System 1', fontsize=16)

pl1 = ax.plot(t, population[0, :], label=fr'Hare')
pl2 = ax.plot(t, population[1, :], label=fr'Lynx')
ax.axhline(y=np.mean(population[0, :]), color=pl1[0].get_color(),
           linestyle='--', label=fr'Mean Hare')
ax.axhline(y=np.mean(population[1, :]), color=pl2[0].get_color(),
           linestyle='--', label=fr'Mean Lynx')

ax.legend()
ax.grid()
ax.set(xlabel=fr'Time [a.u.]', ylabel=fr'Population [1]')

t, population = model_1.integrate(0, 11)

fig, ax = plt.subplots()
fig.suptitle(fr'Phase Diagram - System 1', fontsize=16)

ax.plot(population[1, :], population[0, :], label=fr'Hare over Lynx')

fix_population = model_1.get_fix_points()[1]
ax.plot(fix_population[1], fix_population[0],
        color='k', marker='x', markersize=9)
ax.axvline(x=fix_population[1], color='k', linestyle='--', label=fr'Fix Point')
ax.axhline(y=fix_population[0], color='k', linestyle='--')

ax.legend()
ax.grid()
ax.set(xlabel='Lynx Population [1]', ylabel='Hare Population [1]')


####################### Second Model #######################

fig_1, ax_1 = plt.subplots()
fig_1.suptitle(fr'Phase Diagram - System 2 & 3', fontsize=16)

ax_1.grid()
ax_1.set(xlabel='Lynx Population [1]', ylabel='Hare Population [1]')

for inital_pop in [[40, 10]]:#, [150, 12]

    model_2 = LotkaVolterraModel(0.3, 0.025, 0.0015, 0.2, *inital_pop)
    t, population = model_2.integrate(0, 100)

    print(model_2.get_fix_points())
    print(model_2.is_stable())

    fig, ax = plt.subplots()
    fig.suptitle(fr'Time Diagram - System 3', fontsize=16)

    pl1 = ax.plot(t, population[0, :], label=fr'Hare')
    pl2 = ax.plot(t, population[1, :], label=fr'Lynx')
    ax.axhline(y=np.mean(population[0, :]), color=pl1[0].get_color(),
               linestyle='--', label=fr'Mean Hare')
    ax.axhline(y=np.mean(population[1, :]), color=pl2[0].get_color(),
               linestyle='--', label=fr'Mean Lynx')

    
    ax.grid()
    ax.set(xlabel='Time [a.u.]', ylabel='Population [1]')
    ax.legend(loc='upper right')
    t, population = model_2.integrate(0, 30)
    ax_1.plot(population[1, :], population[0, :],
              label=fr'$H_0={inital_pop[0]}$, $L_0={inital_pop[1]}$')

fix_population = model_2.get_fix_points()[1]
ax_1.plot(fix_population[1], fix_population[0],
          color='k', marker='x', markersize=9)
ax_1.axvline(x=fix_population[1], color='k', linestyle='--', label=fr'Fix Point')
ax_1.axhline(y=fix_population[0], color='k', linestyle='--')
ax_1.legend()

'''
####################### Strogatz Model #######################

model_3 = StrogatzModel(0.04, 0.04, 50, 10, 10, 2)

print(model_3.get_fix_points())
print(model_3.is_stable())

t, population = model_3.integrate(0, 500)

fig, ax = plt.subplots()
fig.suptitle('Time Diagram - System 4', fontsize=16)

pl1 = ax.plot(t, population[0, :], label=fr'Hare')
pl2 = ax.plot(t, population[1, :], label=fr'Lynx')
#ax.axhline(y=np.mean(population[0, :]), color=pl1[0].get_color(),
#           linestyle='--', label=fr'Mean Hare')
#ax.axhline(y=np.mean(population[1, :]), color=pl2[0].get_color(),
#           linestyle='--', label=fr'Mean Lynx')
ax.axvline(x=90.975, color='k', linestyle='dashed', label=fr'$x_h$ reaches $c_h$')
ax.axhline(y=50, color=pl1[0].get_color(),
           linestyle='dotted', label=fr'Hare Capacity $c_h$')
ax.axhline(y=10, color=pl2[0].get_color(),
           linestyle='dotted', label=fr'Lynx Capacity $c_l$')


ax.legend()
ax.grid()
ax.set(xlabel=fr'Time [a.u.]', ylabel=fr'Population [1]')

t, population = model_3.integrate(0, 250)

fig, ax = plt.subplots()
fig.suptitle(fr'Phase Diagram - System 4', fontsize=16)

ax.plot(population[1, :], population[0, :], label=fr'Hare over Lynx')

fix_population = model_3.get_fix_points()[1]
ax.plot(fix_population[1], fix_population[0],
        color='k', marker='x', markersize=9)
ax.axvline(x=fix_population[1], color='k', linestyle='--', label=fr'Fix Point')
ax.axhline(y=fix_population[0], color='k', linestyle='--')

ax.legend()
ax.grid()
ax.set(xlabel='Lynx Population [1]', ylabel='Hare Population [1]')

plt.show()
