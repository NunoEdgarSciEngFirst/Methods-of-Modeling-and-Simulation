# Third Party Import
import numpy as np
import matplotlib.pyplot as plt

# Own Import
from growth_models import *

####################### Linear Growth #######################

fig, ax = plt.subplots(2, 1)
fig.suptitle('Linear Growth', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

t0, dt, t_end = 0, 1, 100
t = np.array(range(t0, t_end, dt))

r = 1
for x0 in [0, 10, 50, 100]:
    model = LinearGrowthModel(r=r, x0=x0)
    ax[0].plot(t, model.population_at(t), label=fr'$x_0 = {x0}$')

ax[0].legend()
ax[0].grid()
ax[0].set(title=fr'Influence of initial population $x_0$ ($r={r}$)')

x0 = 80
for r in [0, 0.5, 1, 2]:
    model = LinearGrowthModel(r=r, x0=x0)
    ax[1].plot(t, model.population_at(t), label=fr'$r = {r}$')

ax[1].legend()
ax[1].grid()
ax[1].set(title=fr'Influence of growth rate $r$ ($x_0={x0}$)')

fig.tight_layout()

####################### Exponential Growth #######################

fig, ax = plt.subplots(2, 1)
fig.suptitle('Exponential Growth', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

t0, dt, t_end = 0, 1, 100
t = np.array(range(t0, t_end, dt))

r = 0.03
for x0 in [0, 10, 50, 100]:
    model = ExponentialGrowthModel(r=r, x0=x0)
    ax[0].plot(t, model.population_at(t), label=fr'$x_0 = {x0}$')

ax[0].legend()
ax[0].grid()
ax[0].set(title=fr'Influence of initial population $x_0$ ($r={r}$)')

x0 = 80
for r in [0.01, 0.03, 0.05, 0.06]:
    model = ExponentialGrowthModel(r=r, x0=x0)
    ax[1].plot(t, model.population_at(t), label=fr'$r = {r}$')

ax[1].legend()
ax[1].grid()
ax[1].set(title=fr'Influence of growth rate $r$ ($x_0={x0}$)')

fig.tight_layout()

####################### Logistic Growth #######################

fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('Logistic Growth', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

t0, dt, t_end = 0, 1, 300
t = np.array(range(t0, t_end, dt))

x0 = 80
r = 0.03
for K in [100, 200, 250, 300]:
    model = LogisticGrowthModel(r=r, K=K, x0=x0)
    ax[0].plot(t, model.population_at(t), label=fr'$K = {K}$')

ax[0].legend()
ax[0].grid()
ax[0].set(title=fr'Influence of capacity $K$ ($r={r}$; $x_0={x0}$)')

x0 = 80
K = 200
for r in [0.01, 0.03, 0.05, 0.1]:
    model = LogisticGrowthModel(r=r, K=K, x0=x0)
    ax[1].plot(t, model.population_at(t), label=fr'$r = {r}$')

ax[1].legend()
ax[1].grid()
ax[1].set(title=fr'Influence of growth rate $r$ ($K={K}$; $x_0={x0}$)')

fig.tight_layout()

####################### Goal Seeking Growth #######################

fig, ax = plt.subplots(2, 1)
fig.suptitle('Goal Seeking Growth', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

t0, dt, t_end = 0, 1, 300
t = np.array(range(t0, t_end, dt))

x0 = 80
r = 0.03
for Z in [200, 250, 250, 350]:
    model = GoalSeekingGrowthModel(r=r, Z=Z, x0=x0)
    ax[0].plot(t, model.population_at(t), label=fr'$Z = {Z}$')

ax[0].legend()
ax[0].grid()
ax[0].set(title=fr'Influence of goal $Z$ ($r={r}$; $x_0={x0}$)')

r = 0.03
Z = 250
for x0 in [150, 250, 350, 500]:
    model = GoalSeekingGrowthModel(r=r, Z=Z, x0=x0)
    ax[1].plot(t, model.population_at(t), label=fr'$x_0 = {x0}$')

ax[1].legend(loc='upper right')
ax[1].grid()
ax[1].set(title=fr'Influence of initial population $x_0$ ($Z={Z}$; $r={r}$)')

fig.tight_layout()

####################### Goal Setting Growth #######################

fig, ax = plt.subplots(2, 1)
fig.suptitle('Goal Setting Growth', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

t0, dt, t_end = 0, 1, 300
t = np.array(range(t0, t_end, dt))

x0 = 80
r = -0.01
for I in [-0.1, 0, 1, 2]:
    model = GoalSettingGrowthModel(r=r, I=I, x0=x0)
    population = model.population_at(t)
    ax[0].plot(t[population >= 0], population[population >= 0], label=fr'$I = {I}$')

ax[0].legend()
ax[0].grid()
ax[0].set(title=fr'Influence of immigration $I$ ($r={r}$; $x_0={x0}$)')

x0 = 80
I = 1
for r in [-0.03, -0.02, -0.01, 0.001]:
    model = GoalSettingGrowthModel(r=r, I=I, x0=x0)
    ax[1].plot(t, model.population_at(t), label=fr'$r = {r}$')

ax[1].legend()
ax[1].grid()
ax[1].set(title=fr'Influence of growth rate $r$ ($I={I}$; $x_0={x0}$)')

fig.tight_layout()

####################### Comparison #######################

fig, ax = plt.subplots()
fig.suptitle('Comparison', fontsize=16)
fig.text(0.5, 0.01, r'Time $t$ [y]', ha='center')
fig.text(0.01, 0.5, r'Population $x$ [1]', va='center', rotation='vertical')

ax2 = ax.twinx()
color = 'tab:purple'

t0, dt, t_end = 0, 1, 300
t = np.array(range(t0, t_end, dt))

r, x0 = 0.03, 80
model = ExponentialGrowthModel(r=r, x0=x0)
pl1 = ax2.plot(t, model.population_at(t), label=fr'Exponential ($x_0={x0}$; $r={r}$)', color=color)

r, x0 = 1, 80
model = LinearGrowthModel(r=r, x0=x0)
pl2 = ax.plot(t, model.population_at(t), label=fr'Linear ($x_0={x0}$; $r={r}$)')

r, K, x0 = 0.03, 200, 10
model = LogisticGrowthModel(r=r, K=K, x0=x0)
pl3 = ax.plot(t, model.population_at(t), label=fr'Logistic ($x_0={x0}$; $r={r}$; $K={K}$)')

r, Z, x0 = 0.03, 250, 10
model = GoalSeekingGrowthModel(r=r, Z=Z, x0=x0)
pl4 = ax.plot(t, model.population_at(t), label=fr'Goal Seeking ($x_0={x0}$; $r={r}$; $Z={Z}$)')

r, I, x0 = -0.01, 1, 10
model = GoalSettingGrowthModel(r=r, I=I, x0=x0)
pl5 = ax.plot(t, model.population_at(t), label=fr'Goal Setting ($x_0={x0}$; $r={r}$; $I={I}$)')

plots = pl1 + pl2 + pl3 + pl4 + pl5
labels = [p.get_label() for p in plots]
ax.legend(plots, labels)
ax.grid()
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
