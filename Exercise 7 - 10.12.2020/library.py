import time
import numpy as np
from scipy.integrate import odeint


def timeit(func):
    '''Simple auxiliary decorator to time the decorated function.'''

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print('Execution Time ({0}): {1:.5f} seconds'.format(
            func.__name__, end_time - start_time))

        return result

    return wrapper


@timeit
def euler(func, t_end, t_0=0, dt=1):
    '''Integrates the function "func" from "t_0" to "t_end" numerically
    using the Euler method with a step size of "dt."'''
    time = np.arange(t_0, t_end + dt, dt)
    result = [[t_0, func(t_0)]]

    for t in time[1:]:
        next = result[-1][1] + dt * func(t)
        result.append([t, next])

    return np.array(result).T


@timeit
def rk4(func, t_end, t_0=0, dt=1):
    '''Integrates the function "func" from "t_0" to "t_end" numerically
        using the RK4 method with a step size of "dt."'''
    time = np.arange(t_0, t_end + dt, dt)
    result = [[t_0, func(t_0)]]

    for t in time[1:]:
        k1 = func(t)
        k2 = func(t + dt / 2)
        k3 = func(t + dt / 2)
        k4 = func(t + dt)
        next = result[-1][1] + dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
        result.append([t, next])

    return np.array(result).T


@timeit
def ode(func, t_end, t_0=0, dt=1):
    '''Wrapper method for the '''
    time = np.arange(t_0, t_end + dt, dt)
    f = lambda y, t: func(t)
    result = np.stack((time, odeint(f, f(0, t_0), time).flatten()))

    return result
