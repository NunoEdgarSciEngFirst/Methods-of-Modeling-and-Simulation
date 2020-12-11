import numpy as np
import time


def timeit(func):
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
    time = np.arange(t_0, t_end + dt, dt)
    result = [[t_0, func(t_0)]]

    for t in time[1:]:
        next = result[-1][1] + dt * func(t)
        result.append([t, next])

    return np.array(result).T


@timeit
def rk4(func, t_end, t_0=0, dt=1):
    time = np.arange(t_0, t_end + dt, dt)
    result = [[t_0, func(t_0)]]

    for t in time[1:]:
        k1 = dt * func(t)
        k2 = dt * func(t + dt / 2)
        k3 = dt * func(t + dt / 2)
        k4 = dt * func(t + dt)
        next = result[-1][1] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        result.append([t, next])

    return np.array(result).T
