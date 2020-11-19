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
def diffusion_solver(u, D, t_end=1, dt=1, dx=1):
    r = D * dt / dx ** 2
    t = np.arange(0, t_end + dt, dt)
    result = [u]

    for _ in t[1:]:
        result.append(result[-1].copy())
        result[-1][1:-1] = result[-2][1:-1] + r * (
                result[-2][:-2] - 2 * result[-2][1:-1] + result[-2][2:])

    return t, result
