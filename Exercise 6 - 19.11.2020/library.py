import numpy as np
from numpy import linalg
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
def diffusion_solver(u, D, t_end=1, t_krit=1, dx=1, type_='explicit'):
    dt = t_krit * dx**2 / (2 * D)
    r = D * dt / dx**2
    time = np.arange(0, t_end + dt, dt)
    A = get_diff_matrix(len(u), r) if type_ != 'explicit' else None
    result = [u]

    for t in time[1:]:
        next = result[-1].copy()

        if type_ == 'explicit':
            next[1:-1] = next[1:-1] + r * (
                    next[2:] - 2 * next[1:-1] + next[:-2])
        else:
            next = A @ next

        result.append(next)

    return result


def get_diff_matrix(dim, r):
    A = r * (np.diag([-2] * dim) + np.diag([1] * (dim - 1), 1) +
             np.diag([1] * (dim - 1), -1))

    A = np.diag([1] * dim) - A

    A[0][0] = 1
    A[0][1:] = np.zeros(dim - 1)
    A[-1][-1] = 1
    A[-1][:-1] = np.zeros(dim - 1)

    return np.linalg.inv(A)
