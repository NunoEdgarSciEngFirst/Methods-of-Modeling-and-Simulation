import numpy as np
from scipy import stats


def polynomial_fit(x, y, sigma, order=1):

    funcs = get_polynomial_funcs(order=order)

    return fit(np.array(x), np.array(y), sigma, funcs)


def get_polynomial_funcs(order=1):

    return [eval('lambda x: x**{}'.format(power))
            for power in range(order + 1)]


def fit(x, y, sigma, funcs):

    # Design Matrix A
    A = get_design_matrix(x, sigma, funcs)
    N, M = A.shape

    # Generalized Inverse Matrix
    A_inv = np.linalg.inv((A.T.dot(A))).dot(A.T)

    # Covariance and Correlation Matrices
    C = np.linalg.inv(A.T.dot(A))
    R = np.ones((len(funcs), len(funcs)))
    for i in range(len(funcs)):
        for j in range(len(funcs)):
            R[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    if sigma == 0:
        parameter = A_inv.dot(y)
        p_val, C, R = [None, None, None]

    else:
        # Optimal Parameters
        y_w = np.array(y) / sigma
        parameter = A_inv.dot(y_w)

        # Chi^2
        p_val = get_p_val(x, y, sigma, parameter, funcs)

    return parameter, p_val, C, R


def get_design_matrix(x, sigma, funcs):

    if sigma == 0:
        sigma = 1

    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = sigma * np.ones((len(x), 1))

    else:
        sigma = np.reshape(sigma, (len(sigma), 1))

    x, funcs = map(np.array, (x, funcs))

    A = np.zeros((len(x), len(funcs)))

    for col, func in enumerate(funcs):
        A[:, col] = func(x)

    A = A * (1 / sigma)

    return A


def get_p_val(x, y, sigma, parameter, funcs, alpha=0.95):

    y_mod = eval_result(x, parameter, funcs)

    chi2 = np.sum(((y - y_mod) / sigma)**2)
    deg_of_freedom = len(x) - len(funcs)
    p_val = stats.chi2.sf(chi2, df=deg_of_freedom)
    chi2_test = stats.chi2.ppf(q=alpha, df=deg_of_freedom)

    return p_val, chi2, chi2_test


def eval_result(x, parameter, funcs):

    y = []

    for p, func in zip(parameter, funcs):
        y.append(p * func(x))

    return np.sum(np.array(y), axis=0)


def param_distribution(C, contour_levels=[], num_sigma=3, res=50):

    # Set variables
    C_inv = np.linalg.inv(C)
    stds = np.sqrt(np.diagonal(C))
    m = len(stds)

    # Mesh grids for each parameter
    delta_p_i = [np.linspace(-num_sigma * std, num_sigma * std, res)
                 for std in stds]
    meshgrids = np.meshgrid(*delta_p_i)

    # Stack them in the last axis
    delta_a = np.stack(meshgrids, axis=m)

    # Generalised dot product in the exponent
    exponent = np.einsum('...j,...j', delta_a,
                         np.einsum('ij,...j', C_inv, delta_a))

    pre_factor = np.sqrt(np.linalg.det(C_inv) / (2 * np.pi)**m)
    f_a = lambda x: pre_factor * np.exp(-x / 2)

    # Return contour lines for certain alpha or chi^2 values
    levels = []
    for level in contour_levels:
        if level < 1:
            level = stats.chi2.ppf(q=level, df=m)

        levels.append(f_a(level))

    return meshgrids, f_a(exponent), levels
