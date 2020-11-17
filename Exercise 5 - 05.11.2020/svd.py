import numpy as np
from copy import deepcopy

def svd(A, y=None, tol=1E-12):

    U, w, V_T = np.linalg.svd(A)
    V = V_T.T
    w = np.diag(w)

    if y is None:
        return U, w, V

    else:
        w_inv = deepcopy(w)
        if tol is not None:
            w_inv[abs(w_inv) <= tol] = 0

        w_inv = np.linalg.pinv(w_inv)

        a = V @ np.linalg.pinv(w) @ U.T @ y
        r = np.linalg.norm(y - A @ a)
        C = V @ w_inv @ w_inv @ V_T

        return [[U, w, V], a, r, C]
