import numpy as np
from svd import svd

y = np.array([4, 3, 7]).T
A = np.array([[1, -2, 3], [2, -1, 4], [-1, -4, 1]])

# A_red = [[1,-2,3],[0,3,-2],[0,0,0]] -> rank 2

####################### Rank #######################

print('Rank of A:\t\t{0:d}'.format(np.linalg.matrix_rank(A)))

####################### SVD #######################

matrices, a, error, C = svd(A, y=y)
U, w, V = matrices

a_lst, *_, s = np.linalg.lstsq(A, y)

print('Singular Values of A:\t{}'.format(list(np.diag(w))))
print('Solution a:\t\t{}'.format(list(a)))
print('Lstsq Solution a:\t{}'.format(list(a_lst)))
print('Residual Error:\t\t{} ({})'.format(error, error/np.linalg.norm(a)))
print('Covariance Matrix:\n{}'.format(C))
print('U Matrix:\n{}'.format(U))
print('D Matrix:\n{}'.format(w))
print('V Matrix:\n{}'.format(V))


print('U.T@U Matrix:\n{}'.format(U.T@U))
print('U@U.T Matrix:\n{}'.format(U@U.T))
print('V.T@V Matrix:\n{}'.format(V.T@V))
print('V@V.T Matrix:\n{}'.format(V@V.T))