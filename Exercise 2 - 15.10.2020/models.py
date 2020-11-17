# Python Import
import cmath
from abc import abstractmethod
from dataclasses import dataclass

# Third Party Import
import numpy as np
from numpy import linalg
from scipy.integrate import solve_ivp


class PopulationModel():

    def integrate(self, t_0, t_end, max_dt=0.1):

        solution = solve_ivp(self._integration_func,
                             [t_0, t_end],
                             [self.prey_0, self.predator_0],
                             max_step=max_dt)

        return solution.t, solution.y

    def is_stable(self, jacobian):

        trace = np.trace(jacobian)
        det = linalg.det(jacobian)
        root = cmath.sqrt(trace**2 - 4 * det)

        eigenvalue_1 = (trace + root) / 2
        eigenvalue_2 = (trace - root) / 2

        return eigenvalue_1, eigenvalue_2

    @abstractmethod
    def get_fix_points(self):
        pass

    @abstractmethod
    def _integration_func(self, t, y):
        pass


@dataclass
class LotkaVolterraModel(PopulationModel):
    '''
    d/dt prey = alpha*prey - beta*prey*predator
    d/dt predator = gamma*prey*predator - delta*predator 
    '''

    alpha: float
    beta: float
    gamma: float
    delta: float
    prey_0: float = 0
    predator_0: float = 0

    def get_fix_points(self):

        predator_fix_point = self.alpha / self.beta
        prey_fix_point = self.delta / self.gamma

        return [[0, 0], [prey_fix_point, predator_fix_point]]

    def is_stable(self):

        prey, predator = self.get_fix_points()[1]

        a = self.alpha - self.beta * predator
        b = -self.beta * prey
        c = self.gamma * predator
        d = self.gamma * prey - self.delta

        return super().is_stable([[a, b], [c, d]])

    def _integration_func(self, t, y):

        y_new = np.zeros(2)

        y_new[0] = self.alpha * y[0] - self.beta * y[0] * y[1]
        y_new[1] = self.gamma * y[0] * y[1] - self.delta * y[1]

        return y_new


@dataclass
class StrogatzModel(PopulationModel):
    '''
    d/dt prey = alpha*(1 - predator/predator_capacity)*prey
    d/dt predator = -beta*(1 - prey/prey_capacity)*predator 
    '''

    alpha: float
    beta: float
    prey_capacity: float
    predator_capacity: float
    prey_0: float = 0
    predator_0: float = 0

    def get_fix_points(self):

        return [[0, 0], [self.prey_capacity, self.predator_capacity]]

    def is_stable(self):

        prey, predator = self.get_fix_points()[1]

        a = self.alpha * (1 - predator / self.predator_capacity)
        b = -self.alpha * prey / self.predator_capacity
        c = self.beta * predator / prey
        d = -self.beta * (1 - prey / self.prey_capacity)

        return super().is_stable([[a, b], [c, d]])

    def _integration_func(self, t, y):

        y_new = np.zeros(2)

        y_new[0] = self.alpha * (1 - y[1] / self.predator_capacity) * y[0]
        y_new[1] = -self.beta * (1 - y[0] / self.prey_capacity) * y[1]

        return y_new
