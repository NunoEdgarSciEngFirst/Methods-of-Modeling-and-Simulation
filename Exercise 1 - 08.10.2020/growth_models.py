# Python Import
from abc import abstractmethod

# Third Party Import
import numpy as np


class AbstractGrowthModel():

    def __init__(self, r, t0=0, x0=0):
        """Args:
            r (float): Growth rate in units of time steps.
            t0 (float): Zero point for the time in units of time.
            x0 (float): Population at t0.
        """

        self.r = r
        self.t0 = t0
        self.x0 = x0

        x = x0

    @abstractmethod
    def population_at(self, t):
        """Args:
            t (float): Time at which the population is to be returned.

            Returns:
            (float): The population at time t.
        """
        pass


class LinearGrowthModel(AbstractGrowthModel):

    def population_at(self, t):

        t -= self.t0
        x = self.x0 + self.r * t

        return x


class ExponentialGrowthModel(AbstractGrowthModel):

    def population_at(self, t):

        t -= self.t0
        x = self.x0 * np.exp(self.r * t)

        return x


class LogisticGrowthModel(AbstractGrowthModel):

    def __init__(self, K, *args, **kwargs):
        """Args:
            K (float): Capacity.
        """

        super().__init__(*args, **kwargs)

        self.K = K

    def population_at(self, t):

        t -= self.t0
        x = self.K / \
            (1 + (self.K / self.x0 - 1) * np.exp(-self.r * t))

        return x


class GoalSeekingGrowthModel(AbstractGrowthModel):

    def __init__(self, Z, *args, **kwargs):
        """Args:
            Z (float): Goal.
        """

        super().__init__(*args, **kwargs)

        self.Z = Z

    def population_at(self, t):

        t -= self.t0
        x = self.Z * (1 - np.exp(-self.r * t)) + \
            self.x0 * np.exp(-self.r * t)

        return x


class GoalSettingGrowthModel(AbstractGrowthModel):

    def __init__(self, I, *args, **kwargs):
        """Args:
            I (float): Immigration.
        """

        super().__init__(*args, **kwargs)

        self.I = I

    def population_at(self, t):

        t -= self.t0
        x = -self.I / self.r + (self.x0 + self.I / self.r) * np.exp(self.r * t)

        return x
