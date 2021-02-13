from abc import ABCMeta, abstractmethod
from DiscreteWorld.Space import Space


class Reward:
    def __init__(self, space: Space):
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q
        self.T = space.T

    @abstractmethod
    def reward(self, t, state, action=None):
        """
        Computes the reward for a given (time, state, action) triple
        """
        ...
