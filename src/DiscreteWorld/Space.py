from abc import ABCMeta, abstractmethod


class Space:
    """
    Attributes
    __________

    A: set
    S: set
    T: int
    A_s: function
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q: function
        Function that given a (state, action) tuple returns the probability distribution of the next state

    """
    def __init__(self, Actions, States, time_horizon):
        self.A = Actions
        self.S = States
        self.T = time_horizon
        self.adm_A = NotImplemented
        self.Q = NotImplemented
        self.build_admisible_actions()
        self.build_kernel()

    @abstractmethod
    def build_admisible_actions(self):
        ...

    @abstractmethod
    def build_kernel(self):
        ...