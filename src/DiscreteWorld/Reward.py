from abc import abstractmethod, ABC
from DiscreteWorld.Space import Space, finiteTimeSpace, infiniteTimeSpace


class Reward:
    """
    Models the rewards through time.

    Attributes
    __________
    A: set
        Set of accions
    S: set
        Set of states

    Methods
    _______
    adm_A
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q
        Function that given a (state, action) tuple returns the probability distribution of the next state
    """
    def __init__(self, space: Space):
        self.space = space
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q

    @abstractmethod
    def reward(self, t, state, action=None):
        """
        Abstract method that computes the reward for a given (time, state, action) triple
        """
        ...


class finiteTimeReward(Reward, ABC):
    """
    Models the rewards through time.

    Attributes
    __________
    A: set
        Set of accions
    S: set
        Set of states
    T: int
        Time Horizon

    Methods
    _______
    adm_A
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q
        Function that given a (state, action) tuple returns the probability distribution of the next state
    """
    def __init__(self, space: finiteTimeSpace):
        super().__init__(space)
        self.T = space.T


class infiniteTimeReward(Reward, ABC):
    """
    Models the rewards through time.

    Attributes
    __________
    A: set
        Set of accions
    S: set
        Set of states

    Methods
    _______
    adm_A
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q
        Function that given a (state, action) tuple returns the probability distribution of the next state
    """
    def __init__(self, space: finiteTimeSpace):
        super().__init__(space)
