from abc import abstractmethod, ABC


class Space:
    """
    Abstract class for the space of an discrete MDP.

    Attributes
    _________
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
    def __init__(self, actions, states):
        self.A = actions
        self.S = states
        self.adm_A = NotImplemented
        self.Q = NotImplemented
        self.build_admisible_actions()
        self.build_kernel()

    @abstractmethod
    def build_admisible_actions(self):
        """
        Abstract method that builds the admisible actions function and stores it in self.adm_A
        """
        ...

    @abstractmethod
    def build_kernel(self):
        """
        Abstract method that builds the stochastic kernel and stores it in self.Q
        """
        ...


class finiteTimeSpace(Space, ABC):
    """
    Abstract class for the space of an discrete MDP.

    Attributes
    _________
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
    def __init__(self, actions, states, time_horizon):
        super().__init__(actions, states)
        self.T = time_horizon


class infiniteTimeSpace(Space, ABC):
    """
    Abstract class for the space of an discrete MDP.

    Attributes
    _________
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
    def __init__(self, actions, states):
        super().__init__(actions, states)
