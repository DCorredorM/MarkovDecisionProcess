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
        self.l_A = len(self.A)
        self.S = states
        self.l_S = len(self.S)
        self.S_int = {s: i for i, s in enumerate(self.S)}
        self.int_S = {v: k for k, v in self.S_int.items()}

        if self.A is None:
            self._build_A()
        self.A_int = {a: i for i, a in enumerate(self.A)}
        self.int_A = {v: k for k, v in self.A_int.items()}

        self.adm_A = NotImplemented
        self.Q = NotImplemented
        self.build_admisible_actions()
        self.build_kernel()

    def _build_A(self):
        A = set()
        for s in self.S:
            A = A.union(set(self.adm_A(s)))
        self.A = A

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

    @abstractmethod
    def reward(self, state, action=None):
        """
        Abstract method that computes the reward for a given (time, state, action) triple
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
