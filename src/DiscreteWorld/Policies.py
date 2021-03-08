from abc import abstractmethod
from treelib import Node, Tree
from DiscreteWorld.Space import finiteTimeSpace


class Policy:
    """
    Abstract class for policy

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
    def __init__(self, space):
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q

        if finiteTimeSpace in space.__class__.__bases__:
            self.T = space.total_time

        self.tree = None

    def __call__(self, history):
        return self.act(history)

    @abstractmethod
    def act(self, history):
        """
        Abstract method that depending on the implementation returns an action given a history

        Parameters
        ----------
        history: iterable of tuples
            An iterable of (state, action) pairs representing the relevant history for the policy to act.
            history[-1] is the current state.

        Returns
        -------
            The action or a probability distribution on the actions for the given history
        """
        ...


class DMNSPolicy(Policy):
    """
    Representation of a deterministic markovian non-stationary policy.

    Deterministic markovian means that the the act method returns only an action, and that the relevant history consists
    only in the current state.

    Attributes
    __________
    policy: dict
        A dictionary with the deterministic markovian policy.

    tree: treelib.Tree
        a tree object for visualization purposes.
    """
    def __init__(self, space):
        super().__init__(space)
        self.policy = {}

    def __repr__(self):
        st = str(self.tree.show())
        return st

    def _create_tree(self, initial_state):
        """
        A tree object for visualization purposes is created.

        Parameters
        ----------
        initial_state: state
            An initial state
        """
        self.tree = Tree()

        self.tree.add_node(
            Node(f'({0}:{initial_state}:{self.policy[(0, initial_state)]})',
                 f'({0}:{initial_state}:{self.policy[(0, initial_state)]})')
        )

        def add_sons(s, t):
            a = self.policy[(t, s)]
            if t == self.T:
                for st in self.Q(s, a):
                    n = Node(f'({t + 1}:{st})', f'({t + 1}:{st})')
                    self.tree.add_node(node=n, parent=f'({t}:{s}:{a})')

            elif t < self.T - 1:
                for st in self.Q(s, a):
                    at = self.policy[(t + 1, st)]
                    n = Node(f'({t + 1}:{st}:{at})', f'({t + 1}:{st}:{at})')

                    if n.identifier not in map(lambda x: x.identifier, self.tree.all_nodes()):
                        self.tree.add_node(node=n, parent=f'({t}:{s}:{a})')
                        add_sons(st, t + 1)

        add_sons(initial_state, 0)

    def act(self, history):
        """

        Parameters
        ----------
        history

        Returns
        -------

        """
        time, state = history
        return self.policy[(time, state)]

    def add_action(self, time, state, action):
        """
        Add an action to the policy.

        Parameters
        ----------
        time: int
            Time
        state: State
            State
        action: Action
            Action
        """
        # assert state in self.S and action in self.A
        self.policy[time, state] = action

    def add_policy(self, policy, initial_state=None):
        """
        Adds a complete policy

        Parameters
        ----------
        policy: dict
            Policy to add
        initial_state: state
            (Optional)
            If the initial state is given, a tree for visualization purposes is created.
        """
        self.policy = policy
        if initial_state is not None:
            self._create_tree(initial_state)


class DMSPolicy(Policy):
    """
    Representation of a deterministic markovian policy stationary.

    Deterministic markovian means that the the act method returns only an action, and that the relevant history consists
    only in the current state.

    Attributes
    __________
    policy: dict
        A dictionary with the deterministic markovian policy.

    """
    def __init__(self, space):
        super().__init__(space)
        self.policy = {}
        self.S = list(self.S)

    def __repr__(self):

        return str(self.policy)

    def act(self, state):
        """

        Parameters
        ----------
        state
            the state

        Returns
        -------
            the action
        """

        return self.policy[state]

    def add_action(self, state, action):
        """
        Add an action to the policy.

        Parameters
        ----------
        state: State
            State
        action: Action
            Action
        """
        # assert state in self.S and action in self.A
        self.policy[state] = action

    def add_policy(self, policy):
        """
        Adds a complete policy

        Parameters
        ----------
        policy: dict
            Policy to add
        """
        self.policy = policy
