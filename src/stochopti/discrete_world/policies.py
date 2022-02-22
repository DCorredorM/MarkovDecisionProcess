from abc import abstractmethod
from treelib import Node, Tree
from stochopti.discrete_world.space import finiteTimeSpace
import torch as pt
import numpy as np
import scipy.sparse as sp


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
    def __init__(self, space, **kwargs):
        self.space = space
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q

        if finiteTimeSpace in space.__class__.__bases__:
            self.T = space.T

        self.tree = None
        self.matrix = None

        self._handle_kwargs(kwargs)

        self.policy = {}

    def __call__(self, history):
        return self.act(history)

    def _handle_kwargs(self, kwargs):
        if 'verbose' in kwargs.keys():
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

        if 'sparse' in kwargs.keys():
            self.sparse = kwargs['sparse']
        else:
            self.sparse = False

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

    def _policy_to_matrix(self):
        ...

    def items(self):
        return self.policy.items()

    def values(self):
        return self.policy.values()

    def keys(self):
        return self.policy.keys()

    @staticmethod
    def is_deterministic(matrix):
        rows, cols = matrix.shape
        answer = True

        for r in range(rows):
            if not np.close(max(matrix[r, :]), sum(matrix[r, :])):
                answer = False
                break
        return answer


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
    def __init__(self, space, policy=None, **kwargs):
        super().__init__(space)
        self.S = list(self.S)
        if policy is not None:
            if ('from_matrix', True) in kwargs.items():
                self.matrix = policy
            else:
                self.policy = policy
                self._policy_to_matrix()
        elif ('create_random_policy', True) in kwargs.items():
            self.create_random_policy()

    def __repr__(self):

        return str(self.policy)

    def _policy_to_matrix(self):
        S = len(self.S)
        A = len(self.A)
        sparse_dict = {(self.space.A_int[a], self.space.S_int[s]): 1 for s, a in self.policy.items()}

        indices = list(sparse_dict.keys())
        values = list(sparse_dict.values())
        indices = list(zip(*indices))
        dense_shape = (S, A)
        # policy = pt.sparse_coo_tensor(indices, values, dense_shape, dtype=pt.int32)
        policy = sp.coo_matrix((values, (indices[1], indices[0])), shape=dense_shape)
        if self.sparse:
            self.matrix = policy
        else:
            self.matrix = np.asarray(policy.todense())

    def a_matrices(self):
        a_tensors = {}
        for a in self.A:
            ai = self.space.A_int[a]
            a_tensors[a] = sp.diags(self.matrix[:, ai])

        return a_tensors

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
        if state in self.policy.keys():
            return self.policy[state]
        else:
            si = self.space.S_int[state]
            probabilities = self.matrix[si, :]
            i = np.random.choice(list(range(self.space.l_A)), p=list(probabilities))
        return self.space.int_A[i]

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
        if type(policy) == type(dict):
            self.policy = policy
        elif type(policy) == type(self):
            self.policy = policy.policy
            self._policy_to_matrix()
        else:
            raise ValueError

    def create_random_policy(self):
        r_i = list(range(self.space.l_S))
        r_j = np.random.randint(low=0, high=self.space.l_A, size=self.space.l_S)
        values = np.repeat(1, self.space.l_S)
        self.matrix = np.asarray(sp.coo_matrix((values, (r_i, r_j)), shape=(self.space.l_S, self.space.l_A)).todense())


class RMSPolicy(Policy):
    """
    Representation of a deterministic markovian policy stationary.

    Deterministic markovian means that the the act method returns only an action, and that the relevant history consists
    only in the current state.

    Attributes
    __________
    policy: dict
        A dictionary with the deterministic markovian policy.

    """
    def __init__(self, space, policy=None):
        super().__init__(space)
        self.S = list(self.S)
        if policy is not None:
            self.policy = policy
            self._policy_to_matrix()
        else:
            self._create_random_policy()

    def __repr__(self):

        return str(self.policy)

    def _policy_to_matrix(self):
        S = len(self.S)
        A = len(self.A)
        sparse_dict = {(self.space.A_int[a], self.space.S_int[s]): 1 for s, a in self.policy.items()}

        indices = list(sparse_dict.keys())
        values = list(sparse_dict.values())
        indices = list(zip(*indices))
        dense_shape = (A, S)
        policy = pt.sparse_coo_tensor(indices, values, dense_shape, dtype=pt.int32)

        if self.sparse:
            self.matrix = policy
        else:
            self.matrix = policy.to_dense()

    def a_matrices(self):
        a_tensors = {}
        for a in self.A:
            ai = self.space.A_int[a]
            a_tensors[a] = sp.diags(self.matrix[:, ai])

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
        probabilities = self.policy[state]
        i = np.random.choice(list(range(len(probabilities.keys()))), p=list(probabilities.values()))

        return list(probabilities.keys())[i]

    def add_action(self, state, action, probability):
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
        self.policy[state][action] = probability

    def add_policy(self, policy):
        """
        Adds a complete policy

        Parameters
        ----------
        policy: dict
            Policy to add
        """
        if type(policy) == type(dict):
            self.policy = policy
        elif type(policy) == type(self):
            self.policy = policy.policy
            self._policy_to_matrix()
        else:
            raise ValueError

    def _create_random_policy(self):
        self.policy = {}
        for s in self.S:
            self.policy[s] = np.random.choice(list(self.adm_A(s)))

        self._policy_to_matrix()


        # dic_det_pol = {}
        # dic_rand_pol = {}
        # deterministic = True
        # for s in self.S:
        #     si = self.S_int[s]
        #     ai = max(range(self.l_A), key=lambda i: pol[si, i])
        #     if pol[si, ai] != 1:
        #         deterministic = False
        #     dic_det_pol[s] = self.int_A[ai]
        #     dic_rand_pol[s] = {}
        #     for a in self.space.adm_A(s):
        #         if pol[si, self.A_int[a]] > 0:
        #             dic_rand_pol[s][a] = pol[si, self.A_int[a]]
        #
        # if deterministic:
        #     policy = DMSPolicy(self.space, dic_det_pol)
        # else:
        #     policy = RMSPolicy(self.space, dic_rand_pol)



        # a_indices = {a: [[], []] for a in self.A}
        # a_values = {a: [] for a in self.A}
        # for s, si in self.space.S_int.items():
        #     a = self.act(s)
        #     a_indices[a][0] += [si] * self.space.l_S
        #     a_indices[a][1] += list(self.space.S_int.values())
        #     a_values[a] += [1] * self.space.l_S
        #
        # a_tensors = {}
        # for a in self.A:
        #     i = np.array(a_indices[a])
        #     a_tensors[a] = sp.coo_matrix((a_values[a], (i[0, :], i[1, :])), shape=(self.space.l_S, self.space.l_S))