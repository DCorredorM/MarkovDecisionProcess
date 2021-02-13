from abc import ABCMeta, abstractmethod
from itertools import product
from treelib import Node, Tree


class Policy:
    def __init__(self, space):
        self.A = space.A
        self.S = space.S
        self.A_s = space.adm_A
        self.Q = space.Q
        self.T = space.T
        self.tree = None


    @abstractmethod
    def act(self, history):
        ...


class DMPolicy(Policy):
    def __init__(self, space):
        super().__init__(space)
        self.policy = {}

    def _create_tree(self, initial_state):
        self.tree = Tree()

        a = self.policy[(0, initial_state)]
        n = Node(f'({0}:{initial_state}:{a})', f'({0}:{initial_state}:{a})')
        self.tree.add_node(n)

        def add_sons(s, t):
            a = self.policy[(t,s)]
            if t == self.T:
                for st in self.Q(s, a):
                    try:
                        n = Node(f'({t + 1}:{st})', f'({t + 1}:{st})')
                        self.tree.add_node(node=n, parent=f'({t}:{s}:{a})')
                    except:
                        pass
            elif t < self.T - 1:
                for st in self.Q(s, a):
                    try:
                        at = self.policy[(t + 1, st)]
                        n = Node(f'({t + 1}:{st}:{at})', f'({t + 1}:{st}:{at})')
                        self.tree.add_node(node=n, parent=f'({t}:{s}:{a})')
                        add_sons(st, t + 1)
                    except:
                        pass
        add_sons(initial_state, 0)

    def __repr__(self):
        st = self.tree.show()
        return ''

    def act(self, time, state):
        """

        Parameters
        ----------
        time : int            
        state 

        Returns
        -------

        """
        return self.policy[(time, state)]

    def add_action(self, time, state, action):
        """

        Parameters
        ----------
        state
        action

        Returns
        -------

        """
        # assert state in self.S and action in self.A
        self.policy[time, state] = action


