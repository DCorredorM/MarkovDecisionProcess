from abc import abstractmethod

from DiscreteWorld.Policies import Policy, DMPolicy
from DiscreteWorld.Space import Space
from DiscreteWorld.Reward import Reward


class MDP:
    """
    Abstract class for markov decision process.

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
    r
        Function that given a (t, state, action) tuple returns the reward.
    """
    def __init__(self, space: Space, reward: Reward):
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q
        self.T = space.T
        self.r = reward.reward

    @abstractmethod
    def value_of_policy(self, policy: Policy, history):
        """
        Abstract method that when implemented values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value
        history: History
            History

        Returns
        -------
        float
            The value of the policy
        """
        ...

    @abstractmethod
    def optimal_value(self, t, state):
        """
        Abstract method that computes the value function for the problem and creates the optimal policy.

        Parameters
        ----------
        t: int
            The time for which the optimal is computed.
        state
            The state for which the optimal is computed.

        Returns
        -------
        float
            The value function for the given time and state.

        """
        ...


class DeterministicMarkovian(MDP):
    """
    Implements an MDP in which the policy is a deterministic markovian policy


    Attributes
    __________

    v: dict
        Stores the value function for each time state combination
    """

    def __init__(self, space: Space, reward: Reward):
        super().__init__(space, reward)
        self.policy = DMPolicy(space)
        self.v = dict()
        self.a_policy = dict()

    def value_of_policy(self, policy: DMPolicy, history):
        """
        Values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value
        history: History
            History

        Returns
        -------
        float
            The value of the policy
        """
        t = len(history)
        if t == self.T:
            return self.r(t, history[-1])
        else:
            xt, ut = history[-1]
            u = policy((t, xt))
            v = self.r(t, xt, ut) + sum(
                self.Q(xt, u)[x] *
                self.value_of_policy(policy, history=history + [(x, u)])
                for x in self.S)
            return v

    def optimal_value(self, t, state):
        """
        Computes the value function for the problem and creates the optimal policy.

        Parameters
        ----------
        t: int
            The time for which the optimal is computed.
        state
            The state for which the optimal is computed.

        Returns
        -------
        float
            The value function for the given time and state.
        """
        if t < self.T:
            if (t, state) not in self.v.keys():
                def sup(u):
                    return self.r(t, state, u) + sum(
                        p * self.optimal_value(t + 1, y)
                        for y, p in self.Q(state, u).items())
                pairs = [(u, sup(u)) for u in self.adm_A(state)]
                u, v = max(pairs, key=lambda x: x[1])
                self.a_policy[t, state] = u
                self.v[t, state] = v
            return self.v[t, state]
        else:
            if (t, state) not in self.v.keys():
                self.v[t, state] = self.r(t, state)
                self.a_policy[t, state] = None
            return self.v[t, state]

    def solve(self, initial_state):
        """
        Starts the recursion at a given initial state and solves the MDP.

        Parameters
        ----------
        initial_state: state
            State in which the recursion is initialized

        Returns
        -------
        dict
            The optimal policy.
        float
            The value of the optimal policy.
        """
        v = self.optimal_value(0, initial_state)
        self.policy.add_policy(self.a_policy, initial_state)
        return self.policy, v
