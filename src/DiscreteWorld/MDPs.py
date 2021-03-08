from abc import abstractmethod
import numpy as np
from time import time

from DiscreteWorld.Policies import Policy, DMSPolicy, DMNSPolicy
from DiscreteWorld.Space import finiteTimeSpace
from DiscreteWorld.Reward import finiteTimeReward
from Utilities.counters import Timer, Tally

class MDP:
    """
    Abstract class for markov decision process in infinite time.

    Attributes
    _________
    A: set
        Set of actions
    S: set
        Set of states

   Methods
    _______
    adm_A
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q
        Function that given a (state, action) tuple returns the probability distribution of the next state
    r
        Function that given a (t, state, action) tuple returns the reward.
    """
    def __init__(self, space: finiteTimeSpace, reward: finiteTimeReward):
        self.space = space
        self.reward = reward

        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q
        self.r = reward.reward

    @abstractmethod
    def policy_valuation(self, policy: Policy):
        """
        Abstract method that when implemented values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value

        Returns
        -------
        float
            The value of the policy
        """
        ...

    @abstractmethod
    def optimal_value(self):
        """
        Abstract method that computes the value function for the problem and creates the optimal policy.

        Returns
        -------
        float
            The value function for the given time and state.

        """
        ...


class finiteTime(MDP):
    """
    Implements an MDP in which the policy is a deterministic markovian policy


    Attributes
    __________

    v: dict
        Stores the value function for each time state combination
    """

    def __init__(self, space: finiteTimeSpace, reward: finiteTimeReward):
        super().__init__(space, reward)
        self.T = space.T
        self.policy = DMNSPolicy(space)
        self.v = dict()
        self.a_policy = dict()

    def policy_valuation(self, policy: DMNSPolicy, history):
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
                self.policy_valuation(policy, history=history + [(x, u)])
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


class infiniteTime(MDP):
    """
    Implements an MDP in which the policy is a deterministic markovian policy


    Attributes
    __________

    v: dict
        Stores the value function for each time state combination
    """

    def __init__(self, space: finiteTimeSpace, reward: finiteTimeReward, _lambda: float):
        super().__init__(space, reward)
        self.S = list(self.S)
        self.S_int = {s: i for i, s in enumerate(self.S)}
        self.policy = DMSPolicy(space)
        self.v = np.zeros(shape=(len(self.S), 1))
        self._lambda = _lambda
        self.a_policy = dict()

        self._Ps = dict()
        self._rs = dict()

        self.computing_times = {'preprocess': Timer('preprocess'),
                                'VI': Timer('VI'),
                                'JAC': Timer('JAC'),
                                'GS': Timer('GS')}

        self.iteration_counts = {'VI': Tally('VI'),
                                 'JAC': Tally('JAC'),
                                 'GS': Tally('GS')}

        self.computing_times['preprocess'].start()
        self._preprocess()
        self.computing_times['preprocess'].stop()

    def _preprocess(self):
        for s in self.S:
            self._build_r_P(s)

    def build_transition_matrix(self, policy: DMSPolicy):
        """
        Return the transition matrix for the given policy.

        Parameters
        ----------
        policy: DMSPolicy
            The policy to value

        Returns
        -------
            The transition matrix induce by the given policy.
        """
        n = len(self.S)
        P_pi = np.zeros(shape=(n, n))

        for s_i in self.S:
            i = self.S_int[s_i]
            for s_j in self.adm_A(s_i):
                j = self.S_int[s_j]
                P_pi[i][j] = self.Q(s_i, policy(s_i))[s_j]

        return P_pi

    def policy_valuation(self, policy: DMSPolicy):
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
        ...

    def _build_r_P(self, s):

        if s not in self._Ps.keys():
            rs = np.zeros((len(self.A), 1))
            Ps = np.zeros((len(self.A), len(self.S)))

            adm = self.adm_A(s)
            for a in self.A:
                if a in adm:
                    rs[a] = self.r(s, a)
                    Ps[a] = self.Q(s, a)
                else:
                    rs[a] = -np.inf
            self._rs[s], self._Ps[s] = rs, Ps

        return self._rs[s], self._Ps[s]

    def improvement_VI(self, v):
        """
        
        Parameters
        ----------
        v

        Returns
        -------

        """

        v_r = v.copy()
        pol = {}
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)
            to_max = rs + self._lambda * np.dot(Ps, v)
            pairs = list(zip(self.S, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            v_r[i] = v_j
        return pol, v_r

    def improvement_GS(self, v):
        """

        Parameters
        ----------
        v

        Returns
        -------

        """

        v_r = np.zeros(v.shape)
        v = v.copy()
        pol = {}
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)
            to_max = rs + self._lambda * np.dot(Ps, v_r + v)
            pairs = list(zip(self.S, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            v_r[i] = v_j
            v[i] = 0

        return pol, v_r

    def improvement_JAC(self, v):
        """

        Parameters
        ----------
        v

        Returns
        -------

        """
        v = v.copy()
        v_r = v.copy()
        pol = {}
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)

            cs = Ps[:, i].reshape(rs.shape)
            jacob = 1 / (1 - self._lambda * cs)

            v_i = v[i].copy()
            v[i] = 0

            to_max = np.multiply(rs + self._lambda * np.dot(Ps, v), jacob)
            pairs = list(zip(self.S, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            v_r[i] = v_j
            v[i] = v_i
        return pol, v_r

    def optimal_value(self, v_0=None, epsilon=1E-2, method='GS'):
        """
        Computes the value function for the problem and creates the optimal policy.

        Parameters
        ----------
        method
        epsilon
        v_0

        Returns
        -------
        float
            The value function for the given time and state.
        """

        L = self.improvement_VI if method == 'VI' else \
            self.improvement_JAC if method == 'JAC' else self.improvement_GS

        self.computing_times[method].start()

        if v_0 is None:
            v_0 = self.v.copy()

        vt = v_0.copy()
        self.a_policy, self.v = self.improvement_VI(vt)
        stop = np.sqrt(np.dot((self.v - vt).T, self.v - vt))

        while stop > epsilon*(1 - self._lambda) / (2 * self._lambda):
            vt = self.v.copy()
            self.a_policy, self.v = L(self.v)
            stop = np.sqrt(np.dot((self.v - vt).T, self.v - vt))
            self.iteration_counts[method].add(vt)

        self.computing_times[method].stop()

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
        ...
