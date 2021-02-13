from abc import ABCMeta, abstractmethod, ABC
from itertools import product

from DiscreteWorld.Policies import Policy, DMPolicy
from DiscreteWorld.Space import Space
from DiscreteWorld.Reward import Reward


class MDP:
    def __init__(self, space: Space, reward: Reward) -> object:
        self.A = space.A
        self.S = space.S
        self.adm_A = space.adm_A
        self.Q = space.Q
        self.T = space.T
        self.r = reward.reward

    @abstractmethod
    def value_of_pi(self, policy: Policy, history):
        ...

    @abstractmethod
    def backward_induction(self):
        ...


class generalMDP(MDP):
    def __init__(self, space: Space, reward: Reward):
        super().__init__(space, reward)

    def value_of_pi(self, policy: Policy, history):
        """

        Parameters
        ----------
        policy
        history

        Returns
        -------
        """
        t = len(history)
        if t == self.T:
            return self.r[t](history[-1])
        else:
            xt, ut = history[-1]

            v = sum(
                policy.act(history)[u] *
                (self.r[t](xt, ut) +
                 sum(
                     self.Q(xt, u)[x] *
                     self.value_of_pi(policy, history=history + [(x, u)])
                     for x in self.S
                 ))
                for u in self.adm_A(xt)
            )
            return v


class DiscreteMarkovian(MDP):
    """
    Implements an MDP in which the policy si a discrete markovian policy
    """

    def __init__(self, space: Space, reward: Reward):
        super().__init__(space, reward)
        self.policy = DMPolicy(space)
        self.v = {} #(t, s): None for (t, s) in product(range(self.T), self.S)
        self.auxpol = {}

    def value_of_policy(self, policy: DMPolicy, history):
        """

        Parameters
        ----------
        policy
        history

        Returns
        -------
        """
        t = len(history)
        if t == self.T:
            return self.r[t](history[-1])
        else:
            xt, ut = history[-1]
            u = policy.act(t, xt)
            v = self.r(t, xt, ut) + sum(
                self.Q(xt, u)[x] *
                self.value_of_pi(policy, history=history + [(x, u)])
                for x in self.S)
            return v

    def optimal_value(self, t, state):
        if t < self.T:
            if (t, state) not in self.v.keys():
                def sup(u):
                    return self.r(t, state, u) + sum(p * self.optimal_value(t + 1, y) for y, p in self.Q(state, u).items())
                pairs = [(u, sup(u)) for u in self.adm_A(state)]
                u, v = max(pairs, key=lambda x: x[1])
                self.policy.add_action(t, state, u)
                self.auxpol[t, state] = u
                self.v[t, state] = v
            return self.v[t, state]
        else:
            if (t, state) not in self.v.keys():
                self.v[t, state] = self.r(t, state)
                self.policy.add_action(t, state, None)
                self.auxpol[t, state] = None
            return self.v[t, state]

    def solve(self, initial_state):
        v = self.optimal_value(0, initial_state)
        self.policy.policy = self.auxpol
        self.policy._create_tree(initial_state)
        return self.policy, v

