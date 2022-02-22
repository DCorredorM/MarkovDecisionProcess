from discrete_world.space import finiteTimeSpace
from discrete_world.mdp import finiteTime
from math import exp
import matplotlib.pyplot as plt
from functools import reduce


class putOptionSpace(finiteTimeSpace):
	"""
    Implements the space class for the put option MDP
    """

	def __init__(self, actions, states, time_horizon):
		super(putOptionSpace, self).__init__(actions, states, time_horizon)

	def build_admisible_actions(self):
		"""
        Builds the admisible actions function for the put MDP
        """

		def adm_A(s):
			if s != 'Exercised':
				return self.A
			else:
				return {'-'}

		self.adm_A = adm_A

	def build_kernel(self):
		"""
        Builds the stochastic kernel function for the put MDP
        """

		def Q(s, a):
			if a != 'Exercise' and s != 'Exercised':
				density = {round(s + self.u, 2): self.distr['u'], s: self.distr['e'], round(s - self.d, 2): self.distr['d']}
			else:
				density = {'Exercised': 1}
			return density

		self.Q = Q

	def reward(self, t, state, action=None):
		"""
        Reward function for the put option.
        ::math..

        Parameters
        ----------
        t: int
            time
        state:
            state
        action:
            Action

        Returns
        -------
        float
            the reward for the given (time, state, action) triple.

        """


