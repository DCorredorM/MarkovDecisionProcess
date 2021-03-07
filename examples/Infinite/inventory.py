from DiscreteWorld.Space import infiniteTimeSpace
from DiscreteWorld.Reward import infiniteTimeReward
from DiscreteWorld.MDPs import infiniteTime

import matplotlib.pyplot as plt
from Utilities.utilities import norm

import numpy as np
from scipy.stats import poisson


class inventorySpace(infiniteTimeSpace):
	"""
	Implements the space class for the put option MDP
	"""

	def __init__(self, actions, states, dem_distr, M):
		self.dem_distr = dem_distr
		self.M = M
		self.QQ = dict()
		self.pmf = dict()
		self.cdf = dict()
		self.A = actions
		self.S = states
		self.adm_A = NotImplemented
		self.Q = NotImplemented
		self.build_distr()
		self.build_admisible_actions()
		self.build_kernel()

	def build_admisible_actions(self):
		"""
		Builds the admisible actions function for the put MDP
		"""

		def adm_A(s):
			return list(range(self.M - s + 1))

		self.adm_A = adm_A

	def build_kernel(self):
		"""
		Builds the stochastic kernel function for the put MDP
		"""

		def Q(s, a):
			if (s, a) in self.QQ.keys():
				return self.QQ[s, a]
			else:
				distr = np.zeros(shape=(1, len(self.S)))
				for j in range(len(self.S)):
					if self.M >= j > s + a:
						distr[0][j] = 0
					elif self.M >= s + a >= j > 0:
						distr[0][j] = self.pmf[s + a - j]
					else:
						distr[0][j] = 1 - self.cdf[s + a - 1]

				self.QQ[s, a] = distr

				return distr

		self.Q = Q

	def build_distr(self):
		for s in range(len(self.S)):
			self.pmf[s] = self.dem_distr.pmf(s)
			self.cdf[s] = self.dem_distr.cdf(s)

		self.pmf[-1] = 0
		self.cdf[-1] = 0


class inventoryReward(infiniteTimeReward):
	"""
	Implements the reward class for the put option MDP.
	"""

	def __init__(self, space: inventorySpace, f, c, h, K):
		super().__init__(space)
		self.h = h
		self.c = c
		self.f = f
		self.K = K

		self.O = dict()
		self.F = dict()
		self.rew = dict()
		self.build_reward()

	def build_reward(self):
		for s in self.S:
			self.F[s] = sum(self.f(j) * self.space.dem_distr.pmf(j) for j in range(s)) \
						+ f(s) * (1 - self.space.dem_distr.cdf(s - 1))

			for a in self.adm_A(s):
				if a not in self.O.keys():
					self.O[a] = (self.K + self.c(a)) * int(a != 0)

				if s + a not in self.F.keys():
					self.F[s + a] = sum(self.f(j) * self.space.dem_distr.pmf(j) for j in range(s + a)) \
									+ f(s + a) * (1 - self.space.dem_distr.cdf(s + a - 1))
				self.rew[s, a] = self.F[s + a] - self.O[a] - self.h(s + a)

	def reward(self, state, action=None):
		"""
		Reward function for the put option.

		Parameters
		----------
		state:
			state
		action:
			Action

		Returns
		-------
		float
			the reward for the given  (state, action) tuple.

		"""
		return self.rew[state, action]


if __name__ == "__main__":
	M = 100
	A = list(range(M+1))
	S = list(range(M+1))
	K = 2
	_lambda = 0.9

	dem_distr = poisson(mu=10)

	inv_space = inventorySpace(A, S, dem_distr, M)

	def f(s):
		return 10 * s

	def c(a):
		# return 3 * a - 0.01 * a ** 2
		return 2 * a

	def h(s):
		return s

	inv_reward = inventoryReward(inv_space, f, c, h, K)

	inv_mdp = infiniteTime(inv_space, inv_reward, _lambda)

	v_0 = np.ones(shape=(len(S), 1))

	inv_mdp.optimal_value(v_0, 0.001, method='Value_Iteration')
	v_VI = inv_mdp.v
	pol_VI = inv_mdp.a_policy
	inv_mdp.optimal_value(v_0, 0.001, method='Jacobi')
	v_JAC = inv_mdp.v
	pol_JAC = inv_mdp.a_policy
	inv_mdp.optimal_value(v_0, 0.001, method='Gauss-Seidel')
	v_GS = inv_mdp.v
	pol_GS = inv_mdp.a_policy

	print(inv_mdp.computing_times)
	print(inv_mdp.iteration_counts, '\n')

	print(f'VI - JAC: {norm(v_JAC - v_VI)}')
	print(f'JAC - GS: {norm(v_GS - v_JAC)}')
	print(f'VI - GS: {norm(v_GS - v_VI)}', '\n')

	print(f'VI: ', pol_VI)
	print(f'JAC: ', pol_JAC)
	print(f'GS: ', pol_GS, '\n')

	# print(f'VI: ', v_VI)
	# print(f'JAC: ', v_JAC)
	# print(f'GS: ', v_GS)




