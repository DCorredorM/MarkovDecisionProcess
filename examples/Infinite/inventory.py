from discrete_world.space import infiniteTimeSpace
from discrete_world.mdp import infiniteTime
from utilities.counters import Timer
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from utilities.utilities import norm

import numpy as np
import torch as pt
from scipy.stats import poisson


class inventorySpace(infiniteTimeSpace):
	"""
	Implements the space class for the put option MDP
	"""

	def __init__(self, actions, states, dem_distr, M,  f, c, h, K, _lambda, lambda_e_1=False):
		self.dem_distr = dem_distr
		self.M = M
		self.QQ = dict()
		self.pmf = dict()
		self.cdf = dict()
		self.A = actions
		self.S = states
		self.adm_A = NotImplemented
		self.Q = NotImplemented

		self.h = h
		self.c = c
		self.f = f
		self.K = K
		self._lambda = _lambda

		self.O = dict()
		self.F = dict()
		self.rew = dict()
		self.lambda_e_1 = lambda_e_1


		self.build_distr()
		self.build_admisible_actions()
		self.build_kernel()
		self.build_reward()

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
				distrdic = {}
				for j_s, j in enumerate(self.S):
					if self.M >= j > s + a:
						distr[0][j] = 0
					elif self.M >= s + a >= j > 0:
						distr[0][j] = self.pmf[s + a - j]
						distrdic[j_s] = distr[0][j]
					else:
						distr[0][j] = 1 - self.cdf[s + a - 1]
						distrdic[j_s] = distr[0][j]

				self.QQ[s, a] = distrdic

				return distrdic

		self.Q = Q

	def build_distr(self):
		for s in range(len(self.S)):
			self.pmf[s] = self.dem_distr.pmf(s)
			self.cdf[s] = self.dem_distr.cdf(s)

		self.pmf[-1] = 0
		self.cdf[-1] = 0

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

	def build_reward(self):
		for s in self.S:
			self.F[s] = sum(self.f(j) * self.dem_distr.pmf(j) for j in range(s)) \
			            + self.f(s) * (1 - self.dem_distr.cdf(s - 1))

			for a in self.adm_A(s):
				if a not in self.O.keys():
					self.O[a] = (self.K + self.c(a)) * int(a != 0)

				if s + a not in self.F.keys():
					self.F[s + a] = sum(self.f(j) * self.dem_distr.pmf(j) for j in range(s + a)) \
					                + self.f(s + a) * (1 - self.dem_distr.cdf(s + a - 1))
				self.rew[s, a] = self.F[s + a] - self.O[a] - self.h(s + a)
				if self.lambda_e_1:
					self.rew[s, a] *= (1 - self._lambda)


def plot_f1():
	inv_mdp = base_case()
	v_VI = inv_mdp.iteration_counts['VI'].measures
	v_JAC = inv_mdp.iteration_counts['JAC'].measures
	v_GS = inv_mdp.iteration_counts['GS'].measures

	y1 = [norm(v_VI - v) for v in V_VI]
	y2 = [norm(v_JAC - v) for v in V_JAC]
	y3 = [norm(v_GS - v) for v in V_GS]

	x1 = range(len(y1))
	x2 = range(len(y2))
	x3 = range(len(y3))

	plt.plot(x1, y1, label='Value iteration')
	plt.plot(x2, y2, label='Jacobi')
	plt.plot(x3, y3, label='Gauss-Seidel')

	plt.legend()
	plt.xlabel(r'$n$')
	plt.ylabel(r'$||v^n -v^*_\lambda||$')
	plt.show()


def base_case(log=False):
	def f(s):
		return 10 * s

	def c(a):
		return 2 * a

	def h(s):
		return s

	inv_space = inventorySpace(actions=A, states=S, dem_distr=dem_distr, M=M,  f=f, c=c, h=h, K=K, _lambda=_lambda)

	inv_mdp = infiniteTime(inv_space, _lambda)

	v_0 = pt.zeros((len(S), 1))

	pol_VI, v_VI = inv_mdp.solve(v_0, 0.001, improvement_method='VI')

	pol_JAC, v_JAC = inv_mdp.solve(v_0, 0.001, improvement_method='JAC')

	pol_GS , v_GS = inv_mdp.solve(v_0, 0.001, improvement_method='GS')

	if log:
		t_GS = inv_mdp.computing_times['GS'].total_time
		for i, t in inv_mdp.computing_times.items():
			print(i, round(t.total_time, 4), round(t.total_time / t_GS, 4), sep='&', end='\\\\ \n')
		print(inv_mdp.iteration_counts, '\n')

		print(f'VI - JAC: {norm(v_JAC - v_VI)}')
		print(f'JAC - GS: {norm(v_GS - v_JAC)}')
		print(f'VI - GS: {norm(v_GS - v_VI)}', '\n')

		print(f'VI: ', pol_VI)
		print(f'JAC: ', pol_JAC)
		print(f'GS: ', pol_GS, '\n')

	return inv_mdp


def alt_case(log=False):
	def f(s):
		return 10 * s

	def c(a):
		return 3 * a - 0.01 * a ** 2

	def h(s):
		return s

	inv_reward = inventoryReward(inv_space, f, c, h, K, _lambda)

	inv_mdp = infiniteTime(inv_space, inv_reward, _lambda)

	v_0 = np.ones(shape=(len(S), 1))

	inv_mdp._value_iteration(v_0, 0.001, improvement_method='VI')
	v_VI = inv_mdp.v
	pol_VI = inv_mdp.a_policy
	inv_mdp._value_iteration(v_0, 0.001, improvement_method='JAC')
	v_JAC = inv_mdp.v
	pol_JAC = inv_mdp.a_policy
	inv_mdp._value_iteration(v_0, 0.001, improvement_method='GS')
	v_GS = inv_mdp.v
	pol_GS = inv_mdp.a_policy

	if log:
		t_GS = inv_mdp.computing_times['GS'].total_time
		for i, t in inv_mdp.computing_times.items():
			print(i, round(t.total_time, 4), round(t.total_time / t_GS, 4), sep='&', end='\\\\ \n')
		print(inv_mdp.iteration_counts, '\n')

		print(f'VI - JAC: {norm(v_JAC - v_VI)}')
		print(f'JAC - GS: {norm(v_GS - v_JAC)}')
		print(f'VI - GS: {norm(v_GS - v_VI)}', '\n')

		print(f'VI: ', pol_VI)
		print(f'JAC: ', pol_JAC)
		print(f'GS: ', pol_GS, '\n')

	return inv_mdp


def changing_cost():
	inv_mdp1 = base_case()
	inv_mdp2 = alt_case()

	plt.plot(inv_mdp1.S, inv_mdp1.v, label=r'$c(s)=2s$')
	plt.plot(inv_mdp2.S, inv_mdp2.v, label=r'$c(s)=3s - 0.001s^2$')
	
	plt.legend()
	plt.xlabel('Estado')	
	plt.ylabel(r'$v^*_\lambda(s)$')
	
	plt.show()


def cmap_plot():
	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(1, 11)
	ax = fig.add_subplot(gs[0, :10])
	ax_c = fig.add_subplot(gs[0, 10:])

	return fig, ax, ax_c


def _lambda_to_1(lb=0.9, ub=0.999):

	cmap = cm.get_cmap('coolwarm', 256)
	_Lambda = np.linspace(lb, ub, 30)

	def f(s):
		return 10 * s

	def c(a):
		return 3 * a - 0.01 * a ** 2

	def h(s):
		return s

	MDPS = dict()

	f1, ax1, ax1c = cmap_plot()
	f2, ax2, ax2c = cmap_plot()
	f3, ax3, ax3c = cmap_plot()

	T_T = Timer('Con truco')
	T_F = Timer('Sin truco')
	for l in _Lambda:
		print(l)
		T_T.start()
		inv_reward = inventoryReward(inv_space, f, c, h, K, l, lambda_e_1=True)
		MDPS[l] = infiniteTime(inv_space, inv_reward, l)
		MDPS[l]._value_iteration()
		T_T.stop()
		T_F.start()
		inv_reward = inventoryReward(inv_space, f, c, h, K, l, lambda_e_1=False)
		MDPS[l] = infiniteTime(inv_space, inv_reward, l)
		MDPS[l]._value_iteration()
		T_F.stop()

		ax1.plot(MDPS[l].S, MDPS[l].v, c=cmap((l - lb) / (ub - lb)), label=r'$\lambda = $'+str(round(l, 4)))
		ax2.plot(MDPS[l].S, MDPS[l].v * (1 - l), c=cmap((l - lb) / (ub - lb)), label=r'$\lambda = $'+str(round(l, 4)))

		if l == lb:
			c_pol = MDPS[l].a_policy
			c_l = l

		if MDPS[l].a_policy != c_pol:
			c_u = l
			i_0 = 0
			for i in MDPS[l].space.S:
				if c_pol[i] > 0:
					i_0 = i

			i_0 += 5
			ax3.plot(range(i_0), list(c_pol.values())[:i_0], '-o',
			         c=cmap(((c_u + c_l) / 2 - lb) / (ub - lb)),
			         label=r'$\lambda \in$ ' + f'[{round(c_l, 3)}, {round(c_u, 3)})')

			c_pol = MDPS[l].a_policy
			c_l = c_u

	i_0 = 0
	for i in MDPS[l].space.S:
		if c_pol[i] > 0:
			i_0 = i

	i_0 += 5
	ax3.plot(range(i_0), list(c_pol.values())[:i_0], '-o',
	         c=cmap(((c_u + ub) / 2 - lb) / (ub - lb)),
	         label=r'$\lambda \in$ ' + f'[{round(c_l, 3)}, {round(ub, 3)}]')

	norm = mpl.colors.Normalize(vmin=lb, vmax=ub)
	mpl.colorbar.ColorbarBase(ax1c, cmap=cmap, norm=norm)
	mpl.colorbar.ColorbarBase(ax2c, cmap=cmap, norm=norm)
	mpl.colorbar.ColorbarBase(ax3c, cmap=cmap, norm=norm)

	ax1.set_xlabel('Estados')
	ax2.set_xlabel('Estados')
	ax3.set_xlabel('Estados')

	ax1.set_ylabel(r'$(1 - \lambda) v^*_\lambda$')
	ax2.set_ylabel(r'$v^*_\lambda$')
	ax3.set_ylabel('Acción')
	ax3.legend()

	f4, ax4 = plt.subplots()
	ax4.plot(_Lambda, [MDPS[l].computing_times['GS'].total_time for l in _Lambda])
	ax4.set_xlabel(r'$\lambda$')
	ax4.set_ylabel(r'Tiempo de cómputo (s)')

	print(f'El tiempo total que se ahorra uno es {(T_F.total_time - T_T.total_time) }, en porcentages {(T_F.total_time - T_T.total_time) / T_T.total_time}')

	plt.show()


if __name__ == "__main__":
	global M, A, S, K, _lambda, dem_distr, inv_space

	M = 100
	A = list(range(M + 1))
	S = list(range(M + 1))
	K = 2
	_lambda = 0.9

	dem_distr = poisson(mu=10)


	base_case(True)


