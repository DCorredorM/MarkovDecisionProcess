from discrete_world.space import finiteTimeSpace
from discrete_world.mdp import finiteTime
from math import exp
import matplotlib.pyplot as plt
from functools import reduce


class putOptionSpace(finiteTimeSpace):
	"""
    Implements the space class for the put option MDP
    """

	def __init__(self, actions, states, time_horizon, up, down, distr, strike, discount_rate, transaction_cost):
		super(putOptionSpace, self).__init__(actions, states, time_horizon, )
		self.u = up
		self.d = down
		self.distr = distr
		self.strike = strike
		self.delta = discount_rate
		self.tau = transaction_cost

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
		if state != 'Exercised' and action == 'Exercise':
			return (100 * max(0, self.strike -  state) - self.tau) * exp(- self.delta * t)
		else:
			return 0


def plot_put_policy(mdp, S0):
	pol = mdp.policy
	T = range(mdp.total_time)
	tau = []
	for ti in T:
		l = [p for (t, p), a in pol.policy.items() if t == ti and a == 'Exercise']
		try:
			ep = max(l)
		except:
			ep = None
		tau.append(ep)
	min_price = [S0 - mdp.space.d * t for t in T]
	max_price = [S0 + mdp.space.u * t for t in T]

	fig = plt.figure(figsize=(12, 7), constrained_layout=True)
	gs = fig.add_gridspec(1, 6)
	ax = fig.add_subplot(gs[0, :5])
	axh = fig.add_subplot(gs[0, 5:])

	ax.plot(T, tau, '-o', label='Policy')
	ax.plot(T, min_price, c='r', label='Minimum price')
	ax.plot(T, max_price, c='g', label='Maximum price')
	ax.hlines(y=mdp.reward.strike, xmin=0, xmax=mdp.total_time, colors='black', linestyles='--', label='Strike price')
	ax.legend()

	prob = compute_distribution(mdp, S0)
	pronT = prob[mdp.total_time - 1]
	axh.barh(list(pronT.keys()), list(pronT.values()), height=(mdp.space.u + mdp.space.d) / 3, color='black', alpha=0.6)
	plt.show()


def compute_distribution(mdp, S0):
	p = {(0, S0): 1}
	T = mdp.total_time

	def S(n_u, n_d):
		return round(S0 + n_u * mdp.space.u - n_d * mdp.space.d, 2)

	nodes = {t: sorted({S(n_u, d_u) for n_u in range(t+1) for d_u in range(0, t + 1 - n_u)}) for t in range(T + 1)}

	def parents(t, s):
		par = []
		if s in nodes[t-1]:
			par.append((s, mdp.Q(s, 'Pass')[s]))
		if round(s - mdp.space.u, 2) in nodes[t-1]:
			par.append((round(s - mdp.space.u, 2), mdp.Q(round(s - mdp.space.u, 2), 'Pass')[s]))
		if round(s + mdp.space.d, 2) in nodes[t-1]:
			par.append((round(s + mdp.space.d, 2), mdp.Q(round(s + mdp.space.d, 2), 'Pass')[s]))
		return par

	def pt(t, s):
		if t == 0:
			return 1
		else:
			if (t, s) in p.keys():
				pi = p[(t, s)]
			else:
				pi = sum(pt(t-1, s_i) * prob_trans for s_i, prob_trans in parents(t, s))
				p[(t, s)] = pi
			return pi

	for possible_end in nodes[T]:
		pt(T, possible_end)

	by_epoche = {t: {s: pr for (tt, s), pr in p.items() if tt == t} for t in range(mdp.total_time)}
	return by_epoche


if __name__ == "__main__":
	# creates the set of actions
	A = {'Exercise', 'Pass'}

	# Current price, maturity, up factor, down factor, discount rate
	S0, T, u, d, r, tau = 30, 30, 0.1, 0.1, 0.01, 50
	# Strike price
	strike_p = 29

	# Creates the states
	S = {round(S0 + t * u, 2) for t in range(T + 1)}.union({round(S0 - t * d, 2) for t in range(T + 1)})
	S = S.union({'Exercised'})

	# Define the probabilities:
	distr = {'u': 0.4, 'e': 0.1, 'd': 0.5}

	# Creates the Space object for the put option MDP
	put_space = putOptionSpace(A, S, T, u, d, distr, strike=strike_p, discount_rate=r, transaction_cost=tau)

	# Creates the MDP object with the proper space and reward objects
	mdp = finiteTime(put_space)
	# Solves the MDP and stores its solution
	pol, v = mdp.solve(S0)
	# Prints the value of
	print(f'The optimal value is {v}')
	print('The policy is:', pol, sep='\n')
	plot_put_policy(mdp, S0)
