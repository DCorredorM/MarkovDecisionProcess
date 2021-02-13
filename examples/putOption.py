from DiscreteWorld.Space import Space
from DiscreteWorld.Reward import Reward
from DiscreteWorld.Policies import DMPolicy
from DiscreteWorld.MDPs import DiscreteMarkovian
from math  import exp


class putOptionSpace(Space):
	"""docstring for putOptionSpace"""
	def __init__(self, A, S, time_horizon):
		super(putOptionSpace, self).__init__(A, S, time_horizon)

	def build_admisible_actions(self):
		def adm_A(s):
			if s != 'Exercised':
				return self.A
			else:
				return {'-'}
		self.adm_A = adm_A

	def build_kernel(self):
		def Q(s, a):
			if a != 'Exercise' and s != 'Exercised':
				density = {round(s + 0.1, 2): 0.4, s: 0.1, round(s - 0.1, 2): 0.4}
			else:
				density = {'Exercised': 1}
			return density
		self.Q = Q


class putOptionReward(Reward):
	def __init__(self, space, strike, discount_rate):
		super().__init__(space)
		self.strike = strike
		self.r = discount_rate

	def reward(self, t, state, action=None):
		if state != 'Exercised' and action == 'Exercise':
			return max(0, self.strike - state) * exp(- self.r * t)
		else:
			return 0


if __name__ == "__main__":

	A = {'Exercise', 'Pass'}
	S0 = 29
	T = 30
	u = 0.1
	d = 0.1
	S = {round(S0 + t*u, 2) for t in range(T + 1)}.union({round(S0 - t*d, 2) for t in range(T + 1)})
	S = S.union({'Exercised'})
	strike = S0
	r = 0.01
	space = putOptionSpace(A, S, T)
	reward = putOptionReward(space, strike=strike, discount_rate=r)
	mdp = DiscreteMarkovian(space, reward)

	pol, v = mdp.solve(S0)

	print(v)
	print(pol)
	print(pol.policy)
	# print(mdp.v)






