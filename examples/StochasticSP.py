from DiscreteWorld.Space import Space
from DiscreteWorld.Reward import Reward
from DiscreteWorld.MDPs import DeterministicMarkovian
import networkx as nx


class SSP_space(Space):
	def __init__(self, actions, states, time_horizon, G):
		super(SSP_space, self).__init__(actions, states, time_horizon)
		self.G = G

	def build_admisible_actions(self):
		"""
        Builds the admisible actions function for the put MDP
        """

		def adm_A(s):
			return list(self.G.successors(s))

		self.adm_A = adm_A

	def build_kernel(self):
		"""
        Builds the stochastic kernel function for the put MDP
        """

		def Q(s, a):
			sons = self.adm_A(s)
			print(sons)
			if len(sons) == 3:
				density = {ai: 0.6 if ai == a else 0.2 for ai in sons}
			elif len(sons) == 2:
				density = {ai: 0.7 if ai == a else 0.3 for ai in sons}
			else:
				density = {ai: 1 if ai == a else 0 for ai in sons}
			return density

		self.Q = Q


class SSP_reward(Reward):
	def __init__(self, space):
		super().__init__(space)

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

		probs = self.Q(state, action)
		G = self.space.G

		if t != self.T:
			r = sum(p * G[state][j]['c'] for j, p in probs.items())
		else:
			r = 0
		return r


if __name__ == '__main__':
	# We create the graph
	G = nx.DiGraph()
	G.add_nodes_from(range(1, 9))
	edges = [(1, 2, 2), (1, 3, 4), (1, 4, 3), (2, 5, 4), (2, 6, 5), (3, 5, 5),
	         (3, 6, 6), (3, 7, 1), (4, 7, 2), (5, 8, 1), (6, 8, 2), (7, 8, 6)]
	G.add_weighted_edges_from(edges, weight='c')

	# Create the space object
	actions, states, time_horizon = G.nodes(), G.nodes(), 3
	ssp_space = SSP_space(actions, states, time_horizon, G)

	# Create the reward object
	ssp_reward = SSP_reward(ssp_space)

	mdp = DeterministicMarkovian(ssp_space, ssp_reward)

	# Solves the MDP and stores its solution
	S0 = 1
	pol, v = mdp.solve(S0)
	# Prints the value of
	print(f'The optimal value is {v}')
	print('The policy is:', pol, sep='\n')
	print(mdp.policy.policy)



