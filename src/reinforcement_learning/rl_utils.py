import numpy as np


class LinearSchedule(object):
	def __init__(self, eps_begin, eps_end, nsteps):
		"""
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
		self.epsilon = eps_begin
		self.eps_begin = eps_begin
		self.eps_end = eps_end
		self.nsteps = nsteps

	def update(self, t):
		"""
        Updates epsilon

        Args:
            t: (int) nth frames
        """
		lam = t / self.nsteps
		self.epsilon = max(self.eps_begin + lam * (self.eps_end - self.eps_begin), self.eps_end)


class LinearExploration(LinearSchedule):
	def __init__(self, env, eps_begin, eps_end, nsteps):
		"""
        Args:
            env: gym environment
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
		self.env = env
		super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

	def get_action(self, best_action):
		"""
        Returns a random action with prob epsilon, otherwise return the best_action

        Args:
            best_action: (int) best action according some policy
        Returns:
            an action
        """
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return best_action
