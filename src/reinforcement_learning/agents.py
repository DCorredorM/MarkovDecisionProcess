from abc import abstractmethod, ABC

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# from keras import backend as k
from gym import Env
import numpy as np
import tensorflow.compat.v1 as tf
import logging

tf.disable_v2_behavior()


class AgentConfig:
	net_init = 'uniform'
	verbose = True
	learning_rate = 0.0001
	last_layer_activation = 'softmax'
	epsilon = 0.2

	def __init__(self, **kwargs):
		for name, value in kwargs.items():
			self._set_attribute(name, value)

	# noinspection PyMethodMayBeStatic
	def _set_attribute(self, name, value):
		exec(f'self.{name} = {value}')


class Agent:
	def __init__(self, environment, agent_config=None):
		self.environment: Env = environment
		self.config = agent_config if agent_config is not None else AgentConfig()

		logging.basicConfig()
		self._logger = logging.getLogger('MDP')
		if self.config.verbose:
			self._logger.setLevel(logging.DEBUG)
		self._logger.info("Created logger")

	@abstractmethod
	def act(self, state):
		...

	@abstractmethod
	def learn(self, state, action):
		...


class greedyAgent(Agent, ABC):
	def __init__(self, environment, state_action_function, agent_config=None):
		super().__init__(environment, agent_config)
		self.Q = state_action_function

	def act(self, state, epsilon=None):
		state = list(state)
		n = range(self.environment.action_space.n)
		epsilon = self.config.epsilon if epsilon is None else epsilon
		x = np.array([state + [a] for a in n])
		values = self.Q.predict(x).reshape((self.environment.action_space.n,))
		a = max(n, key=lambda i: values[i])

		return np.random.choice(n, p=[1 - epsilon if i == a else epsilon / (len(n) - 1) for i in n])


class NeuralAgent(Agent):
	"""
	Attributes
	__________


	"""

	def __init__(self, environment, net_config: list, agent_config=None):
		"""

		Parameters
		----------
		environment
		net_config: dict

		"""
		super().__init__(environment, agent_config)
		self.policy = self._create_net(net_config)
		self.sess = tf.InteractiveSession()

	def _check_layer_arg(self, layer, arg, default):
		if layer.get(arg) != default:
			layer[arg] = default
			self._logger.info(f'Argument: {arg}, changed to default ({default}) value to match environment.')
		return layer

	def _create_net(self, net_config):
		# creates the neural network
		net = Sequential()

		# first layer needs to have accurate input dimension
		net_config[0] = self._check_layer_arg(net_config[0], 'input_dim', self.environment.observation_space.shape[0])

		# last layer needs to have a soft_max activation and accurate output dimension,
		# i.e., it needs to be a probability distribution over the action space
		net_config[-1] = self._check_layer_arg(net_config[-1], 'units', self.environment.action_space.n)
		net_config[-1] = self._check_layer_arg(net_config[-1], 'activation', self.config.last_layer_activation)

		for layer in net_config:
			# ensures that either weights are passed or an initialization parameter is passed
			if layer.get('weights', True):
				# If none of them are passed, then the default initialization is used
				layer['kernel_initializer'] = layer.get('kernel_initializer', self.config.net_init)
			# Creates the current layer.
			net.add(Dense(**layer))
		return net

	def act(self, state):
		distribution = self.policy.predict(np.array([state]))[0]
		distribution = distribution / distribution.sum()
		# self._logger.info(f'Distribution is \n:{distribution}')
		action = np.random.choice(range(self.environment.action_space.n), p=distribution)
		return action

	# noinspection PyMethodOverriding
	def learn(self, state, action, delta):
		grad = self._log_gradient(action, state)
		theta = self.policy.get_weights()
		eta = self.config.learning_rate
		theta2 = list(map(lambda x: x[0] + eta * delta * x[1], zip(theta, grad)))
		self.policy.set_weights(theta2)

		changes = list(map(lambda x: x.sum(), map(lambda x: x[0] - x[1], zip(theta, theta2))))

	# self._logger.info(changes)

	def _log_gradient(self, action, state):

		out = self.policy.output
		weights = self.policy.trainable_weights

		func = k.log(out)

		gradients = k.gradients(func, weights)
		self.sess.run(tf.global_variables_initializer())
		evaluated_gradients = self.sess.run(gradients, feed_dict={self.policy.input: np.array([state])})

		return evaluated_gradients
