import gym
from functools import reduce
from reinforcement_learning.agents import NeuralAgent


def create_agent(env: gym.Env):
	net_config = [
		{'input_dim': env.observation_space.shape[0],
		 'units': 200,
		 'activation': 'relu'},
		{'units': 200,
		 'activation': 'relu'},
		{'units': env.action_space.n,
		 'activation': 'relu'}
	]
	agent = NeuralAgent(environment=env, net_config=net_config)
	print(agent.policy.summary())
	return agent


def carpool_test():
	env = gym.make('CartPole-v0')
	print(f'Action space: {env.action_space}')
	print(f'State space: {env.observation_space}')
	state = env.reset()
	print(f'Initial state: {state}')

	agent = create_agent(env)
	# print(help(env.step))

	actions = [0, 0, 2] * 500
	for i in range(1000):
		env.render()
		action = agent.act(state)
		state, reward, done, info = env.step(action)
		agent.learn(action, state, 10000)
		# print(action, reward)
	env.close()


def print_evs():
	print(reduce(lambda x, y: f'{x}\n{y}', list(gym.envs.registry.all())))


if __name__ == '__main__':
	print_evs()
	carpool_test()
