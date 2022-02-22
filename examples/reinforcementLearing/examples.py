from reinforcement_learning.rl_config import RLConfig
from reinforcement_learning.agents import NeuralAgent, AgentConfig

from reinforcement_learning.utils.lake_envs import *
from reinforcement_learning.utils.preprocess import greyscale
from reinforcement_learning.utils.wrappers import PreproWrapper, MaxAndSkipEnv, MountainCar

import gym
from reinforcement_learning.dq_learning import QLearning
from reinforcement_learning.test_env import EnvTest
from reinforcement_learning.rl_utils import LinearSchedule, LinearExploration


def dummy_config():
	output_path = "../results/dummy_example/"
	nsteps_train = 5000
	config = RLConfig(
		# env config
		render_train=False,
		render_test=False,
		overwrite_render=True,
		record=False,
		high=255.,

		# output config
		output_path=output_path,
		model_output=output_path + "model.weights/",
		log_path=output_path + "log.txt",
		plot_output=output_path + "scores.png",

		# model and training config
		num_episodes_test=20,
		grad_clip=True,
		clip_val=10,
		saving_freq=5000,
		log_freq=50,
		eval_freq=100,
		soft_epsilon=0,
		replay_buffer=None,

		# hyper params
		nsteps_train=nsteps_train,
		batch_size=32,
		buffer_size=500,
		target_update_freq=500,
		gamma=0.99,
		learning_freq=4,
		state_history=4,
		lr_begin=0.00025,
		lr_end=0.0001,
		lr_nsteps=nsteps_train / 2,
		eps_begin=1,
		eps_end=0.01,
		eps_nsteps=nsteps_train / 2,
		learning_start=200
	)
	return config


def dummy_train():
	# env = gym.make('MountainCar-v0')
	# env = gym.make('CartPole-v0')
	# env = gym.make('Pendulumgi-v0')
	env = EnvTest((80, 80, 1))
	# env = EnvTest((5, 5, 1))

	config = dummy_config()
	# exploration strategy
	exp_schedule = LinearExploration(env, config.eps_begin,
	                                 config.eps_end, config.eps_nsteps)

	# learning rate schedule

	lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
	                             config.lr_nsteps)
	rl = QLearning(env=env, config=config)
	rl.run(exp_schedule, lr_schedule)
	print('#' * 100, '\nAfter train')
	rl.evaluate()
	rl.save()


def dummy_load():
	# env = gym.make('MountainCar-v0')
	# env = gym.make('CartPole-v0')
	# env = gym.make('Pendulumgi-v0')
	env = EnvTest((80, 80, 1))

	config = dummy_config()

	rl = QLearning(env=env, config=config, load=True)
	rl.evaluate()


def atari_config():
	output_path = "../results/atari/"
	nsteps_train = 5000000
	config = RLConfig(
		# env config
		render_train=False,
		render_test=True,
		env_name="Pong-v0",
		overwrite_render=True,
		record=True,
		high=255.,

		# output config
		output_path=output_path,
		model_output=output_path + "model.weights/",
		log_path=output_path + "log.txt",
		plot_output=output_path + "scores.png",
		record_path=output_path + "monitor/",

		# model and training config
		num_episodes_test=50,
		grad_clip=True,
		clip_val=10,
		saving_freq=250000,
		log_freq=50,
		eval_freq=250000,
		record_freq=250000,
		soft_epsilon=0.05,
		replay_buffer=None,

		# nature paper hyper params
		nsteps_train=nsteps_train,
		batch_size=32,
		buffer_size=1000000,
		target_update_freq=10000,
		gamma=0.99,
		learning_freq=4,
		state_history=4,
		skip_frame=4,
		lr_begin=0.00025,
		lr_end=0.00005,
		lr_nsteps=nsteps_train / 2,
		eps_begin=1,
		eps_end=0.1,
		eps_nsteps=1000000,
		learning_start=50000
	)
	return config


def atari_train():
	config = atari_config()
	env = gym.make(config.env_name)
	env = MaxAndSkipEnv(env, skip=config.skip_frame)
	env = PreproWrapper(env, prepro=greyscale, shape=env.observation_space.shape,
	                    overwrite_render=config.overwrite_render)

	# exploration strategy
	exp_schedule = LinearExploration(env, config.eps_begin,
	                                 config.eps_end, config.eps_nsteps)

	# learning rate schedule
	lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
	                             config.lr_nsteps)

	# train model
	model = QLearning(env, config)
	model.run(exp_schedule, lr_schedule)

	print('#' * 100, '\nAfter train')
	model.evaluate()
	model.save()


def atari_load():
	config = atari_config()
	env = gym.make(config.env_name)
	env = MaxAndSkipEnv(env, skip=config.skip_frame)
	env = PreproWrapper(env, prepro=greyscale, shape=env.observation_space.shape,
	                    overwrite_render=config.overwrite_render)

	rl = QLearning(env=env, config=config, load=True)
	rl.evaluate()


def lake_config():
	output_path = "../results/car/"
	nsteps_train = 1000000
	config = RLConfig(
		# env config
		render_train=False,
		render_test=False,
		env_name="Stochastic-4x4-FrozenLake-v0",
		overwrite_render=True,
		record=True,
		high=1.,

		# output config
		output_path=output_path,
		model_output=output_path + "model.weights/",
		log_path=output_path + "log.txt",
		plot_output=output_path + "scores.png",
		record_path=output_path + "monitor/",

		# model and training config
		num_episodes_test=50,
		grad_clip=True,
		clip_val=10,
		saving_freq=100000,
		log_freq=50,
		eval_freq=100000,
		record_freq=100000,
		soft_epsilon=0.05,
		replay_buffer='simple',

		# nature paper hyper params
		nsteps_train=nsteps_train,
		batch_size=32,
		buffer_size=250000,
		target_update_freq=5000,
		gamma=0.99,
		learning_freq=4,
		state_history=1,
		skip_frame=1,
		lr_begin=0.00025,
		lr_end=0.00005,
		lr_nsteps=nsteps_train / 2,
		eps_begin=1,
		eps_end=0.1,
		eps_nsteps=1000000 / 2,
		learning_start=10000,

		# Net config
		net_config=[
			('dense', {'units': 512, 'activation': 'relu'}),
			('dense', {'units': 512, 'activation': 'relu'}),
			('dense', {'units': 512, 'activation': 'relu'}),
			('dense', {'units': 512, 'activation': 'relu'}),
			('dense', {'activation': 'linear'})
		]
	)
	return config


def lake_train():
	config = lake_config()
	env = gym.make(config.env_name)
	env = MountainCar(env)

	# exploration strategy
	exp_schedule = LinearExploration(env, config.eps_begin,
	                                 config.eps_end, config.eps_nsteps)

	# learning rate schedule
	lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
	                             config.lr_nsteps)

	# train model
	model = QLearning(env, config)
	model.run(exp_schedule, lr_schedule)

	print('#' * 100, '\nAfter train')
	model.evaluate()
	model.save()


if __name__ == '__main__':
	# dummy_train()
	# dummy_load()
	atari_train()
	# atari_load()
	# lake_train()
