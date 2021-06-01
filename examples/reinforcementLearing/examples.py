from reinforcement_learning.rl_lib import DPD, RLConfig
from reinforcement_learning.agents import NeuralAgent, AgentConfig
import gym
from reinforcement_learning.dql import QLearning
from reinforcement_learning.test_env import EnvTest
from reinforcement_learning.rl_utils import LinearSchedule, LinearExploration


def deep_Q_learning():
    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v0')
    # env = gym.make('Pendulumgi-v0')
    # env = EnvTest((80, 80, 1))

    value_net_config = [
        {'units': 200,
         'activation': 'relu'},
        {'units': 200,
         'activation': 'relu'},
        {'units': 1,
         'activation': 'linear'}
    ]

    config = RLConfig(value_net_config=value_net_config, verbose=True)
    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule

    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)
    rl = QLearning(env=env, config=config)
    rl.run(exp_schedule, lr_schedule)


if __name__ == '__main__':
    deep_Q_learning()
