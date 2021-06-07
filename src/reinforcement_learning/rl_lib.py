from abc import abstractmethod, ABC
from collections import deque
from utilities.replay_buffer import ReplayBuffer
from utilities.general import get_logger, Progbar, export_plot

from reinforcement_learning.agents import *
from utilities.counters import Timer, TallyCounter, TallyMeasurer


class RLProblem:
    def __init__(self, environment, agent=None, rl_config=None, **kwargs):
        self.config = rl_config if rl_config is not None else RLConfig(**kwargs)
        self.env = environment
        self.agent = agent if agent is not None else self._add_agent()

        # State placeholder
        self.state_ph = None
        # State placeholder
        self.action_ph = None

        # tf session
        self.sess = tf.InteractiveSession()

        logging.basicConfig()
        self._logger = logging.getLogger('MDP')
        if self.config.verbose:
            self._logger.setLevel(logging.DEBUG)
        self._logger.info("Created logger")

    def _check_layer_arg(self, layer, arg, default):
        if layer.get(arg) != default:
            layer[arg] = default
            self._logger.info(f'Argument: {arg}, changed to default ({default}) value to match environment.')
        return layer

    def _create_net(self, net_config):
        # creates the neural network
        net = Sequential()

        # first layer needs to have accurate input dimension
        s = self.env.observation_space.shape[0]
        input_dim = s if self.config.func_approx == 'state' else s + 1
        net_config[0] = self._check_layer_arg(net_config[0], 'input_dim', input_dim)

        # last layer needs to have a linear activation and accurate output dimension,
        # i.e., it needs to be a probability distribution over the action space
        last_layer = net_config[0]
        net_config[-1] = self._check_layer_arg(net_config[-1], 'units', 1)
        net_config[-1] = self._check_layer_arg(net_config[-1], 'activation', self.config.last_layer_activation)

        for layer in net_config:
            # ensures that either weights are passed or an initialization parameter is passed
            if layer.get('weights', True):
                # If none of them are passed, then the default initialization is used
                layer['kernel_initializer'] = layer.get('kernel_initializer', self.config.net_init)
            # Creates the current layer.
            net.add(Dense(**layer))

        return net

    @abstractmethod
    def _add_agent(self) -> Agent:
        ...

    @abstractmethod
    def solve(self):
        ...

    def render(self, time_horizon):
        env = self.env
        agent = self.agent

        state = env.reset()
        for i in range(time_horizon):
            env.render()
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            if done:
                break
        env.close()


class DPD(RLProblem, ABC):
    """
    implements the Deep Primal Dual method
    """

    def __init__(self, environment, agent=None, rl_config=None, **kwargs):
        super().__init__(environment, agent, rl_config, **kwargs)
        self.value_function = self._create_net(self.config.value_net_config)
        self.discount_factor = self.config.discount_factor
        self.sess = tf.InteractiveSession()

    def _add_agent(self) -> NeuralAgent:
        return NeuralAgent(environment=self.env, net_config=self.config.agent_net_config)

    def _lagrangian_gradient(self, state, action, reward):
        gamma = self.discount_factor
        c = self.config.regularization_parameter

        v = self.value_function.output
        pi = self.agent.policy.predict(np.array([state]))[0, action]

        v_prima = (v - reward) / gamma

        def delta():
            return reward + gamma * v_prima - v

        func = ((1 - gamma) * v
                + pi * delta()
                + c * tf.square(delta()))

        weights = self.value_function.trainable_weights
        gradients = k.gradients(func, weights)
        self.sess.run(tf.global_variables_initializer())
        evaluated_gradients = self.sess.run(gradients, feed_dict={self.value_function.input: np.array([state])})

        return evaluated_gradients

    def value(self, state):
        return self.value_function.predict(np.array([state]))[0]

    def primal_update(self, state, action, reward):
        grad = self._lagrangian_gradient(state, action, reward)
        theta = self.value_function.get_weights()
        eta = self.config.learning_rate
        theta2 = list(map(lambda x: x[0] - eta * x[1], zip(theta, grad)))
        self.value_function.set_weights(theta2)

        changes = list(map(lambda x: x.sum(), map(lambda x: x[0] - x[1], zip(theta, theta2))))
        # self._logger.info(changes)

    def deep_primal_dual_method(self, episodes, time_horizon):
        env = self.env
        T = range(time_horizon)
        gamma = self.discount_factor

        t = Timer("per_episode")
        rew_counter = TallyCounter("Rewards")
        for e in range(episodes):
            # current
            t.start()
            state0 = env.reset()
            r_e = TallyMeasurer(f"reward_{e}")
            for _ in T:
                action = self.agent.act(state0)
                state, reward, done, info = env.step(action)
                delta = reward + gamma * self.value(state) - self.value(state0)

                # primal update
                self.primal_update(state, action, reward)

                # dual update
                # noinspection PyArgumentList
                self.agent.learn(state, action, delta)

                # update current state
                state0 = state

                # save reward
                r_e.measure(reward)
                if done:
                    self._logger.info(f'Stopped because the episode was considered finished (done == True).')
                    break

            t.stop()
            if rew_counter.list:
                improve = r_e('mean') - rew_counter.list[0]('mean')
                self._logger.info(f'{"-" * 100}\nEpisode {e}:\n'
                                  f'\tRuntime: {t("last")}'
                                  f'\tAverage reward: {r_e("mean")} +- {r_e("std")}'
                                  f'\tImprovement: {improve}'
                                  f'\tRewards: {r_e.measures}')
            rew_counter.add(r_e)


class QLearning(RLProblem, ABC):
    def __init__(self, environment, rl_config=None, **kwargs):
        # Create the state_action network and a placeholder for its input

        kwargs.pop('func_approx', None)
        super().__init__(environment, 0, rl_config, func_approx='state_action', **kwargs)

        self.state_action_function, self.state_action_gradient = self._create_net(self.config.state_value_net_config)
        self.state_ph = self.state_action_function.input

        self.agent = self._add_agent()

    def _add_agent(self) -> greedyAgent:
        return greedyAgent(self.env, self.state_action_function)

    # def value_Q(self, state, action):
    # 	return self.state_action_function.predict(np.array([np.array(list(state) + [action])]))
    #
    # def update_weights(self, state, reward, action, future_state):
    # 	gradients = self.sess.run(self.state_action_gradient,
    # 	                          feed_dict={self.state_action_function.input: np.array([np.array(list(state) + [action])])})
    #
    # 	theta = self.state_action_function.get_weights()
    #
    # 	# Greedy future action
    # 	future_a = self.agent.act(future_state, epsilon=0)
    #
    # 	eta = self.config.learning_rate * (reward + self.value_Q(future_state, future_a) - self.value_Q(state, action))[0][0]
    # 	theta2 = list(map(lambda x: x[0] + eta * x[1], zip(theta, gradients)))
    # 	self.state_action_function.set_weights(theta2)

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward),
                                                  ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_eval), ("Max Q", self.max_q),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                               self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward
