from reinforcement_learning.dql import QN

import tensorflow as tf
import numpy as np

from utilities.counters import Timer


class QLearning(QN):

    def __init__(self, env, config, logger=None, load=False):
        super().__init__(env, config, logger, load)

    def build(self, load=False):
        # Create Q values of state
        self.q_model = self._create_net(self.config.net_config)

        # Create Q values of next state
        self.target_q_model = self._create_net(self.config.net_config)

        # Creat loss function
        self.loss = tf.keras.losses.mean_squared_error

        # self.initialize(load=load)

    def initialize(self):
        pass

    def update_target_params(self):
        """
        Update parameters of Q' with parameters of Q
        """
        self.target_q_model.set_weights(self.q_model.get_weights())

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.q_model(np.array([state]))
        return np.argmax(action_values), action_values

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        # Create loss function target vs predicted q
        self.optimizer = tf.keras.optimizers.Adam(lr)
        loss = self.gradient_decent(s_batch, a_batch, r_batch, sp_batch, done_mask_batch)
        # merged op
        return loss, self.grad_norm

    # @tf.function
    def gradient_decent(self, s, a, r, sp, done_mask):
        num_actions = self.env.action_space.n
        with tf.GradientTape() as tape:
            predictions = self.q_model(s, training=True)
            indices = tf.one_hot(a, num_actions)
            predictions = tf.reduce_sum(predictions * indices, axis=1)

            target = self.compute_targets(s, a, r, sp, done_mask)
            prediction_loss = tf.reduce_mean((target - predictions) ** 2)

            gradients = tape.gradient(prediction_loss, self.q_model.trainable_variables)

            if self.config.grad_clip:
                gradients = [tf.clip_by_norm(item, self.config.clip_val) for item in gradients]

            self.optimizer.apply_gradients(zip(gradients, self.q_model.trainable_variables))
            self.grad_norm = tf.linalg.global_norm([item[0] for item in gradients])

        return prediction_loss

    def compute_targets(self, s, a, r, sp, done_mask):
        num_actions = self.env.action_space.n
        # not_done = 1 - tf.cast(done_mask, tf.float32)
        not_done = 1 - done_mask

        target = self.target_q_model(sp)
        q_samp = r + not_done * self.config.gamma * tf.reduce_max(target, axis=1)

        return q_samp
