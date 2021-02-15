from DiscreteWorld.Space import Space
from DiscreteWorld.Reward import Reward
from DiscreteWorld.MDPs import DeterministicMarkovian
from math import exp


class putOptionSpace(Space):
    """
    Implements the space class for the put option MDP
    """

    def __init__(self, actions, states, time_horizon, up, down):
        super(putOptionSpace, self).__init__(actions, states, time_horizon)
        self.u = up
        self.d = down

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
                density = {round(s + self.u, 2): 0.4, s: 0.1, round(s - self.u, 2): 0.4}
            else:
                density = {'Exercised': 1}
            return density

        self.Q = Q


class putOptionReward(Reward):
    """
    Implements the reward class for the put option MDP.
    """

    def __init__(self, space, strike, discount_rate):
        super().__init__(space)
        self.strike = strike
        self.r = discount_rate

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
            return 100 * max(0, self.strike - state) * exp(- self.r * t)
        else:
            return 0


if __name__ == "__main__":
    # creates the set of actions
    A = {'Exercise', 'Pass'}

    # Current price, maturity, up factor, down factor, discount rate
    S0, T, u, d, r = 10, 30, 0.1, 0.1, 0.01
    # Strike price
    strike_p = S0

    # Creates the states
    S = {round(S0 + t * u, 2) for t in range(T + 1)}.union({round(S0 - t * d, 2) for t in range(T + 1)})
    S = S.union({'Exercised'})

    # Creates the Space object for the put option MDP
    put_space = putOptionSpace(A, S, T, u, d)
    # Creates the Reward object for the put option MDP
    put_reward = putOptionReward(put_space, strike=strike_p, discount_rate=r)
    # Creates the MDP object with the proper space and reward objects
    mdp = DeterministicMarkovian(put_space, put_reward)
    # Solves the MDP and stores its solution
    pol, v = mdp.solve(S0)
    # Prints the value of
    print(f'The optimal value is {v}')
    print('The policy is:', pol, sep='\n')
