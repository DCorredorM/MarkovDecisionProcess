import sys

from DiscreteWorld.Space import infiniteTimeSpace
from DiscreteWorld.MDPs import infiniteTime
from DiscreteWorld.MDPs import MDP
from Utilities.counters import Timer, TallyCounter, TallyMeasurer
import matplotlib.pyplot as plt
from matplotlib import cm
from DiscreteWorld.Policies import DMSPolicy
import logging

import pickle5 as pickle

from Utilities.utilities import norm

import numpy as np
from scipy.stats import poisson
from itertools import product
import torch as pt

global U, D, L, R
U, D, L, R = 'u', 'd', 'l', 'r'


class pplaneSpace(infiniteTimeSpace):
    """
    Implements the space class for the put option MDP
    """

    def __init__(self, actions, states, target, forbidden, M):
        super().__init__(actions, states)
        self.T = target
        self.F = forbidden
        self.M = M
        self.N = int(np.sqrt(len(self.S)))

    def Q_prima(self, s, a):
        distr = dict()
        x, y = s
        if a == U:
            distr = {
                (x, y + 1): 0.3,
                (x, y + 2): 0.4,
                (x - 1, y + 2): 0.2,
                (x - 1, y + 1): 0.1
            }
        elif a == D:
            distr = {
                (x, y): 0.3,
                (x, y - 1): 0.3,
                (x - 1, y): 0.2,
                (x - 1, y - 1): 0.2
            }
        elif a == L:
            distr = {
                (x - 1, y + 1): 0.3,
                (x - 1, y): 0.2,
                (x - 2, y): 0.3,
                (x - 2, y + 1): 0.2
            }
        elif a == R:
            distr = {
                (x + 1, y): 0.3,
                (x + 1, y + 1): 0.4,
                (x, y): 0.2,
                (x, y + 1): 0.1
            }
        return distr

    def build_admisible_actions(self):
        """
        Builds the admisible actions function for the put MDP
        """

        def adm_A(s):
            # return {a for a in self.A if delta(s, a) in self.S}
            return {a for a in self.A if set(self.Q_prima(s, a).keys()).intersection(self.S) != set()}

        self.adm_A = adm_A

    def build_kernel(self):
        """
        Builds the stochastic kernel function for the put MDP
        """

        def Q(s, a):
            adjust = sum(p for j, p in self.Q_prima(s, a).items() if j in self.S)

            distr = {
                j: p / adjust
                for j, p in self.Q_prima(s, a).items() if j in self.S
            }

            return distr

        self.Q = Q

    def reward(self, state, action=None):
        """
        Reward function for the put option.

        Parameters
        ----------
        state:
            state
        action:
            Action

        Returns
        -------
        float
            the reward for the given  (state, action) tuple.

        """

        def r_prima(j):
            return min([norm(np.array(t) - np.array(j)) for t in self.T]) + (lambda x: 1 if j in self.F else 0)(
                j) * self.M

        return sum(p * r_prima(j) for j, p in self.Q(state, action).items())


class Spaceship:
    def __init__(self, space: pplaneSpace, initial_position, policy, name='spaceship'):
        self.initial_position = initial_position
        self.position = initial_position
        self.policy = policy
        self.space = space
        self.crashes = TallyCounter(name + 'crashes')
        self.obj_fun = TallyMeasurer(name + 'obj_fun')
        self.average_crashes = TallyMeasurer(name + 'avg_crashes')
        self.average_obj_fun = TallyMeasurer(name + 'avg_obj_fun')

    def move(self):
        s = self.position
        a = self.policy(s)
        probabilities = self.space.Q(s, a)

        i = np.random.choice(list(range(len(probabilities.keys()))), p=list(probabilities.values()))
        self.position = list(probabilities.keys())[i]
        self.obj_fun.measure(1)
        if self.position in self.space.F:
            self.crashes.count()

    def reset(self, initial_position=None):
        if initial_position is None:
            self.position = self.initial_position
        else:
            self.position = initial_position

        self.average_obj_fun.add(self.obj_fun())
        self.obj_fun.reset()

        self.average_crashes.add(self.crashes())
        self.crashes.reset()


class TheSpace:
    def __init__(self, space: pplaneSpace, spaceship=None, **kwargs):
        self.space = space
        self.spaceship = spaceship
        self.paths = []
        logging.basicConfig()

        self.logger = logging.getLogger('MDP')
        if ('verbose', True) in kwargs.items():
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("Created logger")

    def simulate_path(self, initial_pos):
        self.spaceship.reset(initial_pos)
        path = []
        n = TallyCounter('step_counter')
        while self.spaceship.position not in self.space.T:
            n.count()
            path.append(self.spaceship.position)
            self.spaceship.move()

            if n() > np.sqrt(2) * self.space.N * 100:
                break
            elif n() % 50 == 0:
                self.logger.info(f'The position of the spaceship is {self.spaceship.position}')

        path.append(self.spaceship.position)
        self.paths.append(path)
        return path

    def simulate(self, n, initial_pos):
        i = 0
        while i < n:
            self.simulate_path(initial_pos)
            i += 1

    def paint_space(self):
        fig, ax = plt.subplots()

        grid = [[0 if (i, j) in self.space.F else 0.5 if (i, j) in self.space.T else 1 for j in range(self.space.N)] for
                i in range(self.space.N)]

        ax.matshow(grid, cmap='gist_ncar')

        for p in self.paths:
            TheSpace.plot_path(ax, p)
        plt.show()
        return fig, ax

    @staticmethod
    def plot_path(ax, path):
        X, Y = [], []
        for x, y in path:
            X.append(y)
            Y.append(x)

        ax.plot(X, Y)


def random_red_areas_cuadr(n1, n2, N=200, p=0.05):
    Ss = list(product(tuple(range(n1, n2 + 1)), tuple(range(n1, n2 + 1))))
    if ((N ** 2) / len(Ss)) ** (-1) < 0.05:
        print(f'la region es muy pequeña {((N ** 2) / len(Ss)) ** (-1)}')
        p = 1
    else:
        p = p * (N ** 2) / len(Ss)
    idx = list(map(bool, np.random.binomial(1, p, size=len(Ss))))

    return list(map(tuple, np.array(Ss)[idx].tolist()))


def random_red_areas_rectangulares(I1, I2, N=200, p=0.05):
    Ss = list(product(tuple(range(I1[0], I1[1] + 1)), tuple(range(I2[0], I2[1] + 1))))
    if ((N ** 2) / len(Ss)) ** (-1) < p:
        print(f'la region es muy pequeña {((N ** 2) / len(Ss)) ** (-1)}')
        p = 1
    else:
        p = p * (N ** 2) / len(Ss)
    idx = list(map(bool, np.random.binomial(1, p, size=len(Ss))))

    return list(map(tuple, np.array(Ss)[idx].tolist()))


def small_case():
    tc = Timer('cargando', verbose=True)
    tc.start()
    N = 100

    S = list(product(tuple(range(N + 1)), tuple(range(N + 1))))
    A = {U, D, L, R}

    forbidden = random_red_areas_rectangulares((int(0.3 * N), int(0.7 * N)), (int(0.3 * N), int(0.7 * N)), p=0.03, N=N)
    forbidden += random_red_areas_rectangulares((int(0.8 * N), N), (0, int(0.2 * N)), p=0.01, N=N)
    forbidden += random_red_areas_rectangulares((0, int(0.2 * N)), (int(0.8 * N), N), p=0.01, N=N)

    target = list(product(tuple(range(N - 2, N)), tuple(range(N - 2, N))))

    M = 2000
    p_space = pplaneSpace(A, S, target, forbidden, M)

    _lambda = 0.9

    p_mdp = infiniteTime(p_space, _lambda, sparse=True, sense=MDP.minimize, verbose=False)
    tc.stop()

    polVI, vVI = p_mdp.solve(method=MDP.ValueIteration)
    polPI, vPI = p_mdp.solve(method=MDP.PolicyIteration)
    polLP, vLP = p_mdp.solve(method=MDP.LinearPrograming)

    print(p_mdp.computing_times)
    simulate_and_plot(p_space, polPI, n=10)
    simulate_and_plot(p_space, polVI, n=10)
    simulate_and_plot(p_space, polLP, n=10)


def build_space(N):
    S = list(product(tuple(range(N + 1)), tuple(range(N + 1))))
    A = [U, D, L, R]

    # forbidden = random_red_areas_rectangulares((int(0.3 * N), int(0.7 * N)), (int(0.3 * N), int(0.7 * N)), p=0.03, N=N)
    forbidden = []

    def add_to_forb(x, len_x, y, len_y, prob):
        return random_red_areas_rectangulares((x, x + len_x), (y, y + len_y), p=prob, N=N)

    n_f = 12
    for _ in range(n_f):
        low, high = max(int((15 / 200) * N) - 1, 0), max(int((30 / 200) * N) - 1, 2)
        l_x = np.random.randint(low, high)
        x = np.random.randint(low=0, high=N - l_x)
        l_y = np.random.randint(low, high)
        y = np.random.randint(low=0, high=N - l_y)
        forbidden += add_to_forb(x, l_x, y, l_y, 0.05 / n_f)

    if N == 200:
        target = list(product(tuple(range(155, 157 + 1)), tuple(range(155, 157 + 1))))
    else:
        target = list(product(tuple(range(int(3 * N / 4), int(3 * N / 4) + 2)), tuple(range(int(3 * N / 4), int(3 * N / 4) + 2))))

    M = 2000
    p_space = pplaneSpace(A, S, target, forbidden, M)
    space = TheSpace(p_space)
    return p_space, space


def simulate_and_plot(p_space, pol, start=(0, 0), n=10, verbose=False, plot=True):
    t = Timer("Simulatoin", verbose=verbose)
    t.start()
    spaceship = Spaceship(p_space, start, pol)
    the_space = TheSpace(p_space, spaceship, verbose=verbose)
    the_space.simulate(n, start)
    if plot: the_space.paint_space()
    t.stop()
    return spaceship


def create_and_pickle(N):
    tc = Timer('cargando', verbose=True)
    tc.start()

    p_space, space = build_space(N)
    space.paint_space()
    p_mdp = infiniteTime(p_space, _lambda, sparse=True, sense=MDP.minimize, verbose=False)
    tc.stop()

    path = f'examples/data/Forbidden{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.space.F, file)

    path = f'examples/data/Q_tensor{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.Q_tensor, file)

    path = f'examples/data/r_tensor{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.r_tensor, file)

    path = f'examples/data/a_tensor{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.a_tensors, file)

    path = f'examples/data/A_matrix{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.LP.LP_A, file)

    path = f'examples/data/b_vector{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.LP.LP_b, file)

    path = f'examples/data/c_vector{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(p_mdp.LP.LP_c, file)


def read_objects(N, _lambda):
    tc = Timer('Cargando', verbose=True)
    tc.start()

    p_space, space = build_space(N)

    path = f'examples/data/Forbidden{N}.pickle'
    with open(path, 'rb') as file:
        p_space.F = pickle.load(file)

    path = f'examples/data/Q_tensor{N}.pickle'
    with open(path, 'rb') as file:
        Q_tensor = pickle.load(file)

    path = f'examples/data/r_tensor{N}.pickle'
    with open(path, 'rb') as file:
        r_tensor = pickle.load(file)

    path = f'examples/data/a_tensor{N}.pickle'
    with open(path, 'rb') as file:
        a_tensors = pickle.load(file)

    path = f'examples/data/A_matrix{N}.pickle'
    with open(path, 'rb') as file:
        A_matrix = pickle.load(file)

    path = f'examples/data/b_vector{N}.pickle'
    with open(path, 'rb') as file:
        b_vector = pickle.load(file)

    path = f'examples/data/c_vector{N}.pickle'
    with open(path, 'rb') as file:
        c_vector = pickle.load(file)

    p_mdp = infiniteTime(p_space, _lambda, sparse=True, sense=MDP.minimize, verbose=True, load_from_files=True)

    p_mdp.load_tensors(Q_tensor, r_tensor, a_tensors)
    p_mdp.build_LP(A_matrix, b_vector, c_vector)
    tc.stop()
    space.paint_space()

    return p_mdp


def solve(p_mdp):
    polVI, vVI = p_mdp.solve(method=MDP.ValueIteration)
    polPI, vPI = p_mdp.solve(method=MDP.PolicyIteration)
    polLP, vLP = p_mdp.solve(method=MDP.LinearPrograming)

    path = f'examples/data/PI_policy_mat{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(polPI.matrix, file)

    path = f'examples/data/LP_policy_mat{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(polLP.matrix, file)

    path = f'examples/data/VI_policy_mat{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(polVI.matrix, file)

    path = f'examples/data/PI_value{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(vPI, file)

    path = f'examples/data/LP_value{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(vLP, file)

    path = f'examples/data/VI_value{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(vVI, file)


def analyze_ship(ship: Spaceship, ax):
    print(f'{ship.name}\n{"-" * 10}\nThe average distance was {ship.average_obj_fun(mean=True)}')
    print(f'The average number of crashes was {ship.average_crashes(mean=True)}\n')

    hist, s, s = ax.hist(ship.average_obj_fun.List[1:], alpha=0.6, label=f'{ship.initial_position}')

    # ax.vlines(ymin=0, ymax=max(hist),
    #           x=norm(np.array(ship.initial_position) - np.array(ship.space.T[0])),
    #           label=f'Minimal distance',
    #           colors=['green'])


def read_pol(N):
    p_space, space = build_space(N)

    path = f'examples/data/Forbidden{N}.pickle'
    with open(path, 'rb') as file:
        p_space.F = pickle.load(file)

    path = f'examples/data/PI_policy_mat{N}.pickle'
    with open(path, 'rb') as file:
        pol_PI_mat = pickle.load(file)

    path = f'examples/data/LP_policy_mat{N}.pickle'
    with open(path, 'rb') as file:
        pol_LP_mat = pickle.load(file)

    path = f'examples/data/VI_policy_mat{N}.pickle'
    with open(path, 'rb') as file:
        pol_VI_mat = pickle.load(file)

    path = f'examples/data/PI_value{N}.pickle'
    with open(path, 'rb') as file:
        PI_value = pickle.load(file)

    path = f'examples/data/LP_value{N}.pickle'
    with open(path, 'rb') as file:
        LP_value = pickle.load(file)

    path = f'examples/data/VI_value{N}.pickle'
    with open(path, 'rb') as file:
        VI_value = pickle.load(file)

    polVI = DMSPolicy(p_space, pol_VI_mat, from_matrix=True)
    polPI = DMSPolicy(p_space, pol_PI_mat, from_matrix=True)
    polLP = DMSPolicy(p_space, pol_LP_mat, from_matrix=True)
    
    policies = {'VI': polVI, 'LP': polLP, 'PI': polPI}
    values = {'VI': VI_value, 'LP': LP_value, 'PI': PI_value}
    return p_space, policies, values


def read_pol_and_sim(N_sim, seed, starts, verbose = False):
    p_space, policies, values = read_pol(N)
    to_run = ['VI', 'PI', 'LP']
    for s in starts:
        for npol, pol in policies.keys():
            if npol in to_run:
                ship = simulate_and_plot(p_space, pol, start=s, n=N_sim, verbose=verbose)
                ship.name = npol

                fig, ax = plt.subplots()
                analyze_ship(ship, ax)
                ax.legend()
                plt.show()


def plot_sol(v, p_space: pplaneSpace):
    truncate = -3000
    for i in p_space.S:
        if v[p_space.S_int[i]] < truncate:
            v[p_space.S_int[i]] = truncate

    v = v.reshape((p_space.N, p_space.N))

    plt.matshow(v)
    plt.colorbar()
    plt.show()


def compare_Pols(N, N_comp):
    p_space, policies, values = read_pol(N)
    p_mdp = read_objects(N, _lambda=0.9)

    plot_sol(values['LP'], p_space)
    plot_sol(1 / p_mdp.l_S * np.ones(shape=p_mdp.l_S), p_space)

    def random_b():
        alpha = np.random.randint(low=10, high=100, size=p_mdp.l_S)
        return alpha / alpha.sum()

    for i in range(N_comp):
        a = random_b()
        pol, v = p_mdp.solve(method=MDP.LinearPrograming, alpha=a)
        plot_sol(v, p_space)
        plot_sol(a, p_space)


def lambda_variation(N, min, max, nsteps):
    p_space, policies, values = read_pol(N)
    p_mdp = read_objects(N, _lambda=0.9)

    vm1 = values['VI']
    pols = {}
    vals = {}
    for i in np.linspace(min, max, nsteps):
        p_mdp.lambda_ = i
        print(p_mdp.lambda_)
        pol, v = p_mdp.solve(method=MDP.ValueIteration, v_0=vm1)
        pols[i] = pol.matrix
        vals[i] = v
        vm1 = v

    path = f'examples/data/policies_lambdas{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(pols, file)

    path = f'examples/data/values_lambdas{N}.pickle'
    with open(path, 'wb') as file:
        pickle.dump(vals, file)


def load_analize_policies(N, N_sim, verbose, starts):
    path = f'examples/data/policies_lambdas{N}.pickle'
    with open(path, 'rb') as file:
        pols = pickle.load(file)

    path = f'examples/data/values_lambdas{N}.pickle'
    with open(path, 'rb') as file:
        vals = pickle.load(file)

    p_space, policies, values = read_pol(N)
    distances = {s:{} for s in starts}
    choques = {s:{} for s in starts}
    fig_d, ax_d = plt.subplots()
    fig_c, ax_c = plt.subplots()
    for s in starts:
        for i in pols.keys():
            pol = DMSPolicy(p_space, pols[i], from_matrix=True)
            ship = simulate_and_plot(p_space, pol, start=(0, 0), n=N_sim, verbose=verbose, plot=False)
            ship.name = i
            distances[s][i] = ship.average_obj_fun(mean=True)
            choques[s][i] = ship.average_crashes(mean=True)

        ax_d.plot(list(distances[s].keys()), list(distances[s].values()), label=f'start:{s}')
        ax_c.plot(list(choques[s].keys()), list(choques[s].values()), label=f'start:{s}')

    ax_d.set_xlabel(r'$\lambda$')
    ax_d.legend()
    ax_c.set_xlabel(r'$\lambda$')
    ax_c.legend()
    plt.show()


if __name__ == "__main__":
    seed = 6
    np.random.seed(seed)
    N = 200
    _lambda = 0.9
    N_sim = 50
    verbose = True
    starts = [(0, 0), (100, 0)]
    N_comp = 1

    # create_and_pickle(N)
    p_mdp = read_objects(N, _lambda)
    # solve(p_mdp)
    # read_pol_and_sim(N_sim, seed, starts, verbose)
    # compare_Pols(N, N_comp)

    # lambda_variation(N, 0.8, 0.99, 10)
    # load_analize_policies(N, N_sim, verbose, starts)
