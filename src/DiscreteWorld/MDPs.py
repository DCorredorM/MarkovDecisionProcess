from abc import abstractmethod
import numpy as np
import torch as pt
import logging

from DiscreteWorld.Policies import Policy, DMSPolicy, DMNSPolicy, RMSPolicy
from DiscreteWorld.Space import finiteTimeSpace
from Utilities.counters import Timer, TallyCounter
from Utilities.utilities import check_kwargs

import scipy.sparse as sp


class MDP:
    """
    Abstract class for markov decision process in infinite time.

    Attributes
    _________
    A: set
        Set of actions
    S: set
        Set of states

   Methods
    _______
    adm_A
        Function that given a (time, state) tuple returns the set of admisible actions for that pair
    Q
        Function that given a (state, action) tuple returns the probability distribution of the next state
    r
        Function that given a (t, state, action) tuple returns the reward.
    """
    maximize = 1
    minimize = 0
    ValueIteration = 'Value_Iteration'
    PolicyIteration = 'Policy_Iteration'
    LinearPrograming = 'LinearPrograming'

    def __init__(self, space: finiteTimeSpace, **kwargs):
        self._handle_kwargs(kwargs)

        self.space = space

        self.S = space.S
        self.S_int = space.S_int
        self.int_S = space.int_S
        self.l_S = space.l_S

        self.computing_times = dict()

        self.iteration_counts = dict()

        self.A = space.A
        self.adm_A = space.adm_A
        self.l_A = space.l_A
        self.A_int = space.A_int
        self.int_A = space.int_A

        self.Q = space.Q
        self.r = space.reward

        self._rs = dict()
        self._Ps = dict()

        logging.basicConfig()
        self.logger = logging.getLogger('MDP')
        if ('verbose', True) in kwargs.items():
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("Created logger")

    @abstractmethod
    def policy_valuation(self, policy: Policy):
        """
        Abstract method that when implemented values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value

        Returns
        -------
        float
            The value of the policy
        """
        ...

    @abstractmethod
    def optimal_value(self, method, **kwargs):
        """
        Abstract method that computes the value function for the problem and creates the optimal policy.

        Returns
        -------
        float
            The value function for the given time and state.

        """
        ...

    @staticmethod
    def _handle_indexes(indexes, value):
        s_i = indexes[value]
        return s_i

    def _handle_kwargs(self, kwargs):
        self.verbose = check_kwargs('verbose', False, kwargs)
        self.sense = check_kwargs('sense', MDP.maximize, kwargs)
        self.sparse = check_kwargs('sparse', False, kwargs)

    def _build_tensors(self):
        """
        Builds a transition sparse tensor and a reward tensor that will be stored in memory for the algorithms.

        Returns
        -------
        tf.sparse.SparseTensor:
            The transition tensor. Has shape (S, S, A), and contains in the position (j, s, a) the probability of
            reaching state j given that action a is taken in state s.

        tf.Tensor:
            The reward tensor. Has shape (S, A), and contains in the position (s, a) the reward of taking action a in
            state s.
        """
        S = self.l_S
        A = self.l_A

        a_indices = {a: [] for a in self.A}
        a_values = {a: [] for a in self.A}

        tr_indices = []
        tr_values = []
        tr_dense_shape = (S, A, S)  # s, a, j

        rew_dense_shape = (S, A)

        rew = np.ones(shape=rew_dense_shape) * -np.inf
        for s in self.S:
            s_i = MDP._handle_indexes(self.S_int, s)

            for a in self.adm_A(s):
                a_j = MDP._handle_indexes(self.A_int, a)

                if self.sense == MDP.minimize:
                    rew[s_i, a_j] = -self.r(s, a)
                else:
                    rew[s_i, a_j] = self.r(s, a)
                probs = self.Q(s, a)
                try:
                    for j, p in probs.items():
                        # Q tensor
                        s_j = MDP._handle_indexes(self.S_int, j)
                        tr_indices.append((s_i, a_j, s_j))
                        tr_values.append(p)
                        a_indices[a].append((s_i, s_j))
                        a_values[a].append(p)

                except AttributeError:
                    for j, p in enumerate(probs[0]):
                        # Q tensor
                        s_j = MDP._handle_indexes(self.S_int, j)
                        tr_indices.append((s_i, a_j, s_j))
                        tr_values.append(p)
                        a_indices[a].apend((s_i, s_j))
                        a_values[a].apend(p)

        # Q tensor
        tr_indices = np.array(list(zip(*tr_indices)))
        transition = pt.sparse_coo_tensor(tr_indices, tr_values, tr_dense_shape, dtype=pt.double)

        # p_a matrices
        a_tensors = {}
        for a in self.A:
            i = np.array(list(zip(*a_indices[a])))
            a_tensors[a] = sp.coo_matrix((a_values[a], (i[0, :], i[1, :])), shape=(self.l_S, self.l_S))
            a_tensors[a] = a_tensors[a].tocsc()

        # Reward tensor
        reward = pt.tensor(rew, dtype=pt.double)

        return transition, reward, a_tensors

    def _build_r_P(self, s):
        """
        Slices the transition and reward tensors for the given state.
        Parameters
        ----------
        s: state
            The state for which the transition matrix, and reward is required.

        Returns
        -------

        """
        if s not in self._Ps.keys():
            s_i = self.S_int[s]
            Ps = self.Q_tensor[s_i]
            rs = self.r_tensor[s_i].reshape((self.l_A, 1))
            if not self.sparse:
                Ps = Ps.to_dense()
            self._rs[s], self._Ps[s] = rs, Ps

        return self._rs[s], self._Ps[s]

    def _build_P_mu(self, policy=None):
        if 'building_P_Mu' not in self.computing_times.keys():
            self.computing_times['building_P_Mu'] = Timer('building_P_Mu', verbose=self.verbose)
        self.computing_times['building_P_Mu'].start()
        if policy is None:
            policy = self.policy
        rew = self.r_tensor.numpy()
        if self.sparse:
            pol_tensors = policy.a_matrices()
            z = True
            for a in self.A:
                if z:
                    Ps = self.a_tensors[a].dot(pol_tensors[a])
                    # rd = pol_tensors[a].multiply(rew)
                    z = False
                else:
                    Ps += self.a_tensors[a].dot(pol_tensors[a])
                # rd += self.a_tensors[a].multiply(rew)

            rd = np.nan_to_num((rew * policy.matrix), 0).sum(axis=1)
            rd = sp.csr_matrix(rd).T
        else:
            Ps = pt.einsum("saj,as->sj", self.Q_tensor.to_dense(), policy.matrix.type(pt.double))
            rd = np.nan_to_num((rew * policy.matrix.numpy().T), 0).sum(axis=1)
            rd = sp.csr_matrix(rd).T
            rd = rd.todense()

        self.computing_times['building_P_Mu'].stop()
        return Ps, rd


class finiteTime(MDP):
    """
    Implements an MDP in which the policy is a deterministic markovian policy


    Attributes
    __________

    v: dict
        Stores the value function for each time state combination
    """

    def __init__(self, space: finiteTimeSpace, **kwargs):
        super().__init__(space, **kwargs)
        self.T = space.T
        self.policy = DMNSPolicy(space)
        self.v = dict()
        self.a_policy = dict()

    def policy_valuation(self, policy: DMNSPolicy, history):
        """
        Values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value
        history: History
            History

        Returns
        -------
        float
            The value of the policy
        """
        t = len(history)
        if t == self.T:
            return self.r(t, history[-1])
        else:
            xt, ut = history[-1]
            u = policy((t, xt))
            v = self.r(t, xt, ut) + sum(
                self.Q(xt, u)[x] *
                self.policy_valuation(policy, history=history + [(x, u)])
                for x in self.S)
            return v

    def optimal_value(self, t, state):
        """
        Computes the value function for the problem and creates the optimal policy.

        Parameters
        ----------
        t: int
            The time for which the optimal is computed.
        state
            The state for which the optimal is computed.

        Returns
        -------
        float
            The value function for the given time and state.
        """
        if t < self.T:
            if (t, state) not in self.v.keys():
                def sup(u):
                    return self.r(t, state, u) + sum(
                        p * self.optimal_value(t + 1, y)
                        for y, p in self.Q(state, u).items())

                pairs = [(u, sup(u)) for u in self.adm_A(state)]
                u, v = max(pairs, key=lambda x: x[1])
                self.a_policy[t, state] = u
                self.v[t, state] = v
            return self.v[t, state]
        else:
            if (t, state) not in self.v.keys():
                self.v[t, state] = self.r(t, state)
                self.a_policy[t, state] = None
            return self.v[t, state]

    def solve(self, initial_state):
        """
        Starts the recursion at a given initial state and solves the MDP.

        Parameters
        ----------
        initial_state: state
            State in which the recursion is initialized

        Returns
        -------
        dict
            The optimal policy.
        float
            The value of the optimal policy.
        """
        v = self.optimal_value(0, initial_state)
        self.policy.add_policy(self.a_policy, initial_state)
        return self.policy, v


class infiniteTime(MDP):
    """
    Implements an MDP in which the policy is a deterministic markovian policy


    Attributes
    __________

    v: dict
        Stores the value function for each time state combination
    """

    def __init__(self, space: finiteTimeSpace, _lambda: float, **kwargs):
        super().__init__(space, **kwargs)
        self.S = list(self.S)

        self.policy = DMSPolicy(space)
        self.v = pt.zeros((len(self.S), 1))
        self.lambda_ = _lambda
        self.a_policy = dict()

        if ('load_from_files', True) in kwargs.items():
            self.logger.info('No tensors created, remember to add the later.')
            self.Q_tensor, self.r_tensor, self.a_tensors = None, None, None
            self.LP = None
        else:
            self.logger.info('Starting to create tensors')
            self.computing_times['preprocess'] = Timer('Preprocess', verbose=self.verbose)
            self.computing_times['preprocess'].start()

            # dummy LP object for accessing indexers for building the matrix
            self.LP = MDPLPSolver(self)
            # builds the tensors, matrices needed for the method.
            self.Q_tensor, self.r_tensor, self.a_tensors, LP_matrix, lp_cost = self._build_tensors()
            for s in self.S:
                self._build_r_P(s)
            self.computing_times['preprocess'].stop()

            self.LP = check_kwargs('lp_solver', gurobiMDPSolver(self, LP_matrix, lp_cost), kwargs)

    def load_tensors(self, Q_tensor, r_tensor, a_tensors):
        self.logger.info(f'Started loading tensors')
        self.Q_tensor = Q_tensor
        self.r_tensor = r_tensor
        self.a_tensors = a_tensors
        self.logger.info(f'Ended loading tensors')

    def build_LP(self, A_matrix, b_vector, c_vector):
        self.LP = gurobiMDPSolver(self, A_matrix, c_vector, b_vector)

    def _build_tensors(self):
        """
        Builds a transition sparse tensor and a reward tensor that will be stored in memory for the algorithms.

        Returns
        -------
        tf.sparse.SparseTensor:
            The transition tensor. Has shape (S, S, A), and contains in the position (j, s, a) the probability of
            reaching state j given that action a is taken in state s.

        tf.Tensor:
            The reward tensor. Has shape (S, A), and contains in the position (s, a) the reward of taking action a in
            state s.
        """
        S = self.l_S
        A = self.l_A

        a_indices = {a: [] for a in self.A}
        a_values = {a: [] for a in self.A}

        tr_indices = []
        tr_values = []
        tr_dense_shape = (S, A, S)  # s, a, j

        rew_dense_shape = (S, A)

        A_mat_index = [[], []]
        A_mat_values = []

        # costs pf the linear program
        lp_c = np.zeros((1, self.l_S * self.l_A))

        rew = np.ones(shape=rew_dense_shape) * -np.inf
        for s in self.S:
            s_i = MDP._handle_indexes(self.S_int, s)

            for a in self.adm_A(s):
                a_j = MDP._handle_indexes(self.A_int, a)

                if self.sense == MDP.minimize:
                    rew[s_i, a_j] = -self.r(s, a)
                    # rew[s_i, a_j] = -10000
                else:
                    rew[s_i, a_j] = self.r(s, a)
                    # rew[s_i, a_j] = 10000
                si = self.S_int[s]
                ai = self.A_int[a]
                xi = self.LP.index_to_index(si, ai)
                lp_c[0, xi] = rew[s_i, a_j]

                probs = self.Q(s, a)
                try:
                    for j, p in probs.items():
                        # Q tensor
                        s_j = MDP._handle_indexes(self.S_int, j)
                        tr_indices.append((s_i, a_j, s_j))
                        tr_values.append(p)
                        a_indices[a].append((s_i, s_j))
                        a_values[a].append(p)

                        # LP matrix
                        column = self.LP.index_to_index(s_i, a_j)
                        row = s_j

                        A_mat_index[0].append(row)
                        A_mat_index[1].append(column)
                        A_mat_values.append(- self.lambda_ * p)
                    if s not in probs.items():
                        column = self.LP.index_to_index(s_i, a_j)
                        row = s_i
                        A_mat_index[0].append(row)
                        A_mat_index[1].append(column)
                        A_mat_values.append(1)

                except AttributeError:
                    for j, p in enumerate(probs[0]):
                        # Q tensor
                        s_j = MDP._handle_indexes(self.S_int, j)
                        tr_indices.append((s_i, a_j, s_j))
                        tr_values.append(p)
                        a_indices[a].apend((s_i, s_j))
                        a_values[a].apend(p)

                        # LP matrix
                        column = self.LP.index_to_index(s_i, a_j)
                        row = s_j

                        A_mat_index[0].append(row)
                        A_mat_index[1].append(column)
                        A_mat_values.append(- self.lambda_ * p)
                    if s not in probs.items():
                        column = self.LP.index_to_index(s_i, a_j)
                        row = s_i
                        A_mat_index[0].append(row)
                        A_mat_index[1].append(column)
                        A_mat_values.append(1)
        # Q tensor
        tr_indices = np.array(list(zip(*tr_indices)))
        transition = pt.sparse_coo_tensor(tr_indices, tr_values, tr_dense_shape, dtype=pt.double)

        # p_a matrices
        a_tensors = {}
        for a in self.A:
            i = np.array(list(zip(*a_indices[a])))
            a_tensors[a] = sp.coo_matrix((a_values[a], (i[0, :], i[1, :])), shape=(self.l_S, self.l_S))
            a_tensors[a] = a_tensors[a].tocsc()

        # Reward tensor
        reward = pt.tensor(rew, dtype=pt.double)

        # LP sparse matrix
        LP_A = sp.csr_matrix((A_mat_values, (A_mat_index[0], A_mat_index[1])), shape=(self.l_S, self.l_S * self.l_A))

        return transition, reward, a_tensors, LP_A, lp_c

    def policy_valuation(self, policy: DMSPolicy):
        """
        Values a given policy.

        Parameters
        ----------
        policy: Policy
            Policy to value
        history: History
            History

        Returns
        -------
        float
            The value of the policy
        """
        if 'policy_valuation' not in self.computing_times.keys():
            self.computing_times['policy_valuation'] = Timer('policy_valuation', verbose=self.verbose)
        self.computing_times['policy_valuation'].start()
        Pd, rd = self._build_P_mu(policy)
        if self.sparse:
            A = (sp.identity(self.l_S) - self.lambda_ * Pd)
            A = A.tocsc()
            v = sp.linalg.spsolve(A, rd)
        else:
            A = (np.identity(self.l_S) - self.lambda_ * Pd.numpy())
            v = np.linalg.solve(A, rd)
        self.computing_times['policy_valuation'].stop()
        return v

    def select_method(self):
        if self.l_S > self.l_A * 100:
            return MDP.PolicyIteration
        else:
            return MDP.PolicyIteration

    def _improvement_VI(self, v):
        """
        
        Parameters
        ----------
        v

        Returns
        -------

        """

        v_r = v.clone()
        pol = {}
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)
            # to_max = rs + self._lambda * np.dot(Ps, v)
            to_max = rs + self.lambda_ * Ps.matmul(v)

            pairs = list(zip(self.A, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            v_r[i] = v_j[0]

        pol = DMSPolicy(self.space, pol)
        return pol, v_r

    def _improvement_GS(self, v):
        """

        Parameters
        ----------
        v

        Returns
        -------

        """

        if 'policy_improvement_GS' not in self.computing_times.keys():
            self.computing_times['policy_improvement_GS'] = Timer('policy_improvement_GS', verbose=self.verbose)
        self.computing_times['policy_improvement_GS'].start()
        v_r = pt.zeros(v.shape, dtype=pt.double)
        try:
            v = v.clone()
        except AttributeError:
            v = v.copy()
        pol = {}
        pol_indexes = [[], []]
        pol_values = []
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)
            to_max = rs + self.lambda_ * Ps.matmul(v_r + v)

            pairs = list(zip(self.A, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            pol_indexes[0].append(self.S_int[s])
            pol_indexes[1].append(self.A_int[u])
            pol_values.append(1)

            v_r[i] = v_j[0]
            v[i] = 0
            if i % 10000 == 0:
                self.logger.info(f'Policy improvment iteration {i}')

        matrix = np.asarray(sp.coo_matrix((pol_values, (pol_indexes[0], pol_indexes[1])), shape=(self.space.l_S, self.space.l_A)).todense())
        policy = DMSPolicy(self.space, matrix, from_matrix=True)
        policy.policy = pol
        self.computing_times['policy_improvement_GS'].stop()
        return policy, v_r

    def _improvement_JAC(self, v):
        """

        Parameters
        ----------
        v

        Returns
        -------

        """
        try:
            v = v.clone()
        except AttributeError:
            v = v.copy()
        try:
            v_r = v.clone()
        except AttributeError:
            v_r = v.copy()
        pol = {}
        for s in self.S:
            i = self.S_int[s]

            rs, Ps = self._build_r_P(s)

            cs = pt.transpose(Ps, 0, 1)[i]
            jacob = 1 / (1 - self.lambda_ * cs)

            v_i = v[i].clone()
            v[i] = 0

            to_max = pt.mul(rs + self.lambda_ * np.dot(Ps, v), jacob.reshape((self.l_A, 1)))
            pairs = list(zip(self.A, to_max.tolist()))
            u, v_j = max(pairs, key=lambda x: x[1])

            pol[s] = u
            v_r[i] = v_j[0]
            v[i] = v_i[0]

        pol = DMSPolicy(self.space, pol)
        return pol, v_r

    def _policy_iteration(self, initial_policy: DMSPolicy = None):
        """
        Pertforms the policy iteration algorithm.
        Returns
        -------

        """
        stop = False
        pol = initial_policy
        while not stop:
            # Policy valuation
            v = self.policy_valuation(pol)

            # Policy improvement
            policy, v = self._improvement_GS(v)

            if not np.array_equal(pol.matrix, policy.matrix):
                pol = policy
            else:
                pol = policy
                stop = True
        self.policy = pol

    def _policy_from_lp(self, x):
        """
        Given a solution creates a policy object representing the solution.
        Parameters
        ----------
        x: LP solution.

        Returns
        -------

        """
        self.logger.info(f'Creating policy from lp')
        x = x.reshape((self.l_S, self.l_A))
        x_sum = np.repeat(x.sum(axis=1).reshape(self.l_S, 1), self.l_A, axis=1)
        pol = np.divide(x, x_sum)
        # pol = sp.csr_matrix(pol)
        policy = DMSPolicy(self.space, pol, from_matrix=True)

        self.logger.info(f'Done with policy')
        return policy

    def _linear_programing(self, alpha):
        """
        Solves the LP model and creates a policy given the x variables

        Returns
        -------

        """
        self.logger.info(f'Started solving problem with LP approach')
        self.LP.LP_b = alpha

        if self.sparse:
            self.LP.build_linear_program_sparse()
        else:
            self.LP.build_linear_program()

        x = self.LP.solve_lp()
        self.logger.info(f'Finish solving problem with LP approach')
        return x

    def _value_iteration(self, v_0, epsilon, improvement_method):
        """
        Computes the value function for the problem and creates the optimal policy.

        Parameters
        ----------
        improvement_method
        epsilon
        v_0

        Returns
        -------
        float
            The value function for the given time and state.
        """

        L = self._improvement_VI if improvement_method == 'VI' else \
            self._improvement_JAC if improvement_method == 'JAC' else self._improvement_GS

        vt = v_0.clone().double()
        self.a_policy, self.v = L(vt)
        stop = np.sqrt(np.dot((self.v - vt).T, self.v - vt))
        self.iteration_counts[improvement_method] = TallyCounter(improvement_method)

        while stop > epsilon * (1 - self.lambda_) / (2 * self.lambda_):
            try:
                vt = self.v.clone()
            except AttributeError:
                vt = self.v.copy()
            self.a_policy, self.v = L(self.v)
            stop = np.sqrt(np.dot((self.v - vt).T, self.v - vt))
            self.iteration_counts[improvement_method].add(vt)

            self.logger.info(f'The current stopping criterion is {stop}')

    def optimal_value(self, method, **kwargs):
        """
        Abstract method that computes the value function for the problem and creates the optimal policy.

        Arguments
        _________
        method:
            the method ttha
        Returns
        -------
        float
            The value function for the given time and state.

        """
        if method == MDP.ValueIteration:
            try:
                v_0 = check_kwargs('v_0', self.v.clone().double(), kwargs)
            except AttributeError:
                v_0 = check_kwargs('v_0', self.v.copy().double(), kwargs)
            epsilon = check_kwargs('epsilon', 1E-1, kwargs)
            improvement_method = check_kwargs('improvement_method', 'GS', kwargs)
            t = Timer(f'{MDP.ValueIteration}_{improvement_method}', verbose=self.verbose)
            self.computing_times[t.name] = t
            t.start()
            self._value_iteration(v_0, epsilon, improvement_method)
            self.policy.add_policy(self.a_policy)
            t.stop()

        elif method == MDP.PolicyIteration:
            initial_policy = check_kwargs('initial_policy', self.policy, kwargs)
            if 'initial_policy' in kwargs.keys():
                initial_policy = kwargs['initial_policy']
            else:
                if self.policy.matrix is None:
                    self.policy.create_random_policy()
            t = Timer(f'{MDP.PolicyIteration}')
            self.computing_times[t.name] = t
            t.start()
            self._policy_iteration(initial_policy)
            t.stop()

        elif method == MDP.LinearPrograming:
            alpha = check_kwargs('alpha', 1 / self.l_S * np.ones(shape=self.l_S), kwargs)
            t = Timer(f'{MDP.LinearPrograming}')
            self.computing_times[t.name] = t
            t.start()
            x = self._linear_programing(alpha)
            t.stop()
            if x is not None:
                self.policy = self._policy_from_lp(x)
                self.v = self.policy_valuation(self.policy)
            else:
                self.logger.warning("Something went wrong. Infeasible model...")

    def solve(self, **kwargs):
        """
        Starts the recursion at a given initial state and solves the MDP.

        Parameters
        ----------
        improvement_method
        epsilon
        v_0

        Returns
        -------
        dict
            The optimal policy.
        float
            The value of the optimal policy.
        """

        method = check_kwargs('method', self.select_method(), kwargs)
        if 'method' in kwargs.keys():
            del kwargs['method']

        self.optimal_value(method, **kwargs)
        return self.policy, self.v


class MDPLPSolver:
    def __init__(self, mdp: infiniteTime, matrix_A=None, vector_c=None, alpha=None):
        self.space = mdp.space
        self.mdp = mdp

        self.S = mdp.S
        self.A = mdp.A
        self.lambda_ = mdp.lambda_

        self.S_int = mdp.S_int
        self.int_S = mdp.int_S
        self.A_int = mdp.A_int
        self.int_A = mdp.int_A

        self.l_S = mdp.l_S
        self.l_A = mdp.l_A

        try:
            self.Q_tensor = mdp.Q_tensor
            self.r_tensor = mdp.r_tensor
        except AttributeError:
            pass

        self.LP_A = matrix_A
        self.LP_c = vector_c
        self.LP_b = alpha

        # The optimization model object
        self.model = None

        self.model_built = False

        self.logger = self.mdp.logger

    def index_to_index(self, i_s, i_a):
        return self.l_A * i_s + i_a

    def set_to_index(self, s, a):
        i_s = self.S_int[s]
        i_s = self.A_int[a]
        return self.index_to_index(i_s, i_s)

    def index_to_index_inv(self, xi):
        si, ai = xi // self.A, xi % self.A
        return si, ai

    def index_to_set(self, xi):
        si, ai = self.index_to_index_inv(xi)
        return self.int_S[si], self.int_A[ai]

    def build_linear_program_sparse(self):
        """
        This method builds the linear programm.

        Returns
        -------

        """
        ...

    def solve_lp(self):
        """
        Solves the lp.

        Returns
        -------

        """
        ...

    def _build_sparse_A_matrix(self):
        """
        Builds and stores a sparse matrix for A.
        """
        indexes = self.Q_tensor._indices()
        size = len(indexes)

        A_index = [[], []]
        A_values = []
        for i in range(size):
            si, ai, ji = indexes[0, i], indexes[1, i], indexes[2, i]
            s, a, j = self.int_S[si], self.int_A[ai], self.int_S[ji]

            column = self.index_to_index(si, ai)
            row = ji
            A_index[0].append(row)
            A_index[1].append(column)
            p = self.Q_tensor[si, ai, ji]

            if a in self.mdp.space.adm_A(j):
                A_values.append(1 - self.lambda_ * p)
            else:
                A_values.append(-self.lambda_ * p)

        self.LP_A = sp.csr_matrix((A_values, (A_index[0], A_index[1])), shape=(self.l_S, self.l_S * self.l_A))

    def _build_cost_c_vector(self):
        self.c = self.r_tensor.numpy().reshape((1, self.l_S * self.l_A))


class gurobiMDPSolver(MDPLPSolver):
    def __init__(self, mdp: MDP, matrix_A=None, vector_c=None, alpha=None):
        super().__init__(mdp, matrix_A, vector_c, alpha=None)
        global gp
        import gurobipy as gp
        if matrix_A is None:
            self._build_sparse_A_matrix()
        if vector_c is None:
            self._build_cost_c_vector()
        if alpha is None:
            self._build_cost_c_vector()

    def build_linear_program_sparse(self):
        self.logger.info(f'Started building LP problem')
        model = gp.Model()
        model._x = model.addMVar(self.l_S * self.l_A, lb=0, obj=-self.LP_c)
        model.addConstr(self.LP_A @ model._x == self.LP_b)
        self.model = model
        self.model.write('loaded_lp.lp')
        self.logger.info(f'Ended building LP problem')

    def build_linear_program(self):
        model = gp.Model()
        x = {(s, a): model.addVar(lb=0, obj=self.space.reward(s, a))
                    for s in self.S for a in self.A}

        model.addConstrs(sum(x[j, a] for a in self.space.adm_A(j))
                         - sum(self.lambda_ * self.space.Q(s, a)[j] * x[s, a]
                               for s in self.S for a in self.space.adm_A(s)
                               if j in self.space.Q(s, a).keys())
                         == self.LP_b[self.S_int[j]]
                         for j in self.S)
        model._x = x
        self.model = model

    def solve_lp(self):
        self.logger.info(f'Started Solving  LP')
        if not self.model_built:
            if self.mdp.sparse:
                self.build_linear_program_sparse()
            else:
                self.build_linear_program()

        self.model.optimize()
        self.logger.info(f'Ended Solving LP')
        if self.model.status == 2:
            if self.mdp.sparse:
                return self.model._x.X
            else:
                x = np.array([[self.model._x[s, a].x for a in self.A] for s in self.S])
                return x

