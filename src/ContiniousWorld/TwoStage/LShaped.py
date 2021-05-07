import logging

import gurobipy as gur
import numpy as np
import scipy.stats as sts

from Utilities.counters import TallyCounter, Timer


class FirstStage:
    def __init__(self, **kwargs):
        self.model = gur.Model('FS')
        self.model.setParam('OutputFlag', 0)

        self.A = kwargs.pop('A', None)
        assert self.A is not None, 'The A matrix needs to be given as a parameter'

        self.m, self.n = self.A.shape

        self.b = kwargs.pop('b', None)
        assert self.b is not None, 'The b vector needs to be given as a parameter'

        self.c = kwargs.pop('c', None)
        assert self.c is not None, 'The c vector needs to be given as a parameter'

        self.var_index = kwargs.pop('var_index', None)
        if self.var_index is not None:
            self.index_var = {value: key for key, value in self.var_index.items()}
        else:
            self.var_index = {i: i for i in range(self.n)}
            self.index_var = self.var_index

        self.x = None
        self.primal_constraints = None

        # Will model the expected value of the second stage
        self.theta = None
        self.consider_theta = False

    def build_model(self, multi_cut=False, num_scenarios=None):
        """
        Build an optimization model in canonical minimization form. i.e,

            min c^tx
            s.t.,
                Ax >= b
                 x >= 0
        """

        if multi_cut:
            assert num_scenarios is not None, \
                'When multicut the number of scenarios is required in order to build the theta variables'
            self.theta = self.model.addMVar(
                (num_scenarios,),
                lb=-gur.GRB.INFINITY,
                name="theta")
        else:
            self.theta = self.model.addMVar((1,), lb=-gur.GRB.INFINITY, name="theta")

        self.x = self.model.addMVar((self.n,), obj=self.c, name='x')
        self.primal_constraints = self.model.addMConstr(self.A, self.x, '>=', self.b)
        self.model.update()

    def solve(self):
        """
        Solves the model and returns the current value for x and the estimate of the second stage expected value, Theta.

        Returns
        -------
            x_hat: np.array
                The value of the decision variables for the current iteration

            theta_hat: float or np.array
                The estimate of the second stage expected value.
        """

        if self.consider_theta:
            self.model.update()
            self.model.optimize()
            x_hat, theta_hat = self.x.x, self.theta.x
        else:
            K = self.theta.shape[0]
            self.model.update()
            self.model.optimize()
            x_hat, theta_hat = self.x.x, [-np.inf] * K

        return x_hat.reshape((self.n, 1)), theta_hat

    def reset_model(self, multi_cut=False, num_scenarios=None):
        # Remove all the cuts
        self.model = gur.Model('FS')
        self.model.setParam('OutputFlag', 0)
        self.build_model(multi_cut, num_scenarios)
        self.consider_theta = False
        self.model.update()


class SecondStage:
    index = 0

    def __init__(self, **kwargs):
        index_ = SecondStage.__handle_index(**kwargs)
        self.primal = gur.Model(f'SSP_{index_}')
        self.dual = gur.Model(f'SSD_{index_}')

        self.primal.setParam('OutputFlag', 0)
        self.dual.setParam('OutputFlag', 0)

        self.W = kwargs.pop('W')
        assert self.W is not None, 'The W matrix needs to be given as a parameter'
        self.m, self.n = self.W.shape

        self.T_k = kwargs.pop('T_k')
        assert self.T_k is not None, 'The T_k matrix needs to be given as a parameter'

        self.h_k = kwargs.pop('h_k')
        assert self.h_k is not None, 'The h_k vector needs to be given as a parameter'

        self.q_k = kwargs.pop('q_k')
        assert self.q_k is not None, 'The q_k vector needs to be given as a parameter'

        self.var_index = kwargs.pop('var_index')
        assert self.var_index is not None, 'The var_index map needs to be given as a parameter'
        self.index_var = {value: key for key, value in self.var_index.items()}

        # If the probability is none the scenarios will be treated as equally probable
        self.prob = kwargs.pop('prob')

        # Creates the variables and sets the objective of the primal problem.
        # The formuation (constraints) of the primal depend on x_hat,
        # so it needs to be built every time an x_hat is queried
        self.y = self.primal.addMVar(shape=self.W.shape[1], obj=self.q_k)
        self.primal_constraints = None

        # Creates the variables and the constraints of the dual problem.
        # Here the objective depends on x_hat, so the objective will be updated everytime the x_hat is queried.
        self.pi = self.dual.addMVar(shape=self.W.shape[0], lb=-gur.GRB.INFINITY)
        self.dual_constraints = self.dual.addMConstr(self.W.T, self.pi, '<=', self.q_k.T)

    @staticmethod
    def __handle_index(**kwargs):
        index_ = kwargs.pop('index', None)
        if index_ is None:
            index_ = SecondStage.index
            SecondStage.index += 1
        return index_

    def build_primal(self, x_hat):
        """
        Assigns to the self.model attribute a gurobi model object with the seconds stage optimization model.

        Parameters
        ----------
        kwargs: dict
            key worded arguments needed to build the model. Model parameters.

        """
        if self.primal_constraints is not None: self.primal.remove(self.primal_constraints)
        self.primal_constraints = self.primal.addMConstr(self.W, self.y, '=', self.h_k - self.T_k @ x_hat)
        self.primal.update()

    def build_dual(self, x_hat):
        """
        Assigns to the self.model attribute a gurobi model object with the seconds stage optimization model.

        Parameters
        ----------
        kwargs: dict
            key worded arguments needed to build the model. Model parameters.

        """
        self.dual.setObjective((self.h_k - (self.T_k @ x_hat)).T @ self.pi, sense=gur.GRB.MAXIMIZE)

        self.dual.update()

    def solve_dual(self):
        """
        Solves the model and returns either the dual variables or the extreme rays.

        Returns
        -------

        """
        self.dual.setParam("DualReductions", 0)
        self.dual.setParam("InfUnbdInfo", 1)

        self.dual.optimize()
        if self.dual.status == 5:
            return False, np.array(self.dual.UnbdRay).reshape((self.m, 1)), self.dual.objVal
        return True, np.array(self.pi.x).reshape((self.m, 1)), self.dual.objVal

    def solve_primal(self):
        """
        Solves the model and returns either the dual variables or the extreme rays.

        Returns
        -------

        """
        self.primal.optimize()


class TwoStageSP:
    def __init__(self, first_stage, second_stage, **kwargs):
        self.first_stage = first_stage
        self.second_stage_sps = second_stage
        self.num_scenarios = len(second_stage)
        self.probabilities = np.array([sp.prob for sp in self.second_stage_sps])

        self.multi_cut = kwargs.pop('multi_cut', False)
        self.first_stage.build_model(multi_cut=self.multi_cut, num_scenarios=self.num_scenarios)

        epsilon = kwargs.pop('epsilon', 10e-2)
        self.epsilon = epsilon

        logging.basicConfig()
        self.logger = logging.getLogger('TwoStageSP')
        if ('verbose', True) in kwargs.items():
            self.logger.setLevel(logging.DEBUG)
        self.logger.info("Created logger")

        self.computing_times = dict()
        self.x_hat = None
        self.theta_hat = None
        self.objVal = None

        self.solved = False

    def multi_cut_l_shaped(self, reset=False):
        """
        Implements the multi cut L-shaped algorithm for solving the two stage stochastic program.

        Returns
        -------


        """
        if reset:
            self.first_stage.reset_model(True, self.num_scenarios)

        assert self.first_stage.theta.shape[0] > 1
        self.computing_times['multi_cut'] = Timer('multi_cut')
        # Stopping criterion
        stop = False

        # feasibility and optimality cuts counters
        r = TallyCounter('Feasibility')
        s = [TallyCounter(f'Optimality_{k}') for k in range(self.num_scenarios)]
        v = TallyCounter('Iterations')

        self.computing_times['multi_cut'].start()
        consider_theta = [False] * self.num_scenarios

        self.logger.info(f'\n{"-" * 100}\nMulti-cut L-Shaped' + \
                         f'\nConstraints: {len(self.first_stage.model.getConstrs())}' + \
                         f'\tVariables: {len(self.first_stage.model.getVars())}\n{"-" * 100}')
        self.logger.info(f'Lower bound\tUpper Bound\tGAP')
        while not stop:
            # Solve first stage and query x_hat, Theta_hat.
            x_hat, theta_hat = self.first_stage.solve()

            # Solve for each sub-problem solve its dual and add respective cuts.
            infeasible = False
            optimal = True

            w = 0
            for k, sp in enumerate(self.second_stage_sps):

                sp.build_dual(x_hat)
                feasibility, pi_sigma, wk = sp.solve_dual()
                w += sp.prob * wk
                if feasibility:
                    pi = pi_sigma
                    if round(theta_hat[k], 4) < round(wk, 4):
                        # Add optimality cuts
                        optimal = False
                        E = pi.T @ sp.T_k
                        e = pi.T @ sp.h_k
                        self.first_stage.model.addConstr(
                            E @ self.first_stage.x + self.first_stage.theta[k] >= e, name=f'OC_{k}{s[k].Count}')
                        s[k].count()
                        consider_theta[k] = True

                else:
                    infeasible = True
                    sigma = pi_sigma
                    D = sigma.T @ sp.T_k
                    d = sigma.T @ sp.h_k

                    # Add feasibility cuts
                    self.first_stage.model.addConstr(D @ self.first_stage.x >= d, name=f'FC_{k}{r.Count}')
                    self.logger.info(f'-inf\tinf\t-\tFC')
                    r.count()
                    break

            if not infeasible and optimal:
                stop = True
            elif not infeasible and not optimal:
                if all(consider_theta) and not self.first_stage.consider_theta:
                    self.first_stage.consider_theta = True

                    self.first_stage.model.setObjective(
                        self.first_stage.c @ self.first_stage.x
                        + self.probabilities.T @ self.first_stage.theta)

            if sum(theta_hat) > -np.inf:
                base = (self.first_stage.c @ x_hat)[0][0]
                lb, ub = (base + self.probabilities.T @ theta_hat, base + w)
                v.add((lb, ub))
                self.logger.info(f'{round(lb, 4)}\t{round(ub, 4)}\t{round((ub - lb) / ub, 4)}')
            else:
                v.count()

        self.computing_times['multi_cut'].stop()
        self.x_hat, self.theta_hat = self.first_stage.solve()
        self.objVal = self.first_stage.model.objVal
        self.solved = True

        self.logger.info(f'\nMulti-cut L-shaped converged\n{"-" * 100}' + \
                         f'\nTime:\t{self.computing_times["multi_cut"].total_time} seconds' + \
                         f'\nIterations:\t{v.Count}' + \
                         f'\nOptimal value:\t{self.first_stage.model.objVal}' + \
                         f'\nTotal cuts:\t{r.Count + sum(o.Count for o in s)}' + \
                         f'\n\tfeasibility:\t{r.Count}({100 * round(r.Count / (r.Count + sum(o.Count for o in s)), 3)}%)' + \
                         f'\n\toptimality:\t{sum(o.Count for o in s)}' + \
                         f'({round(100 * sum(o.Count for o in s) / (r.Count + sum(o.Count for o in s)), 3)}%)')
        return r, s, v

    def l_shaped(self, reset=False):
        if reset:
            self.first_stage.reset_model()
        assert self.first_stage.theta.shape[0] == 1
        self.computing_times['L_shaped'] = Timer('L_shaped')
        # Stopping criterion
        stop = False

        # feasibility and optimality cuts counters
        r, s, v = TallyCounter('Feasibility'), TallyCounter('Optimality'), TallyCounter('Iterations')

        self.computing_times['L_shaped'].start()
        self.logger.info(f'\n{"-" * 100}\nL-Shaped' + \
                         f'\nConstraints: {len(self.first_stage.model.getConstrs())}' + \
                         f'\tVariables: {len(self.first_stage.model.getVars())}\n{"-" * 100}')
        self.logger.info(f'Lower bound\tUpper Bound\tGAP')
        while not stop:
            # Solve first stage and query x_hat, Theta_hat.
            x_hat, theta_hat = self.first_stage.solve()

            # Solve for each sub-problem solve its dual and add respective cuts.
            E = np.zeros(shape=(1, self.first_stage.n))
            e = 0
            infeasible = False
            for sp in self.second_stage_sps:
                sp.build_dual(x_hat)
                feasibility, pi_sigma, wk = sp.solve_dual()
                if feasibility:
                    pi = pi_sigma
                    E += sp.prob * pi.T @ sp.T_k
                    e += sp.prob * pi.T @ sp.h_k

                else:
                    infeasible = True
                    sigma = pi_sigma
                    D = sigma.T @ sp.T_k
                    d = sigma.T @ sp.h_k

                    # Add feasibility cuts
                    self.first_stage.model.addConstr(D @ self.first_stage.x >= d, name=f'FC_{r.Count}')
                    self.logger.info(f'-inf\tinf\t-\tFC')

                    r.count()
                    break

            if not infeasible:
                # Add optimality cut

                self.first_stage.model.addConstr(
                    E @ self.first_stage.x >= e - self.first_stage.theta, name=f'OC_{s.Count}')
                if not self.first_stage.consider_theta:
                    self.first_stage.consider_theta = True
                    self.first_stage.model.setObjective(
                        self.first_stage.c @ self.first_stage.x
                        + self.first_stage.theta)

                s.count()

            # Check optimality conditions
            w = e - E @ x_hat
            if round(theta_hat[0], 4) >= round(w[0][0], 4):
                stop = True

            if theta_hat[0] > -np.inf:
                base = (self.first_stage.c @ x_hat)[0][0]
                lb, ub = ((base + theta_hat)[0], (base + w[0][0]))
                v.add((lb, ub))

                self.logger.info(f'{round(lb, 4)}\t{round(ub, 4)}\t{round((ub - lb) / ub, 4)}')
            else:
                v.count()

        self.computing_times['L_shaped'].stop()
        self.x_hat, self.theta_hat = self.first_stage.solve()
        self.objVal = self.first_stage.model.objVal
        self.solved = True

        self.logger.info(f'\nL-shaped converged\n{"-" * 100}' + \
                         f'\nTime:\t{self.computing_times["L_shaped"].total_time} seconds' + \
                         f'\nIterations:\t{v.Count}' + \
                         f'\nOptimal value:\t{self.objVal}' + \
                         f'\nTotal cuts:\t{r.Count + s.Count}' + \
                         f'\n\tfeasibility:\t{r.Count}({100 * round(r.Count / (r.Count + s.Count), 3)}%)' + \
                         f'\n\toptimality:\t{s.Count}' + \
                         f'({round(100 * s.Count / (r.Count + s.Count), 3)}%)')
        return r, s, v

    def solve(self, multi_cut=False, reset=None):
        if reset is None:
            reset = False
            if multi_cut != self.multi_cut:
                reset = True
        self.multi_cut = multi_cut
        if multi_cut:
            return self.multi_cut_l_shaped(reset)
        else:
            return self.l_shaped(reset)

    def _upper_bound(self, sample, alpha=0.01):
        vals = []
        N = len(sample)
        self.logger.info(f'{"-"*100}\nStarted computing upper bound\n{"-"*100}')
        t = Timer('upper bound')
        self.computing_times['upper_bound'] = t
        for s in sample:
            s.build_dual(self.x_hat)
            feasibility, pi_sigma, wk = s.solve_dual()
            vals.append(wk)

        mu = sum(vals) / len(vals)
        sigma = np.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

        upper_bound = self.first_stage.c @ self.x_hat + mu + sts.norm.ppf(1 - alpha) * sigma / np.sqrt(N)
        self.logger.info(f'{"-"*100}\nFinish computing upper bound. Total time was {t.total_time} seconds.\n{"-"*100}')
        return mu, sigma, upper_bound[0][0]

    def _lower_bound(self, samples, alpha=0.01):
        vals = []
        M = len(samples)
        n = len(samples[0])
        self.logger.info(f'{"-"*100}\nStarted computing lower bound\n{"-"*100}')
        t = Timer('lower bound')
        self.computing_times['lower_bound'] = t
        t.start()

        for s in samples:
            self.second_stage_sps = s
            self.solve(reset=True)
            vals.append(self.objVal)

        mu = sum(vals) / len(vals)
        sigma = np.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

        lower_bound = mu + sts.norm.ppf(alpha) * sigma / np.sqrt(M)
        t.stop()
        self.logger.info(f'Finish computing lower bound. Total time was {t.total_time} seconds.')

        return mu, sigma, lower_bound

    def confidence_interval(self, upper_bound_sample, lower_bound_samples, alpha=0.01):
        if not self.solved:
            self.solve()
        f_hat = self.objVal
        _, _, lb = self._lower_bound(lower_bound_samples, alpha)
        _, _, ub = self._upper_bound(upper_bound_sample, alpha)
        self.logger.info(f'Confidence interval for the GAP is ({lb - f_hat}, {ub - f_hat})')
        return lb - f_hat, ub - f_hat
