import numpy as np
import scipy.stats as sts
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce
import seaborn as sns
sns.set(font_scale=1, rc={'text.usetex':True})
import pandas as pd

from Utilities.counters import Timer

from ContiniousWorld.TwoStage.LShaped import FirstStage, SecondStage, TwoStageSP


def plot_discrete_pmf(values, probs, **kwargs):
    xlab = kwargs.pop('xlabel', '')
    ylab = kwargs.pop('ylabel', '')
    plt.bar(values, probs, **kwargs)

    plt.xlabel(xlab)
    plt.ylabel(ylab)


def discrete_instances():

    G = range(3)
    P = range(3)

    demand_scenarios = 6
    capacities_scenarios = 3

    min_demands = (10, 20, 30)
    max_demands = (100, 150, 300)

    demand_values = [list(np.linspace(min_demands[k], max_demands[k], demand_scenarios)) for k in P]
    binom_params = [np.random.rand() for k in P]
    demand_probabilities = [[sts.binom(demand_scenarios - 1, binom_params[k]).pmf(i)
                             for i in range(demand_scenarios)] for k in P]

    # for k in G:
    # 	plot_discrete_pmf(demand_values[k], demand_probabilities[k],
    # 	                  width=5, xlabel='Demanda', ylabel='Probabilidad')
    # 	plt.show()

    max_perdida = (0.3, 0.5, 0.6)
    cap_values = [list(np.linspace(max_perdida[k], 1, capacities_scenarios)) for k in G]

    cap_params = [0.4 + np.random.rand() / 2 for k in G]
    cap_probabilities = [[sts.binom(capacities_scenarios - 1, cap_params[k]).pmf(i)
                          for i in range(capacities_scenarios)] for k in G]
    
    # for k in G:
    # 	plot_discrete_pmf(cap_values[k], cap_probabilities[k],
    # 	                  width=0.1, xlabel='Pérdida de eficiencia', ylabel='Probabilidad')
    # 	plt.show()

    # create costs
    q = {i: np.random.randint(5, 20) for i in G}
    q = {(i, k): q[i] for i, k in product(G, P)}

    return q, dict(zip(product(*demand_values, *cap_values),
        map(lambda x: reduce(lambda y, z: z * y, x), product(*demand_probabilities, *cap_probabilities))))


def continuous_instances():
    G = range(3)
    P = range(3)

    demand_scenarios = 6
    capacities_scenarios = 3

    min_demands = (10, 20, 30)
    max_demands = (100, 150, 300)

    demand_values = [list(np.linspace(min_demands[k], max_demands[k], demand_scenarios)) for k in P]
    binom_params = [np.random.rand() for k in P]
    demand_probabilities = [[sts.binom(demand_scenarios - 1, binom_params[k]).pmf(i)
                             for i in range(demand_scenarios)] for k in P]

    dem_exp = [sum(v * p for v, p in zip(demand_values[k], demand_probabilities[k])) for k in P]

    max_perdida = (0.3, 0.5, 0.6)
    cap_values = [list(np.linspace(max_perdida[k], 1, capacities_scenarios)) for k in G]

    cap_params = [0.4 + np.random.rand() / 2 for k in G]
    cap_probabilities = [[sts.binom(capacities_scenarios - 1, cap_params[k]).pmf(i)
                          for i in range(capacities_scenarios)] for k in G]

    kappa_exp = [sum(v * p for v, p in zip(cap_values[k], cap_probabilities[k])) for k in G]

    # Create the gammas for the demands
    alphas = [1 + np.random.rand() * 10 for k in P]
    loc = [min(demand_values[k]) for k in P]
    betas = [alphas[k] / (dem_exp[k] - loc[k]) for k in P]
    gammas = [sts.gamma(a=alphas[k], loc=loc[k], scale=1 / betas[k]) for k in P]

    # Create kappas
    alphas = [1 + np.random.rand() * 4 for k in G]
    betas = [alphas[k] * (1 - kappa_exp[k]) / kappa_exp[k] for k in G]
    beta_vars = [sts.beta(a=alphas[k], b=betas[k]) for k in G]

    # create costs
    q = {i: np.random.randint(5, 20) for i in G}
    q = {(i, k): q[i] for i, k in product(G, P)}

    return q, gammas, beta_vars


# noinspection PyTypeChecker
def build_ss_matrix(G, P, instance, secondS_costs):
    d = list(instance[:G])
    kappa = list(instance[P:])

    index_var = dict(enumerate(map(lambda x: f'y_{x}', product(range(G), range(P)))))
    index_var.update(dict(enumerate(map(lambda x: f's_{x}', product(range(G), range(P))), start=len(index_var))))
    m = int(G * P + P)
    n = len(index_var)

    var_index = {value: key for key, value in index_var.items()}

    W_idex = []
    W_values = []

    Tk_idex = []
    Tk_values = []

    h_k = np.array([0] * (G * P) + d).reshape((m, 1))

    # Capacity constraints
    row = 0
    for row, (i, k) in enumerate(product(range(G), range(P))):
        col_y = var_index[f'y_{i,k}']
        col_s = var_index[f's_{i,k}']
        W_idex += [(row, col_y), (row, col_s)]
        W_values += [1, 1]

        Tk_idex.append((row, i))
        Tk_values.append(-kappa[i])

    # Demand constraints
    for row, k in enumerate(range(P), start=row + 1):
        W_idex += [(row, var_index[f'y_{i, k}']) for i in range(G)]
        W_values += [1] * G

    W = sp.coo_matrix((W_values, tuple(zip(*W_idex))), shape=(m, n))
    T_k = sp.coo_matrix((Tk_values, tuple(zip(*Tk_idex))), shape=(m, G))

    q = secondS_costs
    q = list(q.values()) + [0] * (G * P)
    q_k = np.array(q).reshape((1, n))

    return W, h_k, T_k, q_k, var_index


def create_SS_d(G=3, P=3):
    q, s = discrete_instances()
    SStages = []
    for si, p in s.items():
        W, h_k, T_k, q_k, var_index = build_ss_matrix(G, P, si, q)
        sp = SecondStage(W=W, h_k=h_k, T_k=T_k, q_k=q_k, var_index=var_index, prob=p)
        SStages.append(sp)

    return SStages


def create_SS_c(n_insts=5000, G=3, P=3, new=True):
    global q, dem, cap
    if new:
        q, dem, cap = continuous_instances()

    SStages = []
    for i in range(n_insts):
        s = tuple([dem[k].rvs() for k in range(G)] + [cap[k].rvs() for k in range(P)])
        W, h_k, T_k, q_k, var_index = build_ss_matrix(G, P, s, q)
        sp = SecondStage(W=W, h_k=h_k, T_k=T_k, q_k=q_k, var_index=var_index, prob=1 / n_insts)
        SStages.append(sp)

    return SStages


def create_FS(G=3):

    c = np.random.randint(10, size=(1, G))
    b = (c * 10).reshape((G, 1))
    b = np.vstack((b, -2000 * b))
    A = np.vstack((np.eye(G), -np.eye(G)))

    fs = FirstStage(A=A, b=b, c=c)

    return fs


def build_TSSP_d(multicut=False):
    G=3
    P=3
    fs = create_FS(G)
    ssps = create_SS_d(G, P)

    tssp = TwoStageSP(fs, ssps, verbose=True, multi_cut=multicut)
    return tssp


def discrete_experiments():
    G=3
    P=3
    fs = create_FS(G)
    ssps = create_SS_d(G, P)

    tssp = TwoStageSP(fs, ssps, verbose=True)

    l_feas, l_opt, l_v = tssp.solve(multi_cut=False)
    print(f'L sol:\nx:\t{tssp.x_hat}\ntheta:\t{tssp.theta_hat}')
    m_feas, m_opt, m_v = tssp.solve(multi_cut=True)
    print(f'MC sol:\n{tssp.x_hat}\ntheta:\t{tssp.theta_hat @ tssp.probabilities}')


def cont_experiments():
    G=3
    P=3

    n, M, N = 5000, 100, 15000
    alpha = 0.01
    timers = dict()

    t = Timer('first stage creation')

    t.start()
    fs = create_FS(G)
    t.stop()
    timers['fs'] = t

    t = Timer('second stage creation', verbose=True)
    t.start()
    ssps = create_SS_c(n, G, P)
    tssp = TwoStageSP(fs, ssps, verbose=True)
    t.stop()
    timers['ss'] = t

    tssp.solve(multi_cut=False)
    # print(f'L sol:\nx:\t{tssp.x_hat}\ntheta:\t{tssp.theta_hat}')

    t = Timer('ub ss creation', verbose=True)
    t.start()
    ub_sstages = create_SS_c(N, new=False)
    t.stop()
    timers['ub_ss'] = t

    t = Timer('lb ss creation', verbose=True)
    t.start()
    lb_samples = []
    for i in range(M):
        sample = create_SS_c(n, new=False)
        lb_samples.append(sample)

    t.stop()
    timers['lb_ss'] = t

    # tssp.confidence_interval(lower_bound_samples=lb_samples, upper_bound_sample=ub_sstages, alpha=alpha)

    vals, mean, sigma, ub = tssp.upper_bound(ub_sstages, alpha=alpha)
    print(mean, sigma, ub)
    sns.displot(data=vals, kde=True)
    plt.xlabel(r"$Q(\hat{x}, \xi)$")
    plt.show()
    vals, mean, sigma, lb = tssp.lower_bound(lb_samples, alpha=alpha)
    print(mean, sigma, lb)
    sns.displot(data=vals, kde=True)
    plt.xlabel(r"$\hat{f}_n$")
    plt.show()


if __name__ == '__main__':
    np.random.seed(1788980)
    cont_experiments()


