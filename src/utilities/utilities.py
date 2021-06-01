import numpy as np


def norm(x):
    try:
        return np.sqrt(np.dot(x.T, x))[0][0]
    except TypeError:
        x = np.array(x)
        return np.sqrt(np.dot(x.T, x))[0]
    except IndexError:
        return np.sqrt(np.dot(x.T, x))


def check_kwargs(str_arg, default, kwargs):
    if str_arg in kwargs.keys():
        arg = kwargs[str_arg]
    else:
        arg = default
    return arg
