import numpy as np


def norm(x):
	return np.sqrt(np.dot(x.T, x))[0][0]
