import os, sys; sys.path.append(os.path.join(".."))

import random 
import numpy as np
from numba import njit

from epiforecast.utilities import seed_three_random_states

@njit
def get_random_numba():
    return np.random.random()

print("\nWith seed 123")

seed_three_random_states(123)

print("{:>24s}: {}".format("random.random",       random.random()))
print("{:>24s}: {}".format("numpy.random.random", np.random.random()))
print("{:>24s}: {}".format("numba",               get_random_numba()))

print("\nWith no seed")

print("{:>24s}: {}".format("random.random",       random.random()))
print("{:>24s}: {}".format("numpy.random.random", np.random.random()))
print("{:>24s}: {}".format("numba",               get_random_numba()))

print("\nWith seed 123 again")

seed_three_random_states(123)

print("{:>24s}: {}".format("random.random",       random.random()))
print("{:>24s}: {}".format("numpy.random.random", np.random.random()))
print("{:>24s}: {}".format("numba",               get_random_numba()))
