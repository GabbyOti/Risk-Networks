import os
import numpy as np
from numba import njit, prange, set_num_threads
from numba.np.ufunc.parallel import _get_thread_id
from numba.core.config import NUMBA_DEFAULT_NUM_THREADS

@njit
def random_printing(n):
    nums = np.zeros(n)
    for i in range(n):
        nums[i] = np.random.randint(0, high=10)

    print(nums)

@njit(parallel=True)
def random_multithreaded_printing(n=NUMBA_DEFAULT_NUM_THREADS):
    nums = np.zeros(n)
    for i in prange(n):
        nums[i] = np.random.randint(0, high=10)

    print(nums)

@njit(parallel=True)
def random_multithreaded_printing_no_array(n=NUMBA_DEFAULT_NUM_THREADS):
    for i in prange(n):
        print(np.random.randint(0, high=10))

@njit
def set_seed(seed):
    np.random.seed(seed)

@njit(parallel=True)
def set_many_seeds(seed, n=NUMBA_DEFAULT_NUM_THREADS):
    for i in prange(n):
        np.random.seed(seed)

if __name__ == '__main__':

    print("Random numbers, no seeding:")
    random_printing(4)
    random_printing(4)

    print("\nRandom numbers, seeding, serial execution:")
    set_seed(123)
    random_printing(4)

    set_seed(123)
    random_printing(4)
 
    print("\nRandom numbers, single-thread seeding, multithreaded execution with",
          NUMBA_DEFAULT_NUM_THREADS, "threads")

    for i in range(10):
        set_seed(123)
        random_multithreaded_printing()

    print("\nRandom numbers, multithreaded seeding (?), multithreaded execution with",
          NUMBA_DEFAULT_NUM_THREADS, "threads")

    for i in range(10):
        set_many_seeds(123)
        random_multithreaded_printing()

    set_num_threads(1)
    print("\nRandom numbers, seeding, multithreaded functions with 1 thread.")

    for i in range(10):
        set_many_seeds(123)
        random_multithreaded_printing()
