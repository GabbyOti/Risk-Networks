# Author : Alex Gramfort, <alexandre.gramfort@inria.fr>

import numpy as np
from numba import njit, _helperlib


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _copy_np_state(r, ptr):
    """
    Copy state of Numpy random *r* to Numba state *ptr*.
    """
    ints, index = r.get_state()[1:3]
    _helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))
    return ints, index


def _copyback_np_state(r, ptr):
    """
    Copy state of of Numba state *ptr* to Numpy random *r*
    """
    index, ints = _helperlib.rnd_get_state(ptr)
    r.set_state(('MT19937', ints, index, 0, 0.0))


def get_np_state_ptr():
    return _helperlib.rnd_get_np_state_ptr()


class use_numba_random(object):
    """Decorator that gets the current of numpy random, then
    sets it to numba and then puts the new state of numba
    to numpy to decoration the function of not does not
    affect the output."""
    def __init__(self, random_state=None):
        self.random_state = random_state

    def __call__(self, func):
        def new_func(*args, **kwargs):
            r = check_random_state(self.random_state)
            ptr = get_np_state_ptr()
            _copy_np_state(r, ptr)
            out = func(*args, **kwargs)
            _copyback_np_state(r, ptr)
            return out
        return new_func


def test_no_decorator():
    """Test that shows the problem when not using a decorator."""
    np.random.seed(0)
    r1_ok, r2_ok = np.random.rand(), np.random.rand()

    # Define a numba function using np.random
    @njit()
    def generate_random():
        return np.random.rand()

    np.random.seed(0)
    r1 = generate_random()
    r2 = np.random.rand()
    assert r1 != r1_ok
    assert r2 == r1_ok  # Numba does not increment the state of np.random
    assert r2 != r2_ok


def test_global_random():
    """Test using a global numpy random state."""
    np.random.seed(0)
    r = check_random_state(None)
    r1_ok, r2_ok = r.rand(), r.rand()

    np.random.seed(0)
    r = check_random_state(None)

    # Define a numba function decorated to sync it's seed
    # with a random state or current np.random
    @use_numba_random(r)
    @njit()
    def generate_random():
        return np.random.rand()

    r1 = generate_random()
    r2 = r.rand()
    assert r1 == r1_ok
    assert r2 == r2_ok


def test_random_state():
    """Test using a numpy RandomState."""
    r = check_random_state(42)
    r1_ok, r2_ok = r.rand(), r.rand()

    r = check_random_state(42)

    # Define a numba function decorated to sync it's seed
    # with a random state or current np.random
    @use_numba_random(r)
    @njit()
    def generate_random():
        return np.random.rand()

    r1 = generate_random()
    r2 = r.rand()
    assert r1 == r1_ok
    assert r2 == r2_ok


if __name__ == '__main__':
    test_no_decorator()
    test_global_random()
    test_random_state()
