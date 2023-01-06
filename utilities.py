import numpy as np
import random
from numba import njit

# Utilities for seeding random number generators

@njit
def seed_numba_random_state(seed):
    np.random.seed(seed)

def seed_three_random_states(seed):
    random.seed(seed)
    np.random.seed(seed)
    seed_numba_random_state(seed)

def not_involving(nodes):
    """
    Filters edges that connect to `nodes`.
    """
    def edge_doesnt_involve_any_nodes(edge, nodes=nodes):
        return edge[0] not in nodes and edge[1] not in nodes

    return edge_doesnt_involve_any_nodes

@njit
def normalize(edge):
    """
    Normalize a symmetric edge by placing largest node id first.
    """
    n, m = edge
    if m > n: # switch
        n, m = m, n
    return n, m

def complement_mask(
        indices,
        array_size):
    """
    Get mask of complement `indices` in 0..(array_size-1)

    Input:
        indices (list),
                (np.array): either of the two:
            - boolean array of size `array_size`
            - array with indices, each of which is in 0..(array_size-1)
    Output:
        mask (np.array): boolean array of complement indices
    """
    mask = np.ones(array_size, dtype=bool)
    mask[indices] = False
    return mask

def mask_by_compartment(
        states,
        compartment):
    """
    Get mask of indices for which state is equal to `compartment`

    Input:
        states (dict): a mapping node -> state
        compartment (char): which compartment to return mask for
    Output:
        mask (np.array): boolean array of indices
    """
    states_array = np.fromiter(states.values(), dtype='<U1')
    mask = (states_array == compartment)
    return mask

def compartments_count(states):
    """
    Get number of nodes in each compartment

    Input:
        states (dict): a mapping node -> state
    Output:
        counts (np.array): (6,) array of integers
    """
    n_S = np.count_nonzero(mask_by_compartment(states, 'S'))
    n_E = np.count_nonzero(mask_by_compartment(states, 'E'))
    n_I = np.count_nonzero(mask_by_compartment(states, 'I'))
    n_H = np.count_nonzero(mask_by_compartment(states, 'H'))
    n_R = np.count_nonzero(mask_by_compartment(states, 'R'))
    n_D = np.count_nonzero(mask_by_compartment(states, 'D'))

    return np.array((n_S, n_E, n_I, n_H, n_R, n_D))

def shuffle(states):
    """
    Shuffle states preserving the number of nodes in each compartment

    Input:
        states (dict): a mapping node -> state
    Output:
        shuffled_states (dict): a mapping node -> state, shuffled
    """
    states_array = np.fromiter(states.values(), dtype='<U1')
    np.random.shuffle(states_array) # in-place
    return { node: state for node, state in zip(states.keys(), states_array) }

def dict_slice(
        states,
        nodes):
    """
    Get number of nodes in each compartment

    Input:
        states (dict): a mapping node -> state
        nodes (np.array): (n_nodes,) array of node indices to take a slice of
    Output:
        states_slice (dict): a mapping node -> state (with node in nodes)
    """
    return { node: states[node] for node in nodes }


