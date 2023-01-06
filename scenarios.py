import numpy as np
import contextlib

from .samplers import AgeDependentBetaSampler, GammaSampler
from .populations import TransitionRates

n_states = 5

# Index guide:
#
# 0: Susceptible
# 1: Exposed
# 2: Infected
# 3: Hospitalized
# 4: Resistant
# (5: Decesased)
susceptible  = s = 0
infected     = i = 1
hospitalized = h = 2
resistant    = r = 3
deceased     = d = 4

# Our state is a 1D vector. Thus, accessing the values for a particular state requires
# slicing into this vector. These functions return the appropritate subranges for each state.
def susceptible_indices(population):  return np.arange(start = 0 * population, stop = 1 * population)
def infected_indices(population):     return np.arange(start = 1 * population, stop = 2 * population)
def hospitalized_indices(population): return np.arange(start = 2 * population, stop = 3 * population)
def resistant_indices(population):    return np.arange(start = 3 * population, stop = 4 * population)
def deceased_indices(population):     return np.arange(start = 4 * population, stop = 5 * population)

@contextlib.contextmanager
def temporary_seed(seed):
    """
    Temporarily changes the global random state, and then changes it back.

    Ref: https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """

    state = np.random.get_state()

    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)


def random_epidemic(
        population,
        nodes,
        fraction_infected,
        fraction_exposed=0,
        seed=None):
    """
    Returns a status dictionary associated with a random infection
    within a population associated with node_identifiers.
    """
    local_rng = np.random.default_rng(seed)

    n_initial_infected = int(np.round(fraction_infected * population))
    n_initial_exposed = int(np.round(fraction_exposed * population))

    statuses = {node: 'S' for node in nodes}

    initial_infected = local_rng.choice(nodes, size=n_initial_infected, replace=False)

    uninfected = [ node for node in filter(lambda n: n not in initial_infected, nodes) ]
    initial_exposed = local_rng.choice(uninfected, size=n_initial_exposed, replace=False)

    statuses.update({node: 'I' for node in initial_infected})
    statuses.update({node: 'E' for node in initial_exposed})

    return statuses



def NYC_transition_rates(population_network, random_seed=1234):
    """
    Returns transition rates for a community of size `population`
    whose statistics vaguely resemble the clinical statistics of 
    New York City, NY, USA.
    """
    raise NotImplementedError("this function should be refactored")

    age_distribution = [ 
                         0.21,  # 0-17 years
                         0.40,  # 18-44 years
                         0.25,  # 45-64 years
                         0.08   # 65-75 years
                        ]
    
    age_distribution.append(1 - sum(age_distribution)) # 75+
    
    assign_ages(population_network, age_distribution)

    with temporary_seed(random_seed):

        transition_rates = TransitionRates(population_network,

                          latent_periods = 3.7,
             community_infection_periods = 3.2,
              hospital_infection_periods = 5.0,
                hospitalization_fraction = AgeDependentConstant([0.002, 0.01, 0.04, 0.075, 0.16]),
            community_mortality_fraction = AgeDependentConstant([1e-4, 1e-3, 0.003, 0.01, 0.02]),
             hospital_mortality_fraction = AgeDependentConstant([0.019, 0.075, 0.195, 0.328, 0.514]),

        )
    
    return transition_rates

def midnight_on_Tuesday(kinetic_model, 
                            percent_infected = 0.1,
                             percent_exposed = 0.05,
                                 random_seed = 1234,
                        ):
    """
    Returns an `np.array` corresponding to the epidemiological state of a population
    "at midnight on Tuesday".

    Each person can be in 1 of 5 states, so `state.shape = (5, population)`.
    """

    population = kinetic_model.population

    n_infected = int(np.round(percent_infected * population))
    n_exposed = int(np.round(percent_exposed * population))

    # Generate random indices for infected and exposed
    with temporary_seed(random_seed):
        infected_nodes = np.random.choice(population, n_infected)
        exposed_nodes = np.random.choice(population, n_exposed)

    state = np.zeros((n_states * population,))

    i = infected_indices(population)
    s = susceptible_indices(population)

    # Some are infected...
    i_infected = i[infected_nodes]
    state[i_infected] = 1 

    # and everyone else is susceptible.
    state[s] = 1 - state[i]

    # (except those who are exposed).
    s_exposed = s[exposed_nodes]
    state[s_exposed] = 0

    # (We may want to identify a hospitalized group as well.)

    return state

def percent_infected_at_midnight_on_Tuesday():
    return 0.01

def ensemble_transition_rates_at_midnight_on_Tuesday(ensemble_size, population, random_seed=1234):
    transition_rates = []

    for i in range(ensemble_size):
        random_seed += 1
        transition_rates.append(NYC_transition_rates(population, random_seed=random_seed))

    return transition_rates

def ensemble_transmission_rates_at_midnight_on_Tuesday(ensemble_size, random_seed=1234):

    with temporary_seed(random_seed):
        transmission_rates = np.random.uniform(0.04, 0.06, ensemble_size)

    return transmission_rates

def randomly_infected_ensemble(ensemble_size, population, percent_infected, random_seed=1234):
    """
    Returns an ensemble of states of 5 x population. In each member of the ensemble,
    `percent_infected` of the population is infected.

    Args
    ----

    ensemble_size: The number of ensemble members.

    population: The population. Each ensemble has a state of size `5 * population`. The total
                size of the ensemble of states is `ensemble_size x 5 * population`.

    percent_infected: The percent of people who are infected in each ensemble member.                    

    random_seed: A random seed so that the outcome is reproducible.
    """

    n_infected = int(np.round(population * percent_infected))

    # Extract the indices corresponding to infected and susceptible states
    s = susceptible_indices(population)
    i = infected_indices(population) # is needed later

    # Initialize states with susceptible = 1.
    states = np.zeros([ensemble_size, n_states * population])
    states[:, s] = 1

    with temporary_seed(random_seed):
        for m in range(ensemble_size):

            # Select random indices from a list of indices = [0, population)
            randomly_infected = np.random.choice(population, size=n_infected)

            # Translate random indices to indices of the infected state
            i_randomly_infected = i[randomly_infected]
            s_randomly_infected = s[randomly_infected]

            # Infect the people
            states[m, i_randomly_infected] = 1
            states[m, s_randomly_infected] = 0

    return states


