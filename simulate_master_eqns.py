import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.populations import TransitionRates
from epiforecast.epiplots import plot_master_eqns
from epiforecast.contact_network import ContactNetwork

import numpy as np
import networkx as nx


# define the contact network [networkx.graph] - haven't done this yet
contact_graph = nx.watts_strogatz_graph(1000, 12, 0.1, 1)
network = ContactNetwork.from_networkx_graph(contact_graph)
population = network.get_node_count()
# give the ensemble size (is this required)
ensemble_size = 10

# ------------------------------------------------------------------------------
#  create transition rates and populate the ensemble
latent_periods = 3.0
community_infection_periods = 3.0
hospital_infection_periods = 5.0
hospitalization_fraction = 0.01 
community_mortality_fraction = 0.001
hospital_mortality_fraction = 0.01
transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates = TransitionRates.from_samplers(
            population=population,
            lp_sampler=latent_periods,
            cip_sampler=community_infection_periods,
            hip_sampler=hospital_infection_periods,
            hf_sampler=hospitalization_fraction,
            cmf_sampler=community_mortality_fraction,
            hmf_sampler=hospital_mortality_fraction)
    transition_rates.calculate_from_clinical()
    transition_rates_ensemble.append(transition_rates)

transmission_rate = 10*np.ones([ensemble_size,1])

# WARNING: Do not call the following line, it is only for the kinetic model simulator.
#network.set_transition_rates_for_kinetic_model(transition_rates)


# ------------------------------------------------------------------------------
# create object with all parameters at once
start_time=0.0
master_eqn_ensemble = MasterEquationModelEnsemble(population = population,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = transmission_rate,
                                                  hospital_transmission_reduction = 1.0,
                                                  ensemble_size = ensemble_size,
                                                  start_time = start_time)
print('Created master equations object  -----------------------------------------------------')

# ------------------------------------------------------------------------------
# Fifth test: simulate the epidemic through the master equations
np.random.seed(1)

I_perc = 0.01
states_ensemble = np.zeros([ensemble_size, 5 * population])

for mm, member in enumerate(master_eqn_ensemble.ensemble):
    infected = np.random.choice(population, replace = False, size = int(population * I_perc))
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.

    states_ensemble[mm, : ] = np.hstack((S, I, H, R, D))

#simulate without closure
master_eqn_ensemble.set_states_ensemble(states_ensemble)
master_eqn_ensemble.set_mean_contact_duration(network.get_edge_weights())
res1 = master_eqn_ensemble.simulate(100, min_steps = 20, closure = None)
#simulate on coarse mesh without closure
master_eqn_ensemble.set_start_time(start_time)
master_eqn_ensemble.set_mean_contact_duration(network.get_edge_weights())
master_eqn_ensemble.set_states_ensemble(states_ensemble)
res2 = master_eqn_ensemble.simulate(100, min_steps = 2, closure = None)

#simulate on fine mesh with closure

master_eqn_ensemble.set_start_time(start_time)
master_eqn_ensemble.set_mean_contact_duration(network.get_edge_weights())
master_eqn_ensemble.set_states_ensemble(states_ensemble)
res3 = master_eqn_ensemble.simulate(100, min_steps = 200)
print('Simulation done!')




