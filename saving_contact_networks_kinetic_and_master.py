import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from numba import set_num_threads

set_num_threads(1)

from epiforecast.populations import TransitionRates
from epiforecast.samplers import AgeDependentConstant

from epiforecast.scenarios import  random_epidemic

from epiforecast.epiplots import plot_ensemble_states, plot_scalar_parameters, plot_epidemic_data
from epiforecast.contact_network import ContactNetwork
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService
from epiforecast.measurements import Observation, DataObservation, DataNodeObservation
from epiforecast.utilities import seed_numba_random_state
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.risk_simulator_initial_conditions import deterministic_risk

#
# Set random seeds for reproducibility
#

# Both numpy.random and random are used by the KineticModel.
seed = 212212

np.random.seed(seed)
random.seed(seed)

# set numba seed

seed_numba_random_state(seed)

#
# Load and construct an example network
#

edges_filename = os.path.join('..', 'data', 'networks',
                              'edge_list_SBM_1e3_nobeds.txt')
groups_filename = os.path.join('..', 'data', 'networks',
                               'node_groups_SBM_1e3_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)
population = network.get_node_count()
populace = network.get_nodes()


start_time = 0.0
minute = 1 / 60 / 24
hour = 60 * minute
simulation_length = 30
print("We first create an epidemic for",
      simulation_length,
      "days, then we solve the master equations forward for this time")

# Clinical parameters of an age-distributed population
#

age_distribution =[0.21, 0.4, 0.25, 0.08, 0.06]
health_workers_subset = [1, 2] # which age groups to draw from for h-workers
assert sum(age_distribution) == 1.0
network.draw_and_set_age_groups(age_distribution, health_workers_subset)

# We process the clinical data to determine transition rates between each epidemiological state,
latent_periods = 3.7
community_infection_periods = 3.2
hospital_infection_periods = 5.0
hospitalization_fraction = AgeDependentConstant([0.002,  0.01,   0.04, 0.076,  0.16])
community_mortality_fraction = AgeDependentConstant([ 1e-4,  1e-3,  0.001,  0.07,  0.015])
hospital_mortality_fraction = AgeDependentConstant([0.019, 0.073,  0.193, 0.327, 0.512])

transition_rates = TransitionRates.from_samplers(
        population=network.get_node_count(),
        lp_sampler=latent_periods,
        cip_sampler=community_infection_periods,
        hip_sampler=hospital_infection_periods,
        hf_sampler=hospitalization_fraction,
        cmf_sampler=community_mortality_fraction,
        hmf_sampler=hospital_mortality_fraction,
        distributional_parameters=network.get_age_groups())

transition_rates.calculate_from_clinical() 

network.set_transition_rates_for_kinetic_model(transition_rates)

community_transmission_rate = 12.0

#
# Setup the the epidemic simulator
#
static_contact_interval = 3 * hour

health_service = HealthService(original_contact_network = network,
                               health_workers = network.get_health_workers())

mean_contact_lifetime=0.5*minute
hospital_transmission_reduction = 0.1

min_inception_rate = 2
max_inception_rate = 22

network.set_lambdas(min_inception_rate,max_inception_rate)

epidemic_simulator = EpidemicSimulator(
                 contact_network =network,
     community_transmission_rate = community_transmission_rate,
 hospital_transmission_reduction = hospital_transmission_reduction,
         static_contact_interval = static_contact_interval,
           mean_contact_lifetime = mean_contact_lifetime,
              day_inception_rate = max_inception_rate,
            night_inception_rate = min_inception_rate,
                  health_service = health_service,
                      start_time = start_time
                                      )

# Create storage for networks and data
epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

time = start_time

statuses = random_epidemic(population,
                           populace,
                           fraction_infected=0.01,
                           seed=seed)


epidemic_simulator.set_statuses(statuses)



#
# First we run and save the epidemic
#
fig, axes = plt.subplots(1, 3, figsize = (16, 4))
time_trace = np.arange(start_time,simulation_length,static_contact_interval)
statuses_sum_trace = [[population-int(0.01*population), 0, int(0.01*population),0,0,0]]

for i in range(int(simulation_length/static_contact_interval)):

    network = epidemic_simulator.run(stop_time = epidemic_simulator.time + static_contact_interval,
                                     current_network = network)
    #save the start time network and statuses
    epidemic_data_storage.save_network_by_start_time(contact_network=network, start_time=time)
    epidemic_data_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)
    
    #update the statuses and time
    statuses = epidemic_simulator.kinetic_model.current_statuses
    time=epidemic_simulator.time

    #save the statuses at the new time
    epidemic_data_storage.save_end_statuses_to_network(end_time=time, end_statuses=statuses)

    
    statuses_sum_trace.append([epidemic_simulator.kinetic_model.statuses['S'][-1],
                               epidemic_simulator.kinetic_model.statuses['E'][-1],
                               epidemic_simulator.kinetic_model.statuses['I'][-1],
                               epidemic_simulator.kinetic_model.statuses['H'][-1],
                               epidemic_simulator.kinetic_model.statuses['R'][-1],
                               epidemic_simulator.kinetic_model.statuses['D'][-1]]) 

axes = plot_epidemic_data(population = population,
                       statuses_list = statuses_sum_trace,
                                axes = axes,
                          plot_times = time_trace)

plt.savefig('kinetic_and_master.png', rasterized=True, dpi=150)

#
# reset the time to the start of the simulation
#

time = start_time

#
# Set up the master equations
#

ensemble_size = 1

transition_rates_ensemble = []
for i in range(ensemble_size):
    transition_rates_ensemble.append(transition_rates)

#set transmission_rates
community_transmission_rate_ensemble = community_transmission_rate*np.ones([ensemble_size,1]) 


master_eqn_ensemble = MasterEquationModelEnsemble(population = population,
                                                  transition_rates = transition_rates_ensemble,
                                                  transmission_rate = community_transmission_rate_ensemble,
                                                  hospital_transmission_reduction = hospital_transmission_reduction,
                                                  ensemble_size = ensemble_size,
                                                  start_time = start_time)

loaded_data = epidemic_data_storage.get_network_from_start_time(start_time = time)
statuses = loaded_data.start_statuses


states_ensemble = deterministic_risk(populace,
                                     statuses,
                                     ensemble_size = ensemble_size)


master_eqn_ensemble.set_states_ensemble(states_ensemble)

#
# Run the master equations forward on the loaded networks
#
states_trace_ensemble=np.zeros([ensemble_size,5*population,time_trace.size])
states_trace_ensemble[:,:,0] = states_ensemble

for i in range(int(simulation_length/static_contact_interval)):

    loaded_data=epidemic_data_storage.get_network_from_start_time(start_time=time)
    master_eqn_ensemble.set_mean_contact_duration(loaded_data.contact_network.get_edge_weights()) # contact duration stored on network
 
    states_ensemble = master_eqn_ensemble.simulate(static_contact_interval, min_steps = 25)
    
    #at the update the time
    time = time + static_contact_interval
    master_eqn_ensemble.set_states_ensemble(states_ensemble)
    
    states_trace_ensemble[:,:,i] = states_ensemble

axes = plot_ensemble_states(population,
                            states_trace_ensemble,
                            time_trace,
                            axes = axes,
                            xlims = (0.0, simulation_length),
                            a_min = 0.0)
    
plt.savefig('kinetic_and_master.png', rasterized=True, dpi=150)


