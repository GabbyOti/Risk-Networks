#!python3 --

import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np
import random 

from timeit import default_timer as timer
from numba import set_num_threads

from epiforecast.contact_network import ContactNetwork
from epiforecast.populations import TransitionRates
from epiforecast.samplers import GammaSampler, AgeDependentBetaSampler, AgeDependentConstant
from epiforecast.epidemic_simulator import EpidemicSimulator
from epiforecast.health_service import HealthService

from epiforecast.scenarios import random_epidemic
from epiforecast.utilities import seed_three_random_states

################################################################################
# constants ####################################################################
################################################################################
SAVE_FLAG = True
PLOT_FLAG = False
SIMULATION_PATH = os.path.join('..', '..', 'data', 'simulation_data')

minute = 1 / 60 / 24
hour = 60 * minute
day = 1.0

start_time = -3 / 24
simulation_length = 3 # number of days

latent_period = {
        'mean': 3.7}
community_infection_period = {
        'mean': 3.2}
hospital_infection_period = {
        'mean': 5.0}
community_transmission_rate = {
        'mean': 12.0}

hospital_transmission_reduction = 0.1

λ_min = 2  # minimum contact rate
λ_max = 22 # maximum contact rate

static_contact_interval = 3 * hour
mean_contact_lifetime = 0.5 * minute

# 5 age groups (0-17, 18-44, 45-64, 65-74, >=75) and their respective rates
age_distribution = [0.207, 0.400, 0.245, 0.083, 0.065]
health_workers_subset = [1, 2] # which age groups to draw from for h-workers
age_dep_h      = [0.002   ,  0.010  ,  0.040,  0.076,  0.160]
age_dep_d      = [0.000001,  0.00001,  0.001,  0.007,  0.015]
age_dep_dprime = [0.019   ,  0.073  ,  0.193,  0.327,  0.512]

assert sum(age_distribution) == 1.0

################################################################################
# initialization ###############################################################
################################################################################
# numba
set_num_threads(1)
      
# Set random seeds for reproducibility
seed = 1
seed_three_random_states(seed)

# contact network ##############################################################
nxgraph = nx.barabasi_albert_graph(1000, 
                                   2, 
                                   seed = 1)

network = ContactNetwork.from_networkx_graph(nxgraph)

network.draw_and_set_age_groups(age_distribution, health_workers_subset)
network.set_lambdas(λ_min, λ_max)
                       
# stochastic model #############################################################
# transition rates a.k.a. independent rates (σ, γ etc.)
# constructor takes clinical parameter samplers which are then used to draw real
# clinical parameters, and those are used to calculate transition rates
transition_rates = TransitionRates.from_samplers(
        network.get_node_count(),
        latent_period['mean'],
        community_infection_period['mean'],
        hospital_infection_period['mean'],
        AgeDependentConstant(age_dep_h),
        AgeDependentConstant(age_dep_d),
        AgeDependentConstant(age_dep_dprime),
        network.get_age_groups())

transition_rates.calculate_from_clinical()

network.set_transition_rates_for_kinetic_model(transition_rates)

health_service = HealthService(
        network,
        network.get_health_workers())

epidemic_simulator = EpidemicSimulator(
        network,
        community_transmission_rate = community_transmission_rate['mean'],
        hospital_transmission_reduction = hospital_transmission_reduction,
        static_contact_interval = static_contact_interval,
        mean_contact_lifetime = mean_contact_lifetime,
        day_inception_rate = λ_max,
        night_inception_rate = λ_min,
        health_service = health_service,
        start_time = start_time)

################################################################################
# run simulation ###############################################################
################################################################################

statuses = random_epidemic(
        network.get_node_count(),
        network.get_nodes(),
        fraction_infected=0.05,
        seed=seed)

epidemic_simulator.set_statuses(statuses)

# Within the epidemic_simulator:
# health_service discharge and admit patients [changes the contact network]
# contact_simulator run [changes the mean contact duration on the network]
# set the new contact rates on the network
# run the kinetic model [kinetic produces the current statuses used as data]
network = epidemic_simulator.run(
    stop_time = epidemic_simulator.time + 100,
    current_network = network)

kinetic_model = epidemic_simulator.kinetic_model

################################################################################
# save #########################################################################
################################################################################
if SAVE_FLAG:
    np.savetxt(
            os.path.join(SIMULATION_PATH, 'simple_epidemic.txt'),
            np.c_[
                kinetic_model.times,
                kinetic_model.statuses['S'],
                kinetic_model.statuses['E'],
                kinetic_model.statuses['I'],
                kinetic_model.statuses['H'],
                kinetic_model.statuses['R'],
                kinetic_model.statuses['D']],
            header = 'S E I H R D seed: %d'%seed)


