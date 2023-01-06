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

from epiforecast.scenarios import random_epidemic
from epiforecast.epidemic_data_storage import StaticIntervalDataSeries
from epiforecast.contact_network import ContactNetwork
#
# Set random seeds for reproducibility
#
seed = 212212
np.random.seed(seed)

contact_graph = nx.barabasi_albert_graph(1000, 10)
network = ContactNetwork.from_networkx_graph(contact_graph)

time = 0.0
static_contact_interval = 0.25
simulation_length = 1


epidemic_data_storage = StaticIntervalDataSeries(static_contact_interval)

population = network.get_node_count()
populace = network.get_nodes()
statuses = random_epidemic(population,
                           populace,
                           fraction_infected=0.01,
                           seed=seed)

print("saving all the networks")
current_infected = [node for node in populace if statuses[node] == 'I']
print("infected at time", time, current_infected)

for i in range(int(simulation_length/static_contact_interval)):
    #save network and start time statuses
    epidemic_data_storage.save_network_by_start_time(contact_network=network, start_time=time)
    epidemic_data_storage.save_start_statuses_to_network(start_time=time, start_statuses=statuses)
    
    #pretend we 'simulate' forward, by creating a new network.
    contact_graph = nx.barabasi_albert_graph(1000, 10)
    network = ContactNetwork.from_networkx_graph(contact_graph)
    population = network.get_node_count()
    populace = network.get_nodes()
    statuses = random_epidemic(population,
                               populace,
                               fraction_infected=0.01,
                               seed=(i+2)*seed)
    time = time + static_contact_interval
    current_infected = [node for node in populace if statuses[node] == 'I']
    print("infected at time", time, current_infected)

    #save end time statuses
    epidemic_data_storage.save_end_statuses_to_network(end_time=time, end_statuses = statuses)

print(" ")
print("loading the networks backwards by end time")
for i in range(int(simulation_length/static_contact_interval)):
    load_data = epidemic_data_storage.get_network_from_end_time(end_time=time)
    current_infected = [node for node in load_data.contact_network.get_nodes() if load_data.end_statuses[node] == 'I']
    print("infected at time", load_data.end_time, current_infected)
    time = time - static_contact_interval
                        
print(" ")
print("loading the networks forwards by start time")
for i in range(int(simulation_length/static_contact_interval)):
    load_data = epidemic_data_storage.get_network_from_start_time(start_time=time)
    current_infected = [node for node in load_data.contact_network.get_nodes() if load_data.start_statuses[node] == 'I']
    print("infected at time", load_data.start_time, current_infected)
    time = time + static_contact_interval
