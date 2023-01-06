import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np

from epiforecast.user_base import (FullUserGraphBuilder,
                                   FractionalUserGraphBuilder,
                                   ContiguousUserGraphBuilder)
from epiforecast.contact_network import ContactNetwork

np.random.seed(123)

################################################################################
# create contact network #######################################################
################################################################################
# 1) from nx function:

#contact_graph = nx.watts_strogatz_graph(100000, 12, 0.1, 1)
#network = ContactNetwork.from_networkx_graph(contact_graph)

# 2) Or create from file:
edges_filename = os.path.join('..', 'data', 'networks',
                              'edge_list_SBM_1e4_nobeds.txt')
groups_filename = os.path.join('..', 'data', 'networks',
                               'node_groups_SBM_1e4_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)

print("Network loaded from files:")
print("edges:".ljust(17)       + edges_filename)
print("node groups:".ljust(17) + groups_filename)

################################################################################
# create a full user network ###################################################
################################################################################
full_user_network = network.build_user_network_using(FullUserGraphBuilder())

print("")
print("User base: Full")
print("number of nodes:", full_user_network.get_node_count())
print("number of edges:", full_user_network.get_edge_count())

################################################################################
# create a user base from a random fraction of the population ##################
################################################################################
user_fraction = 0.01
the_one_percent = network.build_user_network_using(
        FractionalUserGraphBuilder(user_fraction))

print("")
print("User base:", user_fraction, "fraction of nodes, randomly chosen")
print("number of nodes:", the_one_percent.get_node_count())
print("number of edges:", the_one_percent.get_edge_count())

################################################################################
# create a user base from a Contiguous region (neighbor method) ################
################################################################################
user_fraction = 0.01
contiguous_one_percent = network.build_user_network_using(
        ContiguousUserGraphBuilder(user_fraction,
                                   method="neighbor",
                                   seed_user=None))

print("")
print("User base:",
      user_fraction,
      "fraction of nodes, chosen using neighbor method")
print("number of nodes:", contiguous_one_percent.get_node_count())
print("number of edges:", contiguous_one_percent.get_edge_count())


