import os, sys; sys.path.append(os.path.join(".."))

import networkx as nx
import numpy as np
from matplotlib import pylab as pl

from epiforecast.user_base import (FullUserGraphBuilder,
                                   FractionalUserGraphBuilder,
                                   ContiguousUserGraphBuilder,
                                   contiguous_indicators)
from epiforecast.contact_network import ContactNetwork

np.random.seed(123)

#plot graphs? NB plotting is very slow for >1000 nodes
plot_figs = True
write_graphs = False
write_files = False

################################################################################
# create contact network #######################################################
################################################################################
# 1) from nx function:

#contact_graph = nx.watts_strogatz_graph(100000, 12, 0.1, 1)
#network = ContactNetwork.from_networkx_graph(contact_graph)

# 2) Or create from file:
edges_filename = os.path.join('..', 'data', 'networks',
                              'edge_list_SBM_1e3_nobeds.txt')
groups_filename = os.path.join('..', 'data', 'networks',
                               'node_groups_SBM_1e3_nobeds.json')

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

if write_graphs:
    nx.write_gexf(full_user_network.get_graph(),
                  '../data/networks/full_user_graph.gexf')
    nx.write_edgelist(full_user_network.get_graph(),
                      '../data/networks/full_user_graph.csv',
                      data=False)

################################################################################
# create a user base from a random fraction of the population ##################
################################################################################
user_fraction = 0.05
fractional_user_network = network.build_user_network_using(
        FractionalUserGraphBuilder(user_fraction))

print("")
print("User base:", user_fraction, "fraction of nodes, randomly chosen")
print("number of nodes:", fractional_user_network.get_node_count())
print("number of edges:", fractional_user_network.get_edge_count())

if write_graphs:
    nx.write_gexf(fractional_user_network.get_graph(),
                  '../data/networks/fractional_user_graph.gexf')
    nx.write_edgelist(fractional_user_network.get_graph(),
                      '../data/networks/fractional_user_graph.csv',
                      data=False)

(interior,
 boundary,
 mean_exterior_neighbors,
 edge_indicator_list,
 node_indicator_list) = contiguous_indicators(
         network.get_graph(),
         fractional_user_network.get_graph())

print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)

if write_files:
    np.savetxt('../data/networks/fractional_indicator_edge_list.csv',
               np.c_[edge_indicator_list],
               fmt="%s",
               header='Source Target Property',
               comments='')
    np.savetxt('../data/networks/fractional_indicator_node_list.csv',
               np.c_[node_indicator_list],
               fmt="%s",
               header='Node Property',
               comments='')

################################################################################
# create a user base from a Contiguous region (neighbor method) ################
################################################################################
user_fraction = 0.05
neighbor_user_network = network.build_user_network_using(
        ContiguousUserGraphBuilder(user_fraction,
                                   method="neighbor",
                                   seed_user=None))

print("")
print("User base:",
      user_fraction,
      "fraction of nodes, chosen using neighbor method")
print("number of nodes:", neighbor_user_network.get_node_count())
print("number of edges:", neighbor_user_network.get_edge_count())

if write_graphs:
    nx.write_gexf(neighbor_user_network.get_graph(),
                  '../data/networks/neighbor_user_graph.gexf')
    nx.write_edgelist(neighbor_user_network.get_graph(),
                      '../data/networks/neighbor_user_graph.csv',
                      data=False)

(interior,
 boundary,
 mean_exterior_neighbors,
 edge_indicator_list,
 node_indicator_list) = contiguous_indicators(
         network.get_graph(),
         neighbor_user_network.get_graph())

print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)

if write_files:
    np.savetxt('../data/networks/neighbor_indicator_edge_list.csv',
               np.c_[edge_indicator_list],
               fmt="%s",
               header='Source Target Property',
               comments='')
    np.savetxt('../data/networks/neighbor_indicator_node_list.csv',
               np.c_[node_indicator_list],
               fmt="%s",
               header='Node Property',
               comments='')

################################################################################
# create a user base from a Contiguous region (clique method) ##################
################################################################################
user_fraction = 0.05
clique_user_network = network.build_user_network_using(
        ContiguousUserGraphBuilder(user_fraction,
                                   method="clique",
                                   seed_user=None))

print("")
print("User base:",
      user_fraction,
      "fraction of nodes, chosen using clique method")
print("number of nodes:", clique_user_network.get_node_count())
print("number of edges:", clique_user_network.get_edge_count())

if write_graphs:
    nx.write_gexf(clique_user_network.get_graph(),
                  '../data/networks/clique_user_graph.gexf')
    nx.write_edgelist(clique_user_network.get_graph(),
                      '../data/networks/clique_user_graph.csv',
                      data=False)

(interior,
 boundary,
 mean_exterior_neighbors,
 edge_indicator_list,
 node_indicator_list) = contiguous_indicators(
     network.get_graph(),
     clique_user_network.get_graph())

print("number of interior nodes:", interior)
print("number of boundary nodes:", boundary)
print("average exterior neighbours of boundary node:", mean_exterior_neighbors)

if write_files:
    np.savetxt('../data/networks/clique_indicator_edge_list.csv',
               np.c_[edge_indicator_list],
               fmt="%s",
               header='Source Target Property',
               comments='')
    np.savetxt('../data/networks/clique_indicator_node_list.csv',
               np.c_[node_indicator_list],
               fmt="%s",
               header='Node Property',
               comments='')

################################################################################
# plot graphs ##################################################################
################################################################################
if plot_figs:
    # figure 1: neighborhood based network
    pl.figure(1,figsize=(10, 10), dpi=100)
    nx.draw_networkx(network.get_graph(),
                     node_color='k',
                     with_labels=False,
                     node_size=10,
                     alpha=0.05)
    nx.draw_networkx(neighbor_user_network.get_graph(),
                     node_color='r',
                     with_labels=False,
                     node_size=10,
                     alpha=0.8)
    pl.title('neighborhood based contact network', fontsize=20)
    pl.savefig('neighbor_network.pdf')

    # figure 2: clique based network
    pl.figure(2,figsize=(10, 10), dpi=100)
    nx.draw_networkx(network.get_graph(),
                     node_color='k',
                     with_labels=False,
                     node_size=10,
                     alpha=0.05)
    nx.draw_networkx(clique_user_network.get_graph(),
                     node_color='r',
                     with_labels=False,
                     node_size=10,
                     alpha=0.8)
    pl.title('clique based contact network',fontsize=20)
    pl.savefig('clique_network.pdf')

    # figure 3: random subset network
    pl.figure(3,figsize=(10, 10), dpi=100)
    nx.draw_networkx(network.get_graph(),
                     node_color='k',
                     with_labels=False,
                     node_size=10,
                     alpha=0.05)
    nx.draw_networkx(fractional_user_network.get_graph(),
                     node_color='r',
                     with_labels=False,
                     node_size=10,
                     alpha=0.8)
    pl.title('random subset contact network',fontsize=20)
    pl.savefig('random_network.pdf')


