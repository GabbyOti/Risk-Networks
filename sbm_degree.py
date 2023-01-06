import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
import math
import os
import itertools
import collections
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from matplotlib import rcParams, patches

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 245    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

# example taken from http://sociograph.blogspot.com/2012/11/visualizing-adjacency-matrices-in-python.html

def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = plt.figure(figsize=(fig_height, fig_height)) # in inches
    plt.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
        
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    ax = plt.gca()
    for partition in partitions:
        current_idx = 0
        for i in range(len(partition)):
            module = partition[i]
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=colors[i],
                                          linewidth="1"))
            current_idx += len(module)
    plt.tight_layout()

    
edge_list = np.loadtxt('edge_list_SBM_1e3.txt')

node_identifier = np.loadtxt('node_identifier_SBM_1e3.txt', dtype = str)

nodes = node_identifier[:,0].astype(np.int)
identifiers = node_identifier[:,1]

hospital_beds  = nodes[identifiers == 'HOSP']
health_workers = nodes[identifiers == 'HCW']
community      = nodes[identifiers == 'CITY']

node_list = [hospital_beds, health_workers, community]
print(len(nodes))

G = nx.Graph()

G.add_edges_from(edge_list)

draw_adjacency_matrix(G, partitions = [node_list], colors = ["red", "blue", "green"])

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

xx = np.linspace(10, 90, 100)

plt.figure()

plt.text(30, 0.015, r'$-2.5$')

plt.plot(deg, np.asarray(cnt)/sum(cnt), '.', markersize = 3)

plt.plot(xx, 0.6*1e2*xx**-2.5, 'k', linewidth = 1)

plt.xlabel(r'$k$')
plt.ylabel(r'$P(k)$')

plt.xlim([1, 10**2])
plt.ylim([10**-4,1])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('degree_distribution.png', dpi = 300)
plt.show()

plt.close()


