from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

class UserGraphBuilder(ABC):
    """
    Abstract class for a user graph builder
    """
    @abstractmethod
    def __call__(self):
        pass

class FullUserGraphBuilder(UserGraphBuilder):
    """
    A class to store which subset of the population are being modeled by the Master Equations
    FullUserGraphBuilder is just the `full_graph`.
    """
    def __call__(
            self,
            full_graph):
        """
        Build user subgraph from the full contact graph

        Input:
            full_graph (networkx.Graph): full contact graph

        Output:
            user_graph (networkx.Graph): users (sub)graph
        """
        return full_graph


class FractionalUserGraphBuilder(UserGraphBuilder):
    """
    A class to store which subset of the population are being modeled by the
    Master Equations FractionalUserGraphBuilder takes a random subset of the
    population and contructs a subgraph of the largest component within this
    fraction.
    """
    def __init__(
            self,
            user_fraction):
        """
        Constructor

        Input:
            user_fraction (float): value in (0,1] to specify fraction
        """
        self.user_fraction = user_fraction

    def __call__(
            self,
            full_graph):
        """
        Build user subgraph from the full contact graph

        Input:
            full_graph (networkx.Graph): full contact graph

        Output:
            user_graph (networkx.Graph): users (sub)graph
        """
        scale_factor = 1.0
        magic_number = 0.9

        n_nodes = full_graph.number_of_nodes()
        n_users_pruned_limit = magic_number * self.user_fraction * n_nodes
        nodes = full_graph.nodes()

        users_pruned = set()
        while len(users_pruned) < n_users_pruned_limit:
            n_users = min(int(scale_factor * self.user_fraction * n_nodes),
                          n_nodes)
            users = np.random.choice(nodes, n_users, replace=False)
            user_graph_fractured = full_graph.subgraph(users)
            users_pruned = max(nx.connected_components(user_graph_fractured),
                               key=len)
            scale_factor *= 1.1

        return full_graph.subgraph(users_pruned)


class ContiguousUserGraphBuilder(UserGraphBuilder):
    """
    A class to store which subset of the population are being modeled by the Master Equations
    ContiguousUserGraphBuilder takes a given sized population of the user base and tries to form an island of users
    around a seed user by iteration, in each iteration we add nodes to our subgraph by either
    1) "neighbor" - [recommended] adds the neighborhood about the seed user and moving to a new user as a seed
    2) "clique" - adds the maximal clique about the seed user and  moving to a new user as a seed 
    """
    def __init__(
            self,
            user_fraction,
            method='neighbor',
            seed_user=None):
        """
        Constructor

        Input:
            user_fraction (float): value in (0,1] to specify fraction
        """
        self.user_fraction = user_fraction
        self.method = method
        self.seed_user = seed_user

    def __call__(
            self,
            full_graph):
        """
        Build user subgraph from the full contact graph

        Input:
            full_graph (networkx.Graph): full contact graph

        Output:
            user_graph (networkx.Graph): users (sub)graph
        """
        if self.seed_user is None:
            self.seed_user = np.random.choice(full_graph.nodes())

        if self.method == 'neighbor':
            new_users_generator = self.__neighbor_generator(full_graph)
        elif self.method == 'clique':
            new_users_generator = self.__clique_generator(full_graph)
        else:
            raise ValueError("unknown method, choose from: neighbor, clique")
        new_users_generator.send(None) # generator initialization, no workaround

        users = self.__choose_users_via(new_users_generator, full_graph)

        return full_graph.subgraph(users)

    def __choose_users_via(
            self,
            new_users_generator,
            full_graph):
        """
        Choose users from the full graph via provided generator of new users

        Input:
            new_users_generator (generator): generator that chooses new users
                                             based on a specified user
            full_graph (networkx.Graph): full contact graph

        Output:
            users (list): users chosen
        """
        idx = 0
        users = [self.seed_user]
        n_users_limit = int(self.user_fraction * full_graph.number_of_nodes())

        while len(users) < n_users_limit and idx<len(users):
            new_users = new_users_generator.send(users[idx])
            new_users = [user for user in filter(lambda u: u not in users, new_users)]
            users.extend(new_users)
            idx += 1

        return users

    @staticmethod
    def __neighbor_generator(full_graph):
        user = yield
        while True:
            user = (yield full_graph.neighbors(user))

    @staticmethod
    def __clique_generator(full_graph):
        node_cliques = list(nx.find_cliques(full_graph))

        user = yield
        while True:
            user = (yield node_cliques[user])


def contiguous_indicators(graph, subgraph):
    """
    A function that returns user base subgraph indicators and corresponding edge/node
    lists with attributes "exterior" and "interior".
    """
    edge_indicator_dict = {edge: "interior" if edge in subgraph.edges() else "exterior" for edge in graph.edges()}
    node_indicator_dict = {node: "interior" if node in subgraph.nodes() else "exterior" for node in graph.nodes()}

    edge_indicator_list = []
    for key, value in  zip(edge_indicator_dict.keys(), edge_indicator_dict.values()):
        edge_indicator_list.append([key[0], key[1], value])

    node_indicator_list = []
    for key, value in  zip(node_indicator_dict.keys(), node_indicator_dict.values()):
        node_indicator_list.append([key, value])

    interior_nodes=0
    boundary_nodes=0
    exterior_neighbor_count=0
    exterior_neighbor_weights = np.zeros(subgraph.number_of_nodes())
    for (idx,node) in enumerate(subgraph.nodes):

        if len(list(subgraph.neighbors(node))) == len(list(graph.neighbors(node))):
            interior_nodes+=1
        else:
            boundary_nodes+=1
            #count number of exterior neighbors of an boundary node
            exterior_neighbors=[nbr for nbr in filter(lambda nbr: nbr not in list(subgraph.neighbors(node)),list(graph.neighbors(node)))]
            exterior_neighbor_count+=len(exterior_neighbors)
            #multiplicative weights
            #exterior_neighbor_weights[idx] =  len(list(graph.neighbors(node)))/(len(list(graph.neighbors(node))) - len(exterior_neighbors)) 
            #additive weights
            exterior_neighbor_weights[idx] = len(exterior_neighbors)

    mean_neighbors_exterior = exterior_neighbor_count / (interior_nodes + boundary_nodes)
                    
    return interior_nodes, boundary_nodes, mean_neighbors_exterior, edge_indicator_list, node_indicator_list, exterior_neighbor_weights
