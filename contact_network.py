import numpy as np
import json
import scipy.sparse as scspa
import networkx as nx

from epiforecast.utilities import complement_mask
from .contact_simulator import diurnal_inception_rate

class ContactNetwork:
    """
    Store and mutate a contact network
    """

    HEALTH_WORKERS_INDEX = 0
    COMMUNITY_INDEX      = 1

    # TODO extract into Glossary class
    AGE_GROUP = 'age_group'

    LAMBDA_MIN = 'minimum_contact_rate'
    LAMBDA_MAX = 'maximum_contact_rate'
    LAMBDA_INTEGRATED = 'integrated_contact_rate'
    WJI = 'edge_weights'

    E_TO_I = 'exposed_to_infected'
    I_TO_H = 'infected_to_hospitalized'
    I_TO_R = 'infected_to_resistant'
    I_TO_D = 'infected_to_deceased'
    H_TO_R = 'hospitalized_to_resistant'
    H_TO_D = 'hospitalized_to_deceased'

    @classmethod
    def from_networkx_graph(
            cls,
            graph,
            check_labels_are_0N=True):
        """
        Create an object from a nx.Graph object

        Input:
            graph (nx.Graph): an object to use as a contact network graph
            check_labels_are_0N (boolean): check that node labels are 0..N-1

        Output:
            contact_network (ContactNetwork): initialized object
        """
        # sorted_graph is identical to graph, but with nodes in ascending order
        sorted_graph = cls.__create_sorted_networkx_graph_from(graph.nodes)
        sorted_graph.update(graph)

        nodes = sorted_graph.nodes
        node_groups = {
                ContactNetwork.HEALTH_WORKERS_INDEX : np.array([]),
                ContactNetwork.COMMUNITY_INDEX      : np.array(nodes) }

        return cls(sorted_graph, node_groups, check_labels_are_0N)

    @classmethod
    def from_files(
            cls,
            edges_filename,
            groups_filename,
            convert_labels_to_0N=True):
        """
        Create an object from files that contain edges and groups

        Input:
            edges_filename (str): path to a txt-file with edges
            groups_filename (str): path to a json-file with node groups
            convert_labels_to_0N (boolean): convert node labels to 0..N-1

        Output:
            contact_network (ContactNetwork): initialized object
        """
        edges       = cls.__load_edges_from(edges_filename)
        node_groups = cls.__load_node_groups_from(groups_filename)

        # in the following, (1) enforce the ascending order of the nodes,
        # (2) add edges, and (3) (possibly) weed out missing labels (for
        # example, there might be no node '0', so every node 'j' gets mapped to
        # 'j-1', and the edges are remapped accordingly)
        #
        # this whole workaround is needed so that we can then simply say that
        # nodes 0..44, for instance, are health workers (instead of dealing with
        # permutations and such)
        graph = cls.__create_sorted_networkx_graph_from(edges) # (1)
        graph.add_edges_from(edges)                            # (2)
        if convert_labels_to_0N:                               # (3)
            graph = nx.convert_node_labels_to_integers(graph, ordering='sorted')

        contact_network = cls(graph, node_groups, convert_labels_to_0N)
        contact_network.set_edge_weights(1.0) # this is done by networkx anyway

        return contact_network

    def __init__(
            self,
            graph,
            node_groups,
            check_labels_are_0N):
        """
        Constructor

        Input:
            graph (nx.Graph): graph object with node and edge attributes
            node_groups (dict): a mapping group_id -> arrays_of_nodes
            check_labels_are_0N (boolean): check that node labels are 0..N-1
        """
        self.graph = graph
        self.node_groups = node_groups

        self.__check_correct_format(check_labels_are_0N)

    @staticmethod
    def __create_sorted_networkx_graph_from(nodes_or_edges):
        """
        Create a nx.Graph object with nodes in ascending order

        Input:
            nodes_or_edges (iterable): a list of nodes or edges

        Output:
            sorted_graph (nx.Graph): an object with nodes in ascending order
        """
        nodes = np.unique(nodes_or_edges)

        sorted_graph = nx.Graph()
        sorted_graph.add_nodes_from(nodes)

        return sorted_graph

    @staticmethod
    def __load_edges_from(filename):
        """
        Load edges from a txt-file

        Input:
            filename (str): path to a txt-file with edges

        Output:
            edges (np.array): (n_edges,2) array of edges
        """
        edges = np.loadtxt(filename, dtype=int, comments='#')
        return edges

    @staticmethod
    def __load_node_groups_from(filename):
        """
        Load node groups from a json-file

        Input:
            filename (str): path to a json-file with total number of nodes in
                            each group
        Output:
            node_groups (dict): a mapping group_id -> arrays_of_nodes
        """
        with open(filename) as f:
            node_group_numbers = json.load(f)

        n_health_workers = node_group_numbers['n_health_workers']
        n_community      = node_group_numbers['n_community']

        health_workers = np.arange(n_health_workers)
        community = np.arange(n_health_workers, n_health_workers + n_community)

        node_groups = {
                ContactNetwork.HEALTH_WORKERS_INDEX : health_workers,
                ContactNetwork.COMMUNITY_INDEX      : community }

        return node_groups

    def __check_correct_format(
            self,
            check_labels_are_0N):
        """
        Check whether the graph is in the correct format

        The following is checked:
            1. nodes are sorted in ascending order
            2. total number of nodes is equal to "community + health workers"
        If `check_labels_are_0N` is true then also check
            3. all nodes are integers in the range 0..N-1

        Input:
            check_labels_are_0N (boolean): check that node labels are 0..N-1

        Output:
            None
        """
        n_checks = 3
        correct_format = np.ones(n_checks, dtype=bool)
        nodes = self.get_nodes()
        n_nodes = self.get_node_count()

        # 1. check
        if not np.all(nodes[:-1] <= nodes[1:]): # if not "ascending order"
            correct_format[0] = False

        # 2. check
        n_health_workers = self.get_health_workers().size
        n_community = self.get_community().size
        if n_health_workers + n_community != n_nodes:
            correct_format[1] = False

        # 3. check
        if check_labels_are_0N:
            if not np.array_equal(nodes, np.arange(n_nodes)):
                correct_format[2] = False

        if not correct_format.all():
            raise ValueError(
                    self.__class__.__name__
                    + ": graph format is incorrect; "
                    + "checks are: "
                    + str(correct_format))

    def __convert_array_to_dict(
            self,
            array):
        """
        Convert numpy array to dictionary with node indices as keys

        Input:
            array (np.array): (n_nodes,) array of values

        Output:
            dictionary (dict): a sequential mapping node -> value
        """
        return { node: array[idx] for idx, node in enumerate(self.get_nodes()) }

    def get_health_workers(self):
        """
        Get health worker nodes

        Output:
            health_workers (np.array): (K,) array of node indices
        """
        return self.node_groups[ContactNetwork.HEALTH_WORKERS_INDEX]

    def get_community(self):
        """
        Get community nodes

        Output:
            community (np.array): (K,) array of node indices
        """
        return self.node_groups[ContactNetwork.COMMUNITY_INDEX]

    def get_node_count(self):
        """
        Get the total number of nodes

        Output:
            n_nodes (int): total number of nodes
        """
        return self.graph.number_of_nodes()

    def get_edge_count(self):
        """
        Get the total number of edges

        Output:
            n_edges (int): total number of edges
        """
        return self.graph.number_of_edges()

    # TODO hide implementation, expose interfaces (i.e. delete get_graph)
    def get_graph(self):
        """
        Get the graph

        Output:
            graph (nx.Graph): graph object with node and edge attributes
        """
        return self.graph

    def get_neighbors(
            self,
            nodes):
        """
        Get neighbors of nodes

        Input:
            nodes (np.array): (K1,) array of nodes whose neighbors to retrieve

        Output:
            neighbors (np.array): (K2,) array of unique neighbors, sorted
        """
        adjacency_matrix = self.get_edge_weights()
        sliced_adjacency_matrix = adjacency_matrix[nodes,:]
        unsorted_neighbors = sliced_adjacency_matrix.nonzero()[1]

        return np.unique(unsorted_neighbors)

    def get_nodes(self):
        """
        Get all nodes of the graph

        Output:
            nodes (np.array): (n_nodes,) array of node indices
        """
        return np.array(self.graph.nodes)

    def get_edges(self):
        """
        Get all edges of the graph

        Output:
            edges (np.array): (n_edges,2) array of pairs of node indices
        """
        return np.array(self.graph.edges)

    def get_incident_edges(
            self,
            node):
        """
        Get incident edges of a node

        Input:
            node (int): node whose incident edges to retrieve

        Output:
            edges (list): list of tuples, each of which is an incident edge
        """
        return list(self.graph.edges(node))

    def get_edge_weights(self):
        """
        Get edge weights of the graph as a scipy.sparse matrix

        Output:
            edge_weights (scipy.sparse.csr.csr_matrix): adjacency matrix
        """
        return nx.to_scipy_sparse_matrix(self.graph, weight=ContactNetwork.WJI)

    def get_age_groups(self):
        """
        Get the age groups of the nodes

        Output:
            age_groups (np.array): (n_nodes,) array of age groups
        """
        age_groups_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.AGE_GROUP)
        return np.fromiter(age_groups_dict.values(), dtype=int)

    def get_lambdas(self):
        """
        Get λ_min and λ_max attributes of the nodes

        Output:
            λ_min (np.array): (n_nodes,) array of values
            λ_max (np.array): (n_nodes,) array of values
        """
        λ_min_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.LAMBDA_MIN)
        λ_max_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.LAMBDA_MAX)
        return (np.fromiter(λ_min_dict.values(), dtype=float),
                np.fromiter(λ_max_dict.values(), dtype=float))

    def get_lambda_integrated(self):
        """
        Get λ_integrated attribute of the nodes

        Output:
            λ_integrated (np.array): (n_nodes,) array of values
        """
        λ_integrated_dict = nx.get_node_attributes(
            self.graph, name=ContactNetwork.LAMBDA_INTEGRATED)
        return np.fromiter(λ_integrated_dict.values(), dtype=float)    

    def set_lambdas(
            self,
            λ_min,
            λ_max):
        """
        Set λ_min and λ_max attributes to the nodes

        Input:
            λ_min (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values
            λ_max (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.set_lambda_min(λ_min)
        self.set_lambda_max(λ_max)

    def set_lambda_min(
            self,
            λ_min):
        """
        Set λ_min attribute to the nodes

        Input:
            λ_min (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.__set_node_attributes(λ_min, ContactNetwork.LAMBDA_MIN)

    def set_lambda_max(
            self,
            λ_max):
        """
        Set λ_max attribute to the nodes

        Input:
            λ_max (int),
                  (float): constant value to be assigned to all nodes
                  (dict): a mapping node -> value
                  (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.__set_node_attributes(λ_max, ContactNetwork.LAMBDA_MAX)

    def set_lambda_integrated(
            self,
            λ_integrated):
        """
        Set λ_integrated attribute to the nodes

        Input:
            λ_integrated (np.array): (n_nodes,) array of values

        Output:
            None
        """
        self.__set_node_attributes(λ_integrated, ContactNetwork.LAMBDA_INTEGRATED)


    def __set_node_attributes(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (int),
                   (float): constant value to be assigned to all nodes
                   (dict): a mapping node -> value
                   (np.array): (n_nodes,) array of values
            name (str): name of the attributes

        Output:
            None
        """
        if isinstance(values, (int, float, dict)):
            self.__set_node_attributes_const_dict(values, name)
        elif isinstance(values, np.ndarray):
            self.__set_node_attributes_array(values, name)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this type of argument is not supported: "
                    + values.__class__.__name__)

    def __set_node_attributes_const_dict(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (int),
                   (float): constant value to be assigned to all nodes
                   (dict): a mapping node -> value
            name (str): name of the attributes

        Output:
            None
        """
        nx.set_node_attributes(self.graph, values=values, name=name)

    def __set_node_attributes_array(
            self,
            values,
            name):
        """
        Set node attributes of the graph by name

        Input:
            values (np.array): (n_nodes,) array of values
            name (str): name of the attributes

        Output:
            None
        """
        values_dict = self.__convert_array_to_dict(values)
        self.__set_node_attributes_const_dict(values_dict, name)

    def set_transition_rates_for_kinetic_model(
            self,
            transition_rates):
        """
        Set transition rates (exposed to infected etc.) as node attributes

        Note: these transition rates are only intended to be used by
        KineticModel; hence, this method does not really belong here, and should
        be implemented in KineticModel instead

        Input:
            transition_rates (TransitionRates): object with instance variables:
                exposed_to_infected
                infected_to_hospitalized
                infected_to_resistant
                infected_to_deceased
                hospitalized_to_resistant
                hospitalized_to_deceased
        Output:
            None
        """
        self.__set_node_attributes(
                transition_rates.exposed_to_infected,
                ContactNetwork.E_TO_I)
        self.__set_node_attributes(
                transition_rates.infected_to_hospitalized,
                ContactNetwork.I_TO_H)
        self.__set_node_attributes(
                transition_rates.infected_to_resistant,
                ContactNetwork.I_TO_R)
        self.__set_node_attributes(
                transition_rates.infected_to_deceased,
                ContactNetwork.I_TO_D)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_resistant,
                ContactNetwork.H_TO_R)
        self.__set_node_attributes(
                transition_rates.hospitalized_to_deceased,
                ContactNetwork.H_TO_D)

    def set_edge_weights(
            self,
            edge_weights):
        """
        Set edge weights of the graph

        Input:
            edge_weights (int),
                         (float): constant value to be assigned to all edges
                         (dict): a mapping edge -> weight
        Output:
            None
        """
        nx.set_edge_attributes(
                self.graph, values=edge_weights, name=ContactNetwork.WJI)

    def add_edges(
            self,
            edges):
        """
        Add edges to the graph

        Input:
            edges (list): list of tuples, each of which is an edge
        Output:
            None
        """
        self.graph.add_edges_from(edges)

    def remove_edges(
            self,
            edges):
        """
        Remove edges from the graph

        Input:
            edges (list): list of tuples, each of which is an edge
        Output:
            None
        """
        self.graph.remove_edges_from(edges)

    @staticmethod
    def __draw_from(
            distribution,
            size):
        """
        Draw from `distribution` an array of numbers 0..k of size `size`

        Input:
            distribution (list),
                         (np.array): (k,) discrete distribution (must sum to 1)
            size (int): number of samples to draw

        Output:
            samples (np.array): (size,) array of samples
        """
        n_groups = len(distribution)
        samples = np.random.choice(n_groups, p=distribution, size=size)

        return samples

    def __draw_community_from(
            self,
            distribution):
        """
        Draw from distribution an array of numbers 0..k of size n_community

        Input:
            distribution (list),
                         (np.array): (k,) discrete distribution (must sum to 1)

        Output:
            samples (np.array): (n_community,) array of samples
        """
        n_community = self.get_community().size

        return self.__draw_from(distribution, n_community)

    def __draw_health_workers_from(
            self,
            distribution,
            health_workers_subset):
        """
        Draw from (normalized with a specified subset) distribution numbers 0..k

        Input:
            distribution (list),
                         (np.array): (k,) discrete distribution
            health_workers_subset (list),
                                  (np.array): subset of age groups (e.g. [1,2])

        Output:
            samples (np.array): (n_health_workers,) array of samples
        """
        complement_subset = complement_mask(health_workers_subset,
                                            len(distribution))

        # create a distribution and make it sum to one
        distribution_health_workers = np.copy(distribution)
        distribution_health_workers[complement_subset] = 0.0
        distribution_health_workers /= distribution_health_workers.sum()

        n_health_workers = self.get_health_workers().size

        return self.__draw_from(distribution_health_workers, n_health_workers)

    def draw_and_set_age_groups(
            self,
            distribution,
            health_workers_subset):
        """
        Draw from `distribution` and set age groups to the nodes

        Input:
            distribution (list),
                         (np.array): discrete distribution (should sum to 1)
            health_workers_subset (list),
                                  (np.array): subset of age groups (e.g. [1,2])

        Output:
            None
        """
        age_groups_community      = self.__draw_community_from(distribution)
        age_groups_health_workers = self.__draw_health_workers_from(
                distribution,
                health_workers_subset)

        n_nodes        = self.get_node_count()
        health_workers = self.get_health_workers()
        community      = self.get_community()

        age_groups = np.empty(n_nodes)
        age_groups[health_workers] = age_groups_health_workers
        age_groups[community]      = age_groups_community

        self.__set_node_attributes(age_groups, ContactNetwork.AGE_GROUP)

    def isolate(
            self,
            sick_nodes,
            λ_isolation=1.0):
        """
        Isolate sick nodes by setting λ's of sick nodes to λ_isolation

        Input:
            sick_nodes (np.array): (n_sick,) array of indices of sick nodes

        Output:
            None
        """
        (λ_min, λ_max) = self.get_lambdas()

        λ_min[sick_nodes] = λ_isolation
        λ_max[sick_nodes] = λ_isolation

        self.set_lambdas(λ_min, λ_max)

    def build_user_network_using(
            self,
            user_graph_builder):
        """
        Build user network using provided builder

        Input:
            user_graph_builder (callable): an object to build user_graph

        Output:
            user_network (ContactNetwork): built user network
        """
        user_graph = user_graph_builder(self.graph)
        return ContactNetwork.from_networkx_graph(user_graph, False)

    def update_from(
            self,
            contact_network):
        """
        Update the graph from another object whose graph is a supergraph

        The contact_network.graph should have at least the same nodes as
        self.graph, plus maybe additional ones.

        Input:
            contact_network (ContactNetwork): object to update from

        Output:
            None
        """
        nodes = self.get_nodes()
        contact_graph = contact_network.get_graph()
        contact_subgraph = contact_graph.subgraph(nodes)

        # nx.Graph.update does not delete edges; hence this workaround
        self.graph = self.__create_sorted_networkx_graph_from(nodes)
        self.graph.update(contact_subgraph)

    # TODO extract into a separate class
    @staticmethod
    def generate_diagram_indep():
        """
        Generate diagram with independent transition rates

        Output:
            diagram_indep (nx.DiGraph): diagram with independent rates
        """
        diagram_indep = nx.DiGraph()
        diagram_indep.add_node('S')
        diagram_indep.add_edge(
                'E', 'I', rate=1, weight_label=ContactNetwork.E_TO_I)
        diagram_indep.add_edge(
                'I', 'H', rate=1, weight_label=ContactNetwork.I_TO_H)
        diagram_indep.add_edge(
                'I', 'R', rate=1, weight_label=ContactNetwork.I_TO_R)
        diagram_indep.add_edge(
                'I', 'D', rate=1, weight_label=ContactNetwork.I_TO_D)
        diagram_indep.add_edge(
                'H', 'R', rate=1, weight_label=ContactNetwork.H_TO_R)
        diagram_indep.add_edge(
                'H', 'D', rate=1, weight_label=ContactNetwork.H_TO_D)
        return diagram_indep

    # TODO extract into a separate class
    @staticmethod
    def generate_diagram_neigh(
            community_rate,
            hospital_rate):
        """
        Generate diagram with transmition rates that depend on neighbors

        Input:
            community_rate (float): rate at which infected infect susceptible
            hospital_rate (float): rate at which hospitalized infect susceptible

        Output:
            diagram_neigh (nx.DiGraph): diagram with neighbor-dependent rates
        """
        diagram_neigh = nx.DiGraph()
        diagram_neigh.add_edge(
                ('I', 'S'),
                ('I', 'E'),
                rate=community_rate,
                weight_label=ContactNetwork.WJI)
        diagram_neigh.add_edge(
                ('H', 'S'),
                ('H', 'E'),
                rate=hospital_rate,
                weight_label=ContactNetwork.WJI)
        return diagram_neigh


