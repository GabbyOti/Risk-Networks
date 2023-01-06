import numpy as np
import networkx as nx
from .simulation import Gillespie_simple_contagion

def print_initial_statuses(statuses,population):
    #use default dict here
    for i in range(population-1):
        print(statuses[i], end=" ")

    print("")

def print_statuses(statuses):

    for i in sorted(list(statuses.keys())):
        print(statuses[i], end=" ")

    print("")


class KineticModel:
    """
    A class to implement a Kinetic Monte-Carlo solver on a provided network.
    """
    def __init__(
            self,
            diagram_indep,
            diagram_neigh,
            start_time = 0.0):
        """
        Constructor

        Input:
            diagram_indep (nx.DiGraph): diagram with independent rates
            diagram_neigh (nx.DiGraph): diagram with neighbor-dependent rates
        """
        # TODO read the following from a Glossary class
        # What statuses to return from Gillespie simulation
        self.return_statuses = ('S', 'E', 'I', 'H', 'R', 'D')

        self.diagram_indep = diagram_indep
        self.diagram_neigh = diagram_neigh

        self.current_time = start_time
        self.current_statuses = None # must be set by set_statuses

        self.times = []
        self.statuses = {s: [] for s in self.return_statuses}

    def set_mean_contact_duration(self, mean_contact_duration):
        """
        Set the weights of self.contact_network, which correspond to the mean contact
        duration over a given time interval.

        Args
        ----

        mean_contact_duration (np.array) : np.array of the contact duration for each edge
                                           in the contact network.
        """
        # TODO DB: I put it here to avoid merge conflicts with Greg's extension
        # of EpidemicSimulator which seems to make use of this method; should be
        # refactored in the future (this is already implemented in
        # ContactNetwork.set_edge_weights)
        raise NotImplementedError("this method should be refactored")

        weights = {tuple(edge): mean_contact_duration[i]
                   for i, edge in enumerate(nx.edges(self.contact_network))}

        nx.set_edge_attributes(self.contact_network, values=weights, name='exposed_by_infected')
        nx.set_edge_attributes(self.contact_network, values=weights, name='exposed_by_hospitalized')

    def set_statuses(self, statuses):
        self.current_statuses = statuses

    def simulate(
            self,
            graph,
            time_interval,
            initial_statuses=None):
        """
        Run the Gillespie solver on a given graph

        Input:
            graph (nx.Graph): graph object with node and edge attributes
            time_interval (float): integration time
            initial_statuses (dict): initial conditions of the form:
                {node_number : node_status}
        """
        if initial_statuses is None:
            initial_statuses = self.current_statuses

        res = Gillespie_simple_contagion(graph,
                                         self.diagram_indep,
                                         self.diagram_neigh,
                                         initial_statuses,
                                         self.return_statuses,
                                         return_full_data=True,
                                         tmin=self.current_time,
                                         tmax=self.current_time + time_interval)

        new_times, new_statuses = res.summary()

        self.current_time += time_interval

        self.times.extend(new_times)

        for s in self.return_statuses:
            self.statuses[s].extend(new_statuses[s])

        self.current_statuses = res.get_statuses(time=self.current_time)

        return self.current_statuses


