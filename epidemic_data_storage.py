import copy
import numpy as np

class StaticIntervalData:
    """
    A container to hold a static contact network,
    its weights and the data (statuses) at the start and end times.
    """
    def __init__(
            self,
            contact_network,
            start_time,
            end_time):
        """
        Constructor

        Input:
            contact_network (object): object (to save) that represents network
            start_time (float): start time of the interval where network holds
            end_time (float): end time of the interval where network holds
        """
        self.contact_network = copy.deepcopy(contact_network)

        self.start_time = start_time
        self.end_time   = end_time

        # start and end statuses of the kinetic model
        self.start_statuses = None
        self.end_statuses   = None

    def set_end_statuses(self, end_statuses):
        self.end_statuses = copy.deepcopy(end_statuses)

    def set_start_statuses(self, start_statuses):
        self.start_statuses = copy.deepcopy(start_statuses)

def start_time(start_end_time):
    return start_end_time.start

class StaticIntervalDataSeries:
    """
    A container to hold a series of StaticIntervalData objects. It stores the networks as a
    dictionary with keys given by a named tuple start_end_time which will set/get networks based
    on the provided start_time, end_time or both
    """
    class StartEndTime:
        """
        A simple class that serves as a key for dictionary; equivalent to
        namedtuple with keys 'start' and 'end'
        """
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def __init__(self, static_contact_interval, max_networks=np.inf):
        """
        Args
        ----
        static_contact_interval (float): the fixed duration at which the network is static. (so we can
                                         deduce end time from start time, start_time from end time).
        """
        self.static_network_series = {}
        self.static_contact_interval = static_contact_interval
        self.max_networks = max_networks

        # anything < 0.5 * static_contact_interval would do
        self.time_tolerance = 0.1 * static_contact_interval

    def create_new_network(self, contact_network, start_time, end_time):
        return StaticIntervalData(contact_network,
                                  start_time,
                                  end_time)

    def find_interval_from_end_time(self, end_time):
        return next(filter(lambda keys: abs(keys.end - end_time) < self.time_tolerance,
                           self.static_network_series.keys()))

    def find_interval_from_start_time(self, start_time):
        return next(filter(lambda keys: abs(keys.start - start_time) < self.time_tolerance,
                           self.static_network_series.keys()))

    def limit_static_network_series_length(self):
        """
        Deletes an entry from the static network series if the number of saved networks
        exceeds max_networks - 1.
        """
        if len(self.static_network_series) > self.max_networks - 1: # delete first network
            first_start_end = min(self.static_network_series.keys(), key=start_time)    
            del self.static_network_series[first_start_end]

    def save_network_by_start_time(self, contact_network, start_time):

        self.limit_static_network_series_length()

        end_time = start_time + self.static_contact_interval
        start_end_time = self.StartEndTime(start=start_time, end=end_time)

        self.static_network_series[start_end_time] = self.create_new_network(contact_network,
                                                                             start_time,
                                                                             end_time)

    def save_network_by_end_time(self, contact_network, end_time):

        self.limit_static_network_series_length()

        start_time = end_time - self.static_contact_interval
        start_end_time = self.StartEndTime(start=start_time, end=end_time)

        self.static_network_series[start_end_time] = self.create_new_network(contact_network,
                                                                             start_time,
                                                                             end_time)

    def save_end_statuses_to_network(self, end_time, end_statuses):
        start_end_time = self.find_interval_from_end_time(end_time)
        self.static_network_series[start_end_time].set_end_statuses(end_statuses)

    def save_start_statuses_to_network(self, start_time, start_statuses):
        start_end_time = self.find_interval_from_start_time(start_time)
        self.static_network_series[start_end_time].set_start_statuses(start_statuses)

    def get_network_from_start_time(self, start_time):
        start_end_time = self.find_interval_from_start_time(start_time)
        return self.static_network_series[start_end_time]

    def get_network_from_end_time(self, end_time):
        start_end_time = self.find_interval_from_end_time(end_time)
        return self.static_network_series[start_end_time]


