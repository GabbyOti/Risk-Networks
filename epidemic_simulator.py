import copy
import numpy as np
import networkx as nx

from timeit import default_timer as timer

from .contact_simulator import ContactSimulator
from .kinetic_model_simulator import KineticModel
from .utilities import not_involving

day = 1
hour = day / 24
minute = hour / 60
second = minute / 60

class EpidemicSimulator:
    """
    Simulates epidemics.
    """
    def __init__(
            self,
            contact_network,
            community_transmission_rate,
            hospital_transmission_reduction,
            static_contact_interval,
            mean_contact_lifetime,
            day_inception_rate = None,
            night_inception_rate = None,
            health_service = None,
            start_time = 0.0,
            seed = None):
        """
        Build a tool that simulates epidemics.

        Args
        ----

        contact_network (ContactNetwork): Network of community members and edges
                                          that represent possible contact
                                          between community members.

        community_transmission_rate (float): Rate of transmission of infection during interaction
                                             between people in the community.

        hospital_transmission_reduction (float): Fractional reduction of rate of transmission of
                                                 infection in hospitals relative to community.

        static_contact_interval (float): Interval over which contact between people is assumed 'static'.
                                         Rapidly fluctuating contact times are averaged over this interval
                                         and then used in kinetic_model.simulate.

        mean_contact_lifetime (float): The *mean* lifetime of a contact between people. Typical values
                                       are O(minutes).

        day_inception_rate (float): The rate of inception of new contacts between people at noon.

        night_inception_rate (float): The rate of inception of new contacts between people at midnight.

        health_service: Manages rewiring of contact_network during hospitalization.

        start_time (float): The initial time of the simulation.

        """

        self.health_service = health_service

        if health_service is None: # number of contacts cannot change; no buffer needed
            buffer_margin = 1
        else:
            buffer_margin = 1.2 # 20% margin seems conservative
            
        #calculate mean_degree for the edges
        mean_degree = np.mean([d for n, d in contact_network.get_graph().degree()])

        self.contact_simulator = ContactSimulator(contact_network.get_edges(),
                                                  mean_degree,
                                                  day_inception_rate = day_inception_rate,
                                                  night_inception_rate = night_inception_rate,
                                                  mean_event_lifetime = mean_contact_lifetime,
                                                  buffer_margin = buffer_margin,
                                                  start_time = start_time,
                                                  seed = seed)

        diagram_indep = contact_network.generate_diagram_indep()
        diagram_neigh = contact_network.generate_diagram_neigh(
                community_transmission_rate,
                community_transmission_rate * hospital_transmission_reduction)

        self.kinetic_model = KineticModel(diagram_indep = diagram_indep,
                                          diagram_neigh = diagram_neigh,
                                          start_time = start_time)

        self.static_contact_interval = static_contact_interval
        self.time = start_time

    def run(
            self,
            stop_time,
            current_network,
            verbose=False):
        """
        Run forward for t in [self.time, stop_time]

        Takes a single step when
            stop_time = self.time + self.static_contact_interval
        This method is almost pure, i.e. outputs will differ depending on the
        seeds to pseudo-random number generators; but has no side effects.

        Input:
            stop_time (float): time to run the simulation until
            current_network (ContactNetwork): current network to run on
            verbose (bool): whether to print info every step of the simulation

        Output:
            next_network (ContactNetwork): updated network
        """

        run_time = stop_time - self.time
        next_network = copy.deepcopy(current_network)

        # Number of constant steps, which are followed by a single ragged step to update to specified stop_time.
        constant_steps = int(np.floor(run_time / self.static_contact_interval))

        interval_stop_times = self.time + self.static_contact_interval * np.arange(start = 1, stop = 1 + constant_steps)

        # Step forward
        for i in range(constant_steps):

            interval_stop_time = interval_stop_times[i]

            if verbose:
                print("")
                print("")
            print("                               *** Day: {:.3f}".format(interval_stop_time))
            print("")

            #
            # Administer hospitalization
            #

            start_health_service = timer()

            if self.health_service is not None:
                (discharged_patients,
                 admitted_patients,
                 contacts_to_add,
                 contacts_to_remove) = (
                        self.health_service.discharge_and_admit_patients(
                            self.kinetic_model.current_statuses,
                            verbose))

                # TODO why are we filtering edges in what follows but not here?
                next_network.add_edges(contacts_to_add)
                next_network.remove_edges(contacts_to_remove)

                # Compile edges to add and remove from contact simulation...
                edges_to_remove = set()
                edges_to_add = set()

                current_patients = self.health_service.current_patient_addresses()

                previous_patients = (current_patients
                                     - {p.address for p in admitted_patients})
                previous_patients.update(p.address for p in discharged_patients)

                # ... ensuring that edges are not removed from previous patients (whose edges were *already* removed),
                # and ensuring that edges are not added to existing patients:
                if len(admitted_patients) > 0:
                    for patient in admitted_patients:
                        edges_to_remove.update(filter(not_involving(previous_patients), patient.community_contacts))
                        edges_to_add.update(patient.health_worker_contacts)

                if len(discharged_patients) > 0:
                    for patient in discharged_patients:
                        edges_to_remove.update(patient.health_worker_contacts)
                        edges_to_add.update(filter(not_involving(current_patients), patient.community_contacts))

            else:
                edges_to_add, edges_to_remove = set(), set()

            end_health_service = timer()

            #
            # Simulate contacts
            #

            start_contact_simulation = timer()

            current_edges  = next_network.get_edges()
            (λ_min, λ_max) = next_network.get_lambdas()
            self.contact_simulator.run(stop_time = interval_stop_time,
                                       current_edges = current_edges,
                                       nodal_day_inception_rate = λ_max,
                                       nodal_night_inception_rate = λ_min,
                                       edges_to_remove = edges_to_remove,
                                       edges_to_add = edges_to_add)

            edge_weights = self.contact_simulator.compute_edge_weights()
            next_network.set_edge_weights(edge_weights)

            λ_integrated = self.contact_simulator.compute_diurnally_averaged_nodal_activation_rate(
                nodal_day_inception_rate = λ_max,
                nodal_night_inception_rate = λ_min)
            next_network.set_lambda_integrated(λ_integrated)

            end_contact_simulation = timer()

            #
            # Run the kinetic simulation
            #

            start_kinetic_simulation = timer()

            self.kinetic_model.simulate(next_network.get_graph(),
                                        self.static_contact_interval)
            self.time += self.static_contact_interval

            end_kinetic_simulation = timer()



            if verbose:
                n_contacts = next_network.get_edge_count()
                health_service_walltime = (
                        end_health_service - start_health_service)
                contact_simulator_walltime = (
                        end_contact_simulation - start_contact_simulation)
                kinetic_model_walltime = (
                        end_kinetic_simulation - start_kinetic_simulation)
                self.print_status_report(n_contacts)
                self.print_walltimes(health_service_walltime,
                                     contact_simulator_walltime,
                                     kinetic_model_walltime)

        if self.time < stop_time: # take a final ragged stop to catch up with stop_time

            contact_duration = self.contact_simulator.mean_contact_duration(stop_time=stop_time)
            self.kinetic_model.set_mean_contact_duration(contact_duration)
            self.kinetic_model.simulate(stop_time - self.time)
            self.time = stop_time

        return next_network

    def print_status_report(
            self,
            n_contacts):
        """
        Print status report after a simulation step

        Input:
            n_contacts (int): total number of possible contacts

        Output:
            None
        """
        print("")
        print("[ Status report ]          Susceptible: {:d}".format(self.kinetic_model.statuses['S'][-1]))
        print("                               Exposed: {:d}".format(self.kinetic_model.statuses['E'][-1]))
        print("                              Infected: {:d}".format(self.kinetic_model.statuses['I'][-1]))
        print("                          Hospitalized: {:d}".format(self.kinetic_model.statuses['H'][-1]))
        print("                             Resistant: {:d}".format(self.kinetic_model.statuses['R'][-1]))
        print("                              Deceased: {:d}".format(self.kinetic_model.statuses['D'][-1]))
        print("             Current possible contacts: {:d}".format(n_contacts))
        print("               Current active contacts: {:d}".format(np.count_nonzero(~self.contact_simulator.active_contacts[:n_contacts])))

    def print_walltimes(
            self,
            hs_walltime,
            cs_walltime,
            km_walltime):
        """
        Print walltimes after a simulation step

        Input:
            hs_walltime (float): health_service walltime
            cs_walltime (float): contact_simulator walltime
            km_walltime (float): kinetic_model walltime

        Output:
            None
        """
        print("")
        print("[ Wall times ]          Health service: {:.4f} s,".format(hs_walltime))
        print("                    Contact simulation: {:.4f} s,".format(cs_walltime))
        print("                    Kinetic simulation: {:.4f} s ".format(km_walltime))
        print("")

    def set_statuses(self, statuses):
        """
        Set the statuses of the kinetic_model.
        """
        self.kinetic_model.set_statuses(statuses)
