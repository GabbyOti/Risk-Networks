import copy
import numpy as np
import networkx as nx

from .utilities import normalize, not_involving

class HealthService:
    """
    Class to represent the actions of a health service for the population,
    This class is initiated with a full contact network of health_workers
    and community known as the `original_contact_network`. Patients are
    added from this population by removing their old connectivity and
    adding new contact edges to some `assigned health workers:

    |----------------network------------------|

    [health_worker 1: 'S']---[community 1: 'S']
                          \ /
                           X-[community 2: 'I']
                          / \
    [health_worker 2: 'S']---[community 3: 'H']

    |-----------------------------------------|

    The primary method of this class is:

    population_network = `discharge_and_admit_patients(current_statuses)`

    The output is a network identified with only human beings, known as the `population network`
    and the connectivity would be given by the following:

    (admit) Any `community` or `health_worker` node that (and not already a `patient`)
    with status 'H' (hospitalized)  will lose their edges connecting them to current
    neighbours and gain the edges of some health workers. If there is capacity to do so.
    We refer to them as a `patient`.

    (discharge) Any `patient` with status !='H' (i.e it is now resistant 'R' or deceased 'D') will lose
    the edges to their current `health_worker` neighbours and regain thier original neighbours.

    We perform first (discharge), then (admit) to gain new patients in the function.

    For example. Applying this function to the world network above yields the following.
    1) discharge: There is no one to discharge
    2) admit: `community 3` has status H and so is placed into hospital. They lose their community edges. We denote them `patient 1`.

    |---------------------Population network---------------------|

    [Patient 1: 'H']---[health_worker 1: 'S']---[community 1: 'S']
                    \                        \ /
                     \                        /-[community 2: 'I']
                      \                      /
                       [health_worker 2: 'S']

    |------------------------------------------------------------|

    """

    def __init__(
            self,
            original_contact_network,
            health_workers,
            health_workers_per_patient=5,
            seed=None):
        """
        Constructor

        Input:
            original_contact_network (ContactNetwork): original network
            health_workers: determines health workers to recruit. Options are:
                * (int): `health_workers` are chosen randomly from the nodes of
                      `original_contact_network`
                * (list or array): denotes the address of `health_workers`

        """
        self.rng = np.random.default_rng(seed)

        self.original_contact_network = copy.deepcopy(original_contact_network)
        self.all_people = original_contact_network.get_nodes()

        self.health_workers = set(
                self.__recruit_health_workers(health_workers, self.all_people))

        self.health_workers_per_patient = health_workers_per_patient
        self.patients = set()

    def __recruit_health_workers(
            self,
            workers,
            all_people):
        """
        Choose which nodes represent health workers

        Input:
            workers (int): number of health workers that are chosen randomly
                           from all_people
                    (list),
                    (np.array): worker indices to use; returned without change
            all_people (np.array): (n_nodes,) array of node indices
        Output:
            workers_chosen (np.array): array of indices chosen for workers
        """
        if isinstance(workers, (list, np.ndarray)):
            return np.array(workers)
        elif isinstance(workers, int):
            return self.rng.choice(all_people, workers, replace=False)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this type of argument is not supported: "
                    + workers.__class__.__name__)

    def current_patient_addresses(self):
        return set(p.address for p in self.patients)

    def assign_health_workers(self, patient_address, viable_health_workers):
        """
        Assign health workers to a patient.
        """
        if self.health_workers_per_patient < len(viable_health_workers):
            # binomial degree distribution
            size = self.rng.binomial(
                    len(viable_health_workers),
                    self.health_workers_per_patient / len(viable_health_workers))

            assigned = self.rng.choice(list(viable_health_workers),
                                       size=size,
                                       replace=False)
        else:
            assigned = viable_health_workers

        health_worker_contacts = [ (patient_address, i) for i in assigned ]

        return health_worker_contacts

    def discharge_and_admit_patients(
            self,
            statuses,
            verbose=False):
        """
        Discharge and admit patients according to statuses

        Input:
            statuses (dict): mapping node -> status
            verbose (bool): whether to print info

        Output:
            discharged_patients (list): list of Patients
            admitted_patients (list): list of Patients
            contacts_to_add (list): list of tuples, each of which is an edge
            contacts_to_remove (list): list of tuples, each of which is an edge
        """
        (discharged_patients,
         discharged_hospital_contacts,
         discharged_community_contacts) = self.discharge_patients(statuses)

        (admitted_patients,
         admitted_hospital_contacts,
         admitted_community_contacts) = self.admit_patients(statuses)

        contacts_to_add    = (discharged_community_contacts
                              + admitted_hospital_contacts)
        contacts_to_remove = (admitted_community_contacts
                              + discharged_hospital_contacts)

        if verbose:
            self.print_manifest(statuses, discharged_patients, admitted_patients)

        return (discharged_patients,
                admitted_patients,
                contacts_to_add,
                contacts_to_remove)

    def discharge_patients(
            self,
            statuses):
        """
        Removes a patient from self.patients and reconnect the patient with their neighbours
        if their status is no longer H.

        Input:
            statuses (dict): mapping node -> status
        """

        discharged_patients = set()
        discharged_hospital_contacts  = []
        discharged_community_contacts = []

        for i, patient in enumerate(self.patients):
            if statuses[patient.address] != 'H': # patient is no longer hospitalized
                discharged_hospital_contacts  += patient.health_worker_contacts
                discharged_community_contacts += patient.community_contacts
                discharged_patients.add(patient)

        # Remove discharged from patient set
        self.patients = self.patients - discharged_patients

        # Filter contacts with current patients from the list of contacts to add to network
        discharged_community_contacts = [
                edge for edge in filter(not_involving(self.current_patient_addresses()),
                                        discharged_community_contacts)]

        return (discharged_patients,
                discharged_hospital_contacts,
                discharged_community_contacts)

    def admit_patients(
            self,
            statuses):
        """
        Admit patients from the community (storing their details).

        Input:
            statuses (dict): mapping node -> status
        """
        # Set of all hospitalized people
        hospitalized_people = set(
                human for human in self.all_people if statuses[human] == 'H')

        # Hospitalized health workers do not care for patients
        viable_health_workers = self.health_workers - hospitalized_people

        # Patients waiting to be admitted
        waiting_room = hospitalized_people - self.current_patient_addresses()

        admitted_patients = set()
        admitted_hospital_contacts  = []
        admitted_community_contacts = []

        # Admit patients
        for person in waiting_room:

            health_worker_contacts = self.assign_health_workers(person, viable_health_workers)

            community_contacts = self.original_contact_network.get_incident_edges(person)

            # Record patient information
            new_patient = Patient(person,
                                  community_contacts,
                                  health_worker_contacts)

            # Admit the patient
            self.patients.add(new_patient)
            admitted_patients.add(new_patient)
            admitted_hospital_contacts  += new_patient.health_worker_contacts
            admitted_community_contacts += new_patient.community_contacts

        return (admitted_patients,
                admitted_hospital_contacts,
                admitted_community_contacts)

    def print_manifest(
            self,
            statuses,
            discharged_patients,
            admitted_patients):
        """
        Print patient manifest after moving patients

        Input:
            statuses (dict): mapping node -> status
            discharged_patients (list): list of Patients
            admitted_patients (list): list of Patients

        Output:
            None
        """
        admitted_people = [p.address for p in admitted_patients]
        discharged_people_and_statuses = [(p.address, statuses[p.address]) for p in discharged_patients]
        current_patient_addresses = self.current_patient_addresses()

        print("[ Patient manifest ]          Admitted: ", end='')
        print(*admitted_people, sep=', ')
        print("                            Discharged: ", end='')
        print(*discharged_people_and_statuses, sep=', ')
        print("                               Current: ", end='')
        print(*current_patient_addresses, sep=', ')


class Patient:
    """
    Patient in hospital and their information.
    """
    def __init__(self, address, community_contacts, health_worker_contacts):
        """
        Args
        ----
        address (int): Location of the patient in the `original_contact_network`.

        community_contacts (list of tuples): List of edges in the `original_contact_network`.
                                             These are stored here while the patient is in hospital

        health_worker_contacts (list of tuples): a list of edges connecting patient to assigned health workers
        """
        self.address = address
        self.community_contacts = community_contacts
        self.health_worker_contacts = health_worker_contacts
