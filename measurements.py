################################################################################# 
# Copyright (c) 2021, the California Institute of Technology (Caltech).         #
# All rights reserved.                                                          #
#                                                                               #
# Redistribution and use in source and binary forms for academic and other      #
# non-commercial purposes with or without modification, are permitted           #
# provided that the following conditions are met. Commercial users contact      #
# innovation@caltech.edu for a license.                                         #
#                                                                               #
# * Redistributions of source code, including modified source code, must        # 
# retain the above copyright notice, this list of conditions and the            # 
# following disclaimer.                                                         #
#                                                                               #
# * Redistributions in binary form or a modified form of the source code        #
# must reproduce the above copyright notice, this list of conditions and        #
# the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                               #
#                                                                               #
# * Neither the name of Caltech, any of its trademarks, the names of its        #
# employees, nor contributors to the source code may be used to endorse or      #
# promote products derived from this software without specific prior            #
# written permission.                                                           #
#                                                                               #
# * Where a modified version of the source code is redistributed publicly       #
# in source or binary forms, the modified source code must be published in      #
# a freely accessible manner, or otherwise redistributed at no charge to        #
# anyone requesting a copy of the modified source code, subject to the same     #
# terms as this agreement.                                                      #
#                                                                               #
# THIS SOFTWARE IS PROVIDED BY CALTECH “AS IS” AND ANY EXPRESS OR IMPLIED       #
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO    #
# EVENT SHALL THE CONTRIBUTORS, CALTECH, ITS FACULTY, STUDENTS, EMPLOYEES, OR   # 
# TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,  #
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF       #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN       #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)       #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE    #
# POSSIBILITY OF SUCH DAMAGE.IMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR      #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, # 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE # 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
#################################################################################

import numpy as np
import copy


STATUS_CATALOG = dict(zip(['S','E','I', 'H', 'R', 'D'], np.arange(6)))


class TestMeasurement:
    def __init__(
            self,
            status,
            data_transform,            
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True):

        self.data_transform = data_transform
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.noisy_measurement=noisy_measurement
        self.status_catalog = STATUS_CATALOG

        self.status = status
        self.n_status = len(self.status_catalog.keys())

    def _set_prevalence(
            self,
            ensemble_states,
            fixed_prevalence=None):
        """
        Inputs:
        -------
            ensemble_states : `np.array` of shape (ensemble_size, num_status * user_population) at a given time
        """
        if fixed_prevalence is None:
            user_population      = int(ensemble_states.shape[1]/self.n_status)
            ensemble_size   = ensemble_states.shape[0]

            prevalence = ensemble_states.reshape(ensemble_size,self.n_status, user_population)[:,self.status_catalog[self.status],:].sum(axis = 1) / float(user_population)
            self.prevalence = np.clip(prevalence,1.0/float(user_population),None)

        else:
            self.prevalence = fixed_prevalence

    def _set_ppv(
            self):

        PPV = self.sensitivity * self.prevalence / \
             (self.sensitivity * self.prevalence + (1 - self.specificity) * (1 - self.prevalence))

        FOR = (1 - self.sensitivity) * self.prevalence / \
             ((1 - self.sensitivity) * self.prevalence + self.specificity * (1 - self.prevalence))

        transformed_ppv = self.data_transform.apply_transform(PPV)
        transformed_for = self.data_transform.apply_transform(FOR)
        #take mean in transformed space
        self.ppv_mean = transformed_ppv.mean()
        self.for_mean = transformed_for.mean()

    def update_prevalence(
            self,
            ensemble_states,
            fixed_prevalence=None):

        self._set_prevalence(ensemble_states, fixed_prevalence)
        self._set_ppv()

    def get_mean(
            self,
            positive_test=True):

        if positive_test:
            return self.ppv_mean
        else:
            return self.for_mean

    def take_measurements(
            self,
            nodes_state_dict):
        """
        Queries the diagnostics from a medical test with defined `self.sensitivity` and `self.specificity` properties in
        population with a certain prevelance (computed from an ensemble of master equations).

        Noisy measurement can be enabled which will report back, for example, in a true infected a negative result with measurement `FOR`.

        Inputs:
        -------

        """
        measurements = {}
        positive_nodes = []
        
        for node in nodes_state_dict.keys():
            if nodes_state_dict[node] == self.status:
                positive_test = not (self.noisy_measurement and (np.random.random() > self.sensitivity))
                measurements[node] = self.get_mean(positive_test = positive_test)
            else:
                positive_test = (self.noisy_measurement and (np.random.random() < 1 - self.specificity))
                measurements[node] = self.get_mean(positive_test = positive_test)
              
            if positive_test:
                positive_nodes.append(node)

        return measurements, positive_nodes

#### Adding Observations in here
class FixedNodeObservation:
    """
    Performs an observation at a fixed set of nodes.
    """
    def __init__(
            self,
            N,
            obs_nodes,
            obs_status):
        """
        Args
        ----
        N (int): user population size
        obs_nodes (list/np.array): collection of node id's from which we always observe
        obs_status (string): status to observe
        """
       
        #number of nodes in the graph
        self.N = N

        #statuses
        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())
        self.obs_status_idx = np.array([self.status_catalog[status] for status in obs_status])
        #fixed observation
        obs_states = [self.N*self.obs_status_idx+i for i in obs_nodes]
        if len(obs_states)>0:
            self.obs_states = np.hstack(obs_states)
        else:
            self.obs_states = np.array([],dtype=int)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        #Candidates for observations are those with a required state >= threshold
        return self.obs_states
            
class StateInformedObservation:
    def __init__(
            self,
            N,
            obs_frac,
            obs_status,
            min_threshold=0.0,
            max_threshold=1.0):
        """
        Args
        ----
        N (int): user population size
        obs_frac (float): the fraction of the states satisfying the threshold to observe
        obs_status (string): status to observe
        min_threshold (float): mean of ensemble >= minimum threshold for observation to occur
        max_threshold (float): mean of ensemble <= maximum threshold for observation to occur
        """
        #number of nodes in the graph
        self.N = N

        #statuses
        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())
        self.obs_status_idx = np.array([self.status_catalog[status] for status in obs_status])

        #The fraction of states
        self.obs_frac = np.clip(obs_frac,0.0,1.0)
        #The minimum threshold (in [0,1]) to be considered for test (e.g 0.7)
        self.obs_min_threshold = np.clip(min_threshold,0.0,1.0)
        self.obs_max_threshold = np.clip(max_threshold,0.0,1.0)

        #default init observation
        self.obs_states = np.empty(0)

    def set_obs_frac(self, obs_frac):
        self.obs_frac=np.clip(obs_frac,0.0,1.0)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        #Candidates for observations are those with a required state >= threshold
        candidate_states = np.hstack([self.N*self.obs_status_idx+i for i in range(self.N)])
        xmean = np.mean(state[:,candidate_states],axis=0)

        candidate_states_ens=candidate_states[(xmean>=self.obs_min_threshold) & \
                                              (xmean<=self.obs_max_threshold)]

        M=candidate_states_ens.size
        if (int(self.obs_frac*M)>=1) and (self.obs_frac < 1.0) :
            # If there is at least one state to sample (...)>=1.0
            # and if we don't sample every state
            choice=np.random.choice(np.arange(M), size=int(self.obs_frac*M), replace=False)
            self.obs_states=candidate_states_ens[choice]
        elif (self.obs_frac == 1.0):
            self.obs_states=candidate_states_ens
        else: #The value is too small
            self.obs_states=np.array([],dtype=int)
            print("no observation was above the threshold")

class BudgetedInformedObservation:
    """
    We observe the status at some nodes. We provide 
    - A status to observe at
    - A maximum budget of observations per node 
    - An interval [a,b] where we observe (first) if the ensemble mean state satisfies the constraint 
    E.g if we have a budget of 10 nodes in an assimilation, to observe status I, and [0.3,0.5] interval.
        Assume the ensemble mean gives 4 nodes where 0.3 <= I <= 0.5, we observe these.
        We then randomly observe outside 6 more nodes where I < 0.3 or I > 0.5.
    """
    def __init__(
            self,
            N,
            obs_budget,
            obs_status,
            min_threshold,
            max_threshold):

        """
        Args
        ----
        N (int): user population size
        obs_budget (int): the number of states to observe in every observation
        obs_status (string): status to observe
        min_threshold (float): mean of ensemble >= minimum threshold for observation preference
        max_threshold (float): mean of ensemble <= maximum threshold for observation preference
        """
       
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in

        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())


        #array of status to observe
        self.obs_status_idx = np.array([self.status_catalog[status] for status in obs_status])

        #The absolute number of nodes to observe
        self.obs_budget = int(obs_budget)
        #The minimum threshold (in [0,1]) to be considered for test (e.g 0.7)
        self.obs_min_threshold = np.clip(min_threshold,0.0,1.0)
        self.obs_max_threshold = np.clip(max_threshold,0.0,1.0)

        #default init observation
        self.obs_states = np.empty(0)

    def set_obs_budget(self, obs_budget):
        self.obs_budget=int(obs_budget)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        #Candidates for observations are those with a required state >= threshold
        candidate_states = np.hstack([self.N*self.obs_status_idx+i for i in range(self.N)])
        xmean = np.mean(state[:,candidate_states],axis=0)

        candidate_states_ens=candidate_states[(xmean>=self.obs_min_threshold) & \
                                              (xmean<=self.obs_max_threshold)]
        other_states_ens=candidate_states[(xmean<self.obs_min_threshold) | \
                                          (xmean>self.obs_max_threshold)]

        cand_size=candidate_states_ens.size
        other_size= other_states_ens.size
        print("number of states within the threshold", cand_size)
        if cand_size == self.obs_budget:
            self.obs_states = candidate_states_ens

        elif cand_size > self.obs_budget:
            choice=np.random.choice(np.arange(cand_size), size=self.obs_budget, replace=False)
            self.obs_states=candidate_states_ens[choice]

        else: #cand_size < self.obs_budget
            choice=np.random.choice(np.arange(other_size), size=self.obs_budget - cand_size, replace=False)
            self.obs_states = np.hstack([candidate_states_ens, other_states_ens[choice]])

        self.obs_states = np.unique(self.obs_states)


class StaticNeighborTransferObservation:
    """
    We observe the status at nodes randomly, until a positive measurement is taken. Then we preferentially sample the neighbors of the positive node. 
    The neighbours are taken from a static address book of the node (not a dynamically changing one)
    We provide
    - A status to observe at
    - A maximum budget of observations per node 
    E.g if we have a budget of 10 nodes in an assimilation, to observe status I.
        Time T1 : We randomly observe 10 nodes. Then test them.
        Assume we obtain a positive test at node x. We say the (e.g 6) neighbors of x.
        On the next observation we test the 6 neighbors of x, and 
        randomly observe 4 more nodes not on the list.
    """
    def __init__(
            self,
            N,
            obs_budget,
            obs_status,
            storage_type="temporary",
            nbhd_sampling_method="random"):

        """
        Args
        ----
        N (int): user population size
        obs_budget (int): the number of states to observe in every observation
        obs_status (string): status to observe
        storage_type (string): "temporary" we only store the nbhd nodes for 1 subsequent observation
                               "permanent" we store the nbhd nodes until they are observed
        nbhd_sampling_method (string): strategy to pick nodes when observing a neighborhood
                                      "random" - random choice of the size of the budget
                                      "mean" - order states by mean value of state and choose the largest
                                      "variance" - order states by variance and choose the largest
        """
       
        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in

        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())


        #array of status to observe
        self.obs_status_idx = np.array([self.status_catalog[status] for status in obs_status])

        #The absolute number of nodes to observe
        self.obs_budget = int(obs_budget)
        
        #default init observation
        self.obs_states = np.empty(0)

        #storage_lists
        self.nodes_to_observe = []
        self.positively_tested_nodes = []
        
        #storage_type: permanent, temporary
        self.storage_type = storage_type
        #nbhd_sampling_method to use: random, mean, variance
        self.nbhd_sampling_method = nbhd_sampling_method
        
        #conversion from node id to np array index
        self.node_to_idx = None
        self.idx_to_node = None

    def set_obs_budget(self, obs_budget):
        self.obs_budget=int(obs_budget)

    def add_positively_tested_nodes(self, nodes):
        """
        We store nodes which have provided a positive test
        The nodes are stored by node id, and not by np.array index.
        """
        if isinstance(nodes,np.ndarray):
            nodes = nodes.tolist()

        self.positively_tested_nodes.extend(nodes)
        print("history of positive nodes", self.positively_tested_nodes)
    def add_nbhds_to_observe(self, nodes):
        """
        We add nodes from a list, so long as they are not on the omission list
        stored as nodes and not by np.array index
        """
        admissible_nodes = [n for n in filter(lambda n: n not in self.positively_tested_nodes, nodes)]

        self.nodes_to_observe.extend(admissible_nodes)
        self.nodes_to_observe = list(set(self.nodes_to_observe)) #uniqueness
        
    def omit_nodes(self, nodes):
        """
        We remove nodes from the list that have been provided 
        """
        #Remove the inputed nodes
        nodes_to_observe = copy.deepcopy(self.nodes_to_observe)
        self.nodes_to_observe = [n for n in filter(lambda n: n not in nodes, nodes_to_observe)] 
                
    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        if self.node_to_idx is None:
            nodes = list(network.get_nodes())
            #store the map from index : node number and vice versa
            self.node_to_idx = {id_node[1] : id_node[0] for id_node in enumerate(nodes)} 
            self.idx_to_node = {id_node[0] : id_node[1] for id_node in enumerate(nodes)}

        candidate_states = np.hstack([self.N * self.obs_status_idx + i for i in range(self.N)])
        #look on the nodes_to_observe list.
        states_to_observe = [ self.node_to_idx[node] for node in self.nodes_to_observe]
        candidate_nbhd_states = candidate_states[states_to_observe]

        print("number of candidates", candidate_nbhd_states.size, "budget", self.obs_budget)
        #If we have more neighbors than budget
        if candidate_nbhd_states.size == self.obs_budget:
            self.obs_states = candidate_nbhd_states
            
        elif candidate_nbhd_states.size > self.obs_budget:
            if self.nbhd_sampling_method == "mean":
                # ordered choice - mean
                xmean=np.mean(state[:,candidate_nbhd_states], axis=0)
                dec_sort_vector = np.argsort(-xmean)
                choice = dec_sort_vector[:self.obs_budget]
                print("chosen_states", choice, "mean value", xmean[choice])
            elif self.nbhd_sampling_method == "variance":
                # ordered choice - variance
                xvar=np.var(state[:,candidate_nbhd_states], axis=0)
                dec_sort_vector = np.argsort(-xvar)
                choice = dec_sort_vector[:self.obs_budget]
                print("chosen_states", choice, "variance value", xvar[choice])
            elif self.nbhd_sampling_method == "random":
                #random choice
                choice = np.random.choice(np.arange(candidate_nbhd_states.size), size=self.obs_budget, replace=False)
            else:
                raise ValueError("unknown nbhd_sampling_method. choose from 'random' (default), 'mean', 'variance' ")

            self.obs_states = candidate_nbhd_states[choice]

        else: #candidate_nbhd_states.size < budget    
            other_idx = np.array([i for i in filter(lambda i: i not in states_to_observe, np.arange(candidate_states.size))])
            other_states = candidate_states[other_idx]
            choice=np.random.choice(np.arange(other_states.size), size=self.obs_budget - candidate_nbhd_states.size, replace=False)
            self.obs_states = np.hstack([candidate_nbhd_states, other_states[choice]])
            
        #perform omissions: (must convert from indices to node ids)
        obs_nodes= [ self.idx_to_node[idx] for idx in np.remainder(self.obs_states,self.N)]
        print("observed_nodes", obs_nodes)
        
        if self.storage_type == "permanent":
            self.omit_nodes(obs_nodes)
        elif self.storage_type == "temporary":
            self.omit_nodes(np.remainder(candidate_nbhd_states, self.N).tolist() )
        else:
            raise ValueError("unknown storage_type. Choose from 'permanent', 'temporary'(default)")

class HighVarianceStateInformedObservation:
    """
    We observe a given fraction of states which have the highest variance (in the ensemble).
    """
    def __init__(
            self,
            N,
            obs_frac,
            obs_status):

        """
        Args
        ----
        N (int): user population size
        obs_frac (float): the fraction of total states to observe
        obs_status (string): status to observe
        """

        #number of nodes in the graph
        self.N = N
        #number of different states a node can be in

        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())

        #array of status to observe
        self.obs_status_idx = np.array([self.status_catalog[status] for status in obs_status])

        #The fraction of states
        self.obs_frac = np.clip(obs_frac,0.0,1.0)

        #default init observation
        self.obs_states = np.empty(0)

    def set_obs_frac(self, obs_frac):
        self.obs_frac=np.clip(obs_frac,0.0,1.0)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        candidate_states = np.hstack([self.N*self.obs_status_idx+i for i in range(self.N)])
        obs_states_size=int(self.obs_frac*self.N)

        if (obs_states_size >= 1) and (self.obs_frac < 1.0) :
            #Candidates for observations are those with a required state >= threshold
            xvar = np.var(state[:,candidate_states],axis=0)
            dec_sort_vector = np.argsort(-xvar)

            self.obs_states=candidate_states[dec_sort_vector[:obs_states_size]]
            
        elif (self.obs_frac == 1.0):
            self.obs_states=candidate_states
        else: #The value is too small
            self.obs_states=np.array([],dtype=int)
            print("no observation - increase obs_frac")

#combine them together
class Observation(StateInformedObservation, TestMeasurement):
    def __init__(
            self,
            N,
            obs_frac,
            obs_status,
            obs_name,
            min_threshold=0.0,
            max_threshold=1.0,
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True,
            obs_var_min = 1e-3):

        self.name=obs_name
        self.obs_var_min = obs_var_min

        StateInformedObservation.__init__(self,
                                          N,
                                          obs_frac,
                                          obs_status,
                                          min_threshold,
                                          max_threshold)
        TestMeasurement.__init__(self,
                                 obs_status,
                                 sensitivity,
                                 specificity,
                                 noisy_measurement)

            
    def set_obs_frac(self, obs_frac):
        """
        Set the observation fraction/budget
        """
        StateInformedObservation.set_obs_frac(self, obs_frac)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,

        Inputs:
            state: np.array of size [self.N * n_status]
        """
        StateInformedObservation.find_observation_states(self,
                                                         network,
                                                         state,
                                                         data)
    

    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #make a measurement of the data
        TestMeasurement.update_prevalence(self,
                                          state)

        nodes=network.get_nodes()
        #mean, var np.arrays of size state
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = nodes[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}
        
        mean, positive_nodes = TestMeasurement.take_measurements(self,
                                                      observed_data)

        observed_mean     = np.array([mean[node] for node in observed_nodes])
        observed_variance = np.array([self.obs_var_min for node in observed_nodes])
        
        self.mean     = observed_mean
        self.variance = observed_variance
        self.positive_nodes = positive_nodes
        


class BudgetedObservation(BudgetedInformedObservation, TestMeasurement):
    def __init__(
            self,
            N,
            obs_budget,
            obs_status,
            data_transform, #see transforms.jl
            obs_name,
            min_threshold=0.0,
            max_threshold=1.0,
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True,
            obs_var_min = 1e-3,
            true_prevalence = False):

        self.name=obs_name
        self.obs_var_min = obs_var_min
        self.true_prevalence = true_prevalence

        BudgetedInformedObservation.__init__(self,
                                          N,
                                          obs_budget,
                                          obs_status,
                                          min_threshold,
                                          max_threshold)
        TestMeasurement.__init__(self,
                                 obs_status,
                                 data_transform,
                                 sensitivity,
                                 specificity,
                                 noisy_measurement)

    def set_obs_budget(self, obs_budget):
        """
        Set the observation budget
        """
        BudgetedInformedObservation.set_obs_budget(self, obs_budget)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,

        Inputs:
            state: np.array of size [self.N * n_status]
        """
        BudgetedInformedObservation.find_observation_states(self,
                                                         network,
                                                         state,
                                                         data)
    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #make a measurement of the data
        if self.true_prevalence == False:
            TestMeasurement.update_prevalence(self,
                                              state)
        else:
            data_prevalence = np.sum([v=='I' for v in data.values()])/len(data) * \
                              np.ones(state.shape[0])
            TestMeasurement.update_prevalence(self,
                                              state,
                                              fixed_prevalence=data_prevalence)

        nodes=network.get_nodes()
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = nodes[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}

        mean, positive_nodes = TestMeasurement.take_measurements(self,
                                                                 observed_data)

        
        observed_mean     = np.array([mean[node] for node in observed_nodes])
        observed_variance = np.array([self.obs_var_min for node in observed_nodes])

        self.mean     = observed_mean
        self.variance = observed_variance
        self.positive_nodes = positive_nodes
                
class FixedObservation(FixedNodeObservation, TestMeasurement):
    def __init__(
            self,
            N,
            obs_nodes,
            obs_status,
            data_transform,
            obs_name,
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True,
            obs_var_min = 1e-3):

        self.name=obs_name
        self.obs_var_min = obs_var_min

        FixedNodeObservation.__init__(self,
                                      N,
                                      obs_nodes,
                                      obs_status)
        TestMeasurement.__init__(self,
                                 obs_status,
                                 data_transform,
                                 sensitivity,
                                 specificity,
                                 noisy_measurement)
    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,

        Inputs:
            state: np.array of size [self.N * n_status]
        """
        FixedNodeObservation.find_observation_states(self,
                                                     network,
                                                     state,
                                                     data)

    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #make a measurement of the data
        TestMeasurement.update_prevalence(self,
                                          state)

        nodes=network.get_nodes()
        #mean, var np.arrays of size state
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = nodes[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}

        mean, positive_nodes = TestMeasurement.take_measurements(self,
                                                 observed_data)

        observed_mean     = np.array([mean[node] for node in observed_nodes])
        #prescribed variance
        observed_variance = np.array([self.obs_var_min for node in observed_nodes])

        self.mean     = observed_mean
        self.variance = observed_variance
        self.positive_nodes = positive_nodes


class StaticNeighborObservation( StaticNeighborTransferObservation, TestMeasurement):
    def __init__(
            self,
            N,
            obs_budget,
            obs_status,
            obs_name,
            storage_type="temporary",
            nbhd_sampling_method="random",
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True,
            obs_var_min = 1e-3):

        self.name=obs_name
        self.obs_var_min = obs_var_min

        StaticNeighborTransferObservation.__init__(self,
                                                   N,
                                                   obs_budget,
                                                   obs_status,
                                                   storage_type,
                                                   nbhd_sampling_method)

        TestMeasurement.__init__(self,
                                 obs_status,
                                 sensitivity,
                                 specificity,
                                 noisy_measurement)

    def set_obs_budget(self, obs_budget):
        """
        Set the observation budget
        """
        StaticNeighborTransferObservation.set_obs_budget(self, obs_budget)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,

        Inputs:
            state: np.array of size [ensemble_size, self.N * n_status]
        """
        StaticNeighborTransferObservation.find_observation_states(self,
                                                                  network,
                                                                  state,
                                                                  data)

    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #make a measurement of the data
        TestMeasurement.update_prevalence(self,
                                          state)

        nodes=network.get_nodes()
        user_graph = network.get_graph()
       
        #mean, var np.arrays of size state
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = nodes[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}

        true_infected = [node for node in nodes if data[node] == 'I']
        print("actually infected nodes", true_infected)
        #for ti in true_infected:
        #    print("neighborhood of ", ti, ": ", list(user_graph.neighbors(ti)))
        
        mean, positive_nodes = TestMeasurement.take_measurements(self, observed_data)

        observed_mean     = np.array([mean[node] for node in observed_nodes])
        observed_variance = np.array([self.obs_var_min for node in observed_nodes])

        self.mean     = observed_mean
        self.variance = observed_variance
        
        #Now to add nodes to the neighbors list
        print("Nodes testing positive", positive_nodes)
        positive_nodes_nbhd = [] 
        for pn in positive_nodes:
            positive_nodes_nbhd.extend(user_graph.neighbors(pn))
       
        #omit the positive nodes from testing
        StaticNeighborTransferObservation.add_positively_tested_nodes(self, positive_nodes)
        StaticNeighborTransferObservation.add_nbhds_to_observe(self, positive_nodes_nbhd)
        

#combine them together
class HighVarianceObservation(HighVarianceStateInformedObservation, TestMeasurement):
    def __init__(
            self,
            N,
            obs_frac,
            obs_status,
            obs_name,
            sensitivity=0.80,
            specificity=0.99,
            noisy_measurement=True,
            obs_var_min = 1e-3):

        self.name=obs_name
        self.obs_var_min = obs_var_min

        HighVarianceStateInformedObservation.__init__(self,
                                                      N,
                                                      obs_frac,
                                                      obs_status)
        TestMeasurement.__init__(self,
                                 obs_status,
                                 sensitivity,
                                 specificity,
                                 noisy_measurement)

    def set_obs_frac(self, obs_frac):
        """
        Set the observation fraction/budget
        """
        HighVarianceStateInformedObservation.set_obs_frac(self, obs_frac)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,

        Inputs:
            state: np.array of size [self.N * n_status]
        """
        HighVarianceStateInformedObservation.find_observation_states(self,
                                                                     network,
                                                                     state,
                                                                     data)

    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #make a measurement of the data
        TestMeasurement.update_prevalence(self,
                                          state)

        nodes=network.get_nodes()
        #mean, var np.arrays of size state
        observed_states = np.remainder(self.obs_states,self.N)
        #convert from np.array indexing to the node id in the (sub)graph
        observed_nodes = nodes[observed_states]
        observed_data = {node : data[node] for node in observed_nodes}

        mean, positive_nodes = TestMeasurement.take_measurements(self,
                                                      observed_data)

        observed_mean     = np.array([mean[node] for node in observed_nodes])
        observed_variance = np.array([self.obs_var_min for node in observed_nodes])
        
        self.mean     = observed_mean
        self.variance = observed_variance
        self.positive_nodes = positive_nodes


class DataInformedObservation:
    def __init__(
            self,
            N,
            bool_type,
            obs_status):

        #number of nodes in the graph
        self.N = N
        #if you want to find the where the status is, or where it is not.
        self.bool_type = bool_type
        self.status_catalog = STATUS_CATALOG
        self.n_status = len(self.status_catalog.keys())
        self.obs_status=obs_status
        self.obs_status_idx=np.array([self.status_catalog[status] for status in obs_status])
        self.obs_states=np.empty(0)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Update the observation model when taking observation
        """
        # Obtain relevant data entries
        user_nodes = network.get_nodes()
        user_data = {node : data[node] for node in user_nodes}

        candidate_nodes = []
        for status in self.obs_status:
            candidate_nodes.extend([node for node in user_data.keys() if (user_data[node] == status) == self.bool_type])
            
        # we now have the node numbers for the statuses we want to measure,
        # but we require an np index for them
        candidate_states_modulo_population = np.array([state for state in range(len(user_nodes))
                                                       if user_nodes[state] in candidate_nodes])

        #now add the required shift to obtain the correct status 'I' or 'H' etc.
        candidate_states = [candidate_states_modulo_population + i*self.N for i in self.obs_status_idx]

        self.obs_states=np.hstack(candidate_states)
        

class DataObservation(DataInformedObservation):
    def __init__(
            self,
            N,
            set_to_one,
            obs_var,
            obs_status,
            data_transform,
            obs_name):
        """
        An observation which uses the current statuses of the epidemic model to influence the risk simulation.
        It doesn't make a measurement, just fixes the value to near 1, or near 0.

        Args
        ----
        N (int)               : number of nodes
        set_to_one (bool)     : set_to_one=True  means we set "state = 1" when "status == obs_status"
                                set_to_one=False means we set "state = 0" when "status != obs_status"
        obs_var (float)       : size of variance of the observation
        obs_status (string)   : character of the status we assimilate
        obs_name (string)     : name of observation
        """
        self.N=N
        self.set_to_one = set_to_one
        self.var_tol = max(obs_var,1e-16)
        self.data_transform = data_transform
        self.name=obs_name

        DataInformedObservation.__init__(self,
                                         N,
                                         set_to_one,
                                         obs_status)

    def find_observation_states(
            self,
            network,
            state,
            data):
        """
        Obtain where one should make an observation based on the current state,
        and the contact network

        Inputs:
            state: np.array of size [self.N * n_status]
        """
        DataInformedObservation.find_observation_states(self,
                                                        network,
                                                        state,
                                                        data)

    def observe(
            self,
            network,
            state,
            data):
        """
        Inputs:
            data: dictionary {node number : status}; data[i] = contact_network.node(i)
        """

        #tolerance,as we cannot set values "equal" to 0 or 1
        # Note: this has to be very small if one assimilates the values for many nodes)
        #       always check the variances in the logit transformed variables.

        # set_to_one=True  means we set "state = 1" when "status == obs_status"
        
        MEAN_TOLERANCE     = 0.0 #0.1/self.N # 1e-9
        VARIANCE_TOLERANCE = self.var_tol  #1e-14

        if self.set_to_one:
            nodes=network.get_nodes()
            #mean, var np.arrays of size state
            observed_states = np.remainder(self.obs_states,self.N)
            #convert from np.array indexing to the node id in the (sub)graph
            observed_nodes = nodes[observed_states]
            positive_nodes = observed_nodes
            

            observed_mean = (1-MEAN_TOLERANCE) * np.ones(self.obs_states.size)
            observed_variance = VARIANCE_TOLERANCE * np.ones(self.obs_states.size)
            transformed_mean = self.data_transform.apply_transform(observed_mean)
            transformed_variance = self.data_transform.apply_transform(observed_variance)
        # set_to_one=False means we set "state = 0" when "status != obs_status"
        else:
            positive_nodes = []
            observed_mean = MEAN_TOLERANCE * np.ones(self.obs_states.size)
            observed_variance = VARIANCE_TOLERANCE * np.ones(self.obs_states.size)
            transformed_mean = self.data_transform.apply_transform(observed_mean)
            transformed_variance = self.data_transform.apply_transform(observed_variance)

         
        self.mean     = transformed_mean
        self.variance = transformed_variance
        self.positive_nodes = positive_nodes

