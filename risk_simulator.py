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

import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from multiprocessing import shared_memory
from timeit import default_timer as timer

import ray
from numba import jit 

from .contact_simulator import diurnal_inception_rate

@jit(nopython=True)
def create_CM_data(nonzeros, rows, cols, data, M, yS, yI, yH, Smean, Imean, Hmean, ensemble_correction=True):
    CM_SI_data = np.zeros(nonzeros)
    CM_SH_data = np.zeros(nonzeros)
    CM_SI_coeff = np.zeros(nonzeros)
    CM_SH_coeff = np.zeros(nonzeros)
    for k,i,j,v in zip(range(nonzeros), rows, cols, data):
        #for SI interactions
        if ensemble_correction==True:
            CM_SI_coeff[k] = yS[:,i].dot(yI[:,j])/M  #create bar{<S_i,I_j>}
            CM_SI_coeff[k] /= Smean[i]*Imean[j]+1e-8 #divide by <S_i><I_j>+eps
        else:
            CM_SI_coeff[k] = 1.
        CM_SI_data[k] = CM_SI_coeff[k] * v       #multiply by the weight matrix wij

        #for SH interactions
        if ensemble_correction==True:
            CM_SH_coeff[k] = yS[:,i].dot(yH[:,j])/M
            CM_SH_coeff[k] /= Smean[i]*Hmean[j]+1e-8
        else:
            CM_SH_coeff[k] = 1.
        CM_SH_data[k] = CM_SH_coeff[k] * v

    return CM_SI_data, CM_SH_data, CM_SI_coeff, CM_SH_coeff

@jit(nopython=True)
def create_CM_data_ptr(nonzeros, rows, cols, data, M, yS, yI, yH, Smean, Imean, Hmean, ptr, ensemble_correction=True):
    CM_SI_data = np.zeros(nonzeros)
    CM_SH_data = np.zeros(nonzeros)
    CM_SI_coeff = np.zeros(nonzeros)
    CM_SH_coeff = np.zeros(nonzeros)
    for k,i,j,v in zip(range(nonzeros), rows, cols, data):
        #for SI interactions
        if ensemble_correction==True:
            CM_SI_coeff[k] = yS[:,i].dot(yI[:,j])/M  #create bar{<S_i,I_j>}
            CM_SI_coeff[k] /= Smean[i]*Imean[j]+1e-8 #divide by <S_i><I_j>+eps
        else:
            CM_SI_coeff[k] = 1.
        CM_SI_data[k] = CM_SI_coeff[k] * v       #multiply by the weight matrix wij
        CM_SI_data[k] *= 0.5*(ptr[:,i] + ptr[:,j]).mean()

        #for SH interactions
        if ensemble_correction==True:
            CM_SH_coeff[k] = yS[:,i].dot(yH[:,j])/M
            CM_SH_coeff[k] /= Smean[i]*Hmean[j]+1e-8
        else:
            CM_SH_coeff[k] = 1.
        CM_SH_data[k] = CM_SH_coeff[k] * v
        CM_SH_data[k] *= 0.5*(ptr[:,i] + ptr[:,j]).mean()

    return CM_SI_data, CM_SH_data, CM_SI_coeff, CM_SH_coeff

class MasterEquationModelEnsemble:
    def __init__(
            self,
            population,
            transition_rates,
            transmission_rate_parameters,
            hospital_transmission_reduction=0.25,
            ensemble_size=1,
            exterior_neighbors=None,
            start_time=0.0,
            parallel_cpu=False,
            num_cpus=1,
            ensemble_correction=True):
        """
        Constructor

        Input:
            population (int): population count
            transition_rates (TransitionRates): same rates for each ensemble
                                                member
                             (list of TransitionRates): list of length
                                                        ensemble_size with
                                                        individual rates for
                                                        each member
            transmission_rate_parameters (np.array): (M, N) array of individual, partial transmission rates (ptr) for each member. 
                                                   The transmission rate from node i to node j is calculated to be the 
                                                   (ptr(:,i) + ptr(:,j)) / 2 * contact_rate(i,j)
                                   
            hospital_transmission_reduction (float): fraction of beta in
                                                     hospitals
            ensemble_size (int): number of ensemble members
            exterior_neighbors (numpy): gives weights > 0 with which to weight user network nodes (based on connectivity to the network and general population graph)
            start_time (float): start time of the simulation
            parallel_cpu (bool): whether to run computation in parallel on CPU
            num_cpus (int): number of CPUs available; only used in parallel mode
        """

        self.M = ensemble_size
        self.N = population
        
        if exterior_neighbors is None:
            self.exterior_neighbors = None
            self.exogenous_flag = False
        else:
            self.exterior_neighbors = exterior_neighbors
            self.exogenous_flag = True
        
        self.start_time = start_time

        self.S_slice = slice(       0,   self.N)
        self.E_slice = slice(  self.N, 2*self.N)
        self.I_slice = slice(2*self.N, 3*self.N)
        self.H_slice = slice(3*self.N, 4*self.N)
        self.R_slice = slice(4*self.N, 5*self.N)
        self.D_slice = slice(5*self.N, 6*self.N)

        self.hospital_transmission_reduction = hospital_transmission_reduction

        self.CM_SI_coeff_history = []
        self.CM_SH_coeff_history = []

        self.ensemble_correction = ensemble_correction

        self.parallel_cpu = parallel_cpu
        if parallel_cpu:
            # need to save SharedMemory objects b/c otherwise they're
            # garbage-collected at the end of constructor

            float_nbytes = np.dtype(np.float_).itemsize

            y0_nbytes = self.M * self.N * 6 * float_nbytes
            self.y0_shm = shared_memory.SharedMemory(create=True,
                                                     size=y0_nbytes)
            self.y0 = np.ndarray((self.M, self.N * 6),
                                 dtype=np.float_,
                                 buffer=self.y0_shm.buf)

            closure_nbytes = self.M * self.N * float_nbytes
            self.closure_shm = shared_memory.SharedMemory(create=True,
                                                          size=closure_nbytes)
            self.closure = np.ndarray((self.M, self.N),
                                      dtype=np.float_,
                                      buffer=self.closure_shm.buf)

            coefficients_nbytes = self.M * self.N * 8 * float_nbytes
            self.coefficients_shm = shared_memory.SharedMemory(
                    create=True,
                    size=coefficients_nbytes)
            self.coefficients = np.ndarray((self.M, 8*self.N),
                                           dtype=np.float_,
                                           buffer=self.coefficients_shm.buf)

            exog_nbytes = self.N * 8 * float_nbytes
            self.exog_shm = shared_memory.SharedMemory(
                create=True,
                size=exog_nbytes)
            self.prevalence_indep_exogenous_rates = np.ndarray(self.N,
                                                               dtype=np.float_,
                                                               buffer=self.exog_shm.buf)

            members_chunks = np.array_split(np.arange(self.M), num_cpus)

            self.integrators = [
                    RemoteIntegrator.remote(members_chunks[j],
                                            self.M,
                                            self.N,
                                            self.y0_shm.name,
                                            self.coefficients_shm.name,
                                            self.closure_shm.name,
                                            self.exog_shm.name)
                    for j in range(num_cpus)
            ]
        else:
            self.y0           = np.empty( (self.M, 6*self.N) )
            self.closure      = np.empty( (self.M,   self.N) )
            self.coefficients = np.empty( (self.M, 8*self.N) )
            self.prevalence_indep_exogenous_rates = np.empty( (self.N) )

        self.update_transition_rates(transition_rates)


        #for transmission or partial transmission rate
        if isinstance(transmission_rate_parameters,list):
            transmission_rate_parameters = np.array(transmission_rate_parameters)
            
        # we have M-sized transmission_rates, then the rates are the real rates
        self.ensemble_beta_infected = np.empty( (self.M,1) )
        self.ensemble_beta_hospital = np.empty( (self.M,1) )
  
        if transmission_rate_parameters.size == self.M: #if it is the size of the ensemble only the 
            self.full_transmission_rate_flag=True
      
        elif transmission_rate_parameters.shape == (self.M,self.N): 
            self.full_transmission_rate_flag=False
        else:
            raise ValueError("incorrect size of transmission_rate_parameter, must be either size M or (M,N)" )

        self.update_transmission_rate_parameters(transmission_rate_parameters)

        self.walltime_eval_closure = 0.0

    def __extract_coefficients(
            self,
            transition_rates):
        """
        Extract coefficients from TransitionRates object into np.array

        Input:
            transition_rates (TransitionRates): transition rates of a member
        Output:
            coefficients (np.array): (8*N,) array of master equations coeffs
        """
        sigma= transition_rates.get_transition_rate('exposed_to_infected')
        delta= transition_rates.get_transition_rate('infected_to_hospitalized')
        xi   = transition_rates.get_transition_rate('infected_to_resistant')
        xip  = transition_rates.get_transition_rate('hospitalized_to_resistant')
        mu   = transition_rates.get_transition_rate('infected_to_deceased')
        mup  = transition_rates.get_transition_rate('hospitalized_to_deceased')

        gamma  = xi  + mu  + delta
        gammap = xip + mup

        return np.hstack((
                sigma,
                gamma,
                delta,
                xi,
                mu,
                gammap,
                xip,
                mup
        ))

    def set_start_time(
            self,
            start_time):
        self.start_time = start_time

    def set_mean_contact_duration(
            self,
            mean_contact_duration):
        """
        Set mean contact duration a.k.a. L matrix

        Input:
            mean_contact_duration (scipy.sparse.csr.csr_matrix):
                adjacency matrix
        Output:
            None
        """
        self.L = mean_contact_duration

    def set_diurnally_averaged_nodal_activation_rate(
            self,
            diurnally_averaged_nodal_activation_rate):
        """
        Set diurnally averaged nodal activation rates (saved to the network as integrated_lambda)
        Used to approximate the edge contacts for edges extenral to a user_network

        Input:
            diurnally_averaged_nodal_activation_rate (np.array): array of rates

        Output:
            None
        """
        self.diurnally_averaged_nodal_activation_rate = diurnally_averaged_nodal_activation_rate


    def update_transmission_rate_parameters(
            self,
            transmission_rate_parameters):
        """
        Set transmission rates a.k.a. betas

        Input:
            transmission_rate_parameters (np.array):
                             1. (np.array): (M, 1) array of rates
                                    (list): list of rates of length M
                             2. (np.array): (M, N) array of rates
        Output:
            None
        """

        if self.full_transmission_rate_flag: #global transmission rate
            if isinstance(transmission_rate_parameters, list):
                self.ensemble_beta_infected[:,0] = np.fromiter(transmission_rate_parameters,
                                                               dtype=np.float_)
            else:
                self.ensemble_beta_infected[:] = transmission_rate_parameters

            self.ensemble_beta_hospital[:] = (
                self.hospital_transmission_reduction *
                self.ensemble_beta_infected
            )

        else: # partial transmission rate per node
            assert(transmission_rate_parameters.shape == (self.M,self.N) )        
            #self.partial_transition_rates = partial_transition_rates
            self.partial_transmission_rates = transmission_rate_parameters

            self.ensemble_beta_infected[:] = np.ones([self.M,1])
            self.ensemble_beta_hospital[:] = np.ones([self.M,1]) * self.hospital_transmission_reduction



    def update_transition_rates(
            self,
            transition_rates):
        """
        Set transition rates a.k.a. sigma, gamma etc.

        Input:
            transition_rates (TransitionRates): same rates for each ensemble
                                                member
                             (list of TransitionRates): list of length
                                                        ensemble_size with
                                                        individual rates for
                                                        each member
        Output:
            None
        """
        if isinstance(transition_rates, list):
            for j in range(self.M):
                self.coefficients[j] = self.__extract_coefficients(
                        transition_rates[j])
        else:
            # XXX obviously, there's memory overhead here; for simplicity's sake
            coefficients = self.__extract_coefficients(transition_rates)
            for j in range(self.M):
                self.coefficients[j] = coefficients

    def update_ensemble(
            self,
            new_transition_rates,
            new_transmission_rate_parameters):
        """
        Set all parameters of the ensemble (transition and transmission rates)

        Input:
            new_transition_rates (TransitionRates): same rates for each ensemble
                                                    member
                                 (list of TransitionRates): list of length
                                                            ensemble_size with
                                                            individual rates for
                                                            each member
            new_transmission_rate_parameters 
                                 1. (np.array): (M, 1) array of rates
                                        (list): list of rates of length M
                                 2. (np.array): (M, N) array of rates
        Output:
            None
        """
        self.update_transition_rates(new_transition_rates)
        self.update_transmission_rate_parameters(new_transmission_rate_parameters)

    def set_states_ensemble(
            self,
            states_ensemble):
        self.y0[:] = states_ensemble[:]

    def compute_rhs(
            self,
            j,
            member_state):
        """
        Compute right-hand side of master equations

        Input:
            j (int): index of the ensemble member
            member_state (np.array): (6*N,) array of states
        Output:
            rhs (np.array): (6*N,) right-hand side of master equations
        """
        S_substate = member_state[self.S_slice]
        E_substate = member_state[self.E_slice]
        I_substate = member_state[self.I_slice]
        H_substate = member_state[self.H_slice]
        
        (sigma,
         gamma,
         delta,
         xi,
         mu,
         gammap,
         xip,
         mup
        ) = self.coefficients[j].reshape( (8, -1) )

        rhs = np.empty(6 * self.N)

        
        if not self.exogenous_flag:
            rhs[self.S_slice] = -self.closure[j] * S_substate 
            rhs[self.E_slice] =  self.closure[j] * S_substate - sigma * E_substate
        else:
            prevalence = I_substate.mean(axis = 1)

            rhs[self.S_slice] = -(self.closure[j] + prevalence * self.prevalence_indep_exogenous_rates) * S_substate
            rhs[self.E_slice] =  (self.closure[j] + prevalence * self.prevalence_indep_exogenous_rates) * S_substate - sigma * E_substate
        
        rhs[self.I_slice] = sigma * E_substate - gamma  * I_substate
        rhs[self.H_slice] = delta * I_substate - gammap * H_substate
        rhs[self.R_slice] = xi    * I_substate + xip    * H_substate
        rhs[self.D_slice] = mu    * I_substate + mup    * H_substate

        return rhs

    def compute_prevalence_indep_exogenous_rates(self):
        """
        Compute the exogenous rate as a product of terms, except for the prevalence term. 
        This will be multiplied during the calculation based on current infectious rate

        If no rate is required the we set this to a None
        """
        #We make the approximation of taking ensemble means for the transmission rates
        if self.exogenous_flag:
            if self.full_transmission_rate_flag:
                self.prevalence_indep_exogenous_rates[:] = (self.exterior_neighbors *  
                                                            self.diurnally_averaged_nodal_activation_rate * 
                                                            self.ensemble_beta_infected.mean(axis=0)) 
            else:
                self.prevalence_indep_exogenous_rates[:] = (self.exterior_neighbors *  
                                                            self.diurnally_averaged_nodal_activation_rate * 
                                                            self.partial_transmission_rates.mean(axis=0)) 
                      
    def eval_closure(
            self,
            closure_name):
        """
        Evaluate closure from full ensemble state 'self.y0'

        Input:
            closure_name (str): which closure to evaluate; only 'independent'
                                and 'full' are supported at this time
        Output:
            None
        """
        if closure_name == 'independent':
            iS, iI, iH = self.S_slice, self.I_slice, self.H_slice
            y = self.y0

            S_ensemble_mean = y[:,iS].mean(axis=0)
            I_ensemble_mean = y[:,iI].mean(axis=0)
            H_ensemble_mean = y[:,iH].mean(axis=0)
            yS = y[:,iS]
            yI = y[:,iI]
            yH = y[:,iH]

            cooL = self.L.tocoo()
            nonzeros = len(cooL.row)
            cooL_rows = cooL.row
            cooL_cols = cooL.col
            cooL_data = cooL.data
            
            # if we have a global transmission parameter, this is accounted for in self.ensemble_beta_* 
            # if we have a nodally defined transmission parameter, the partial transmissions are 
            # accounted for in the closure calculation
            if self.full_transmission_rate_flag:
                CM_SI_data,CM_SH_data, CM_SI_coeff, CM_SH_coeff = create_CM_data(
                    nonzeros,
                    cooL_rows,
                    cooL_cols,
                    cooL_data,
                    self.M,
                    yS,
                    yI,
                    yH,
                    S_ensemble_mean,
                    I_ensemble_mean,
                    H_ensemble_mean,
                    ensemble_correction=self.ensemble_correction)
            else:
                CM_SI_data,CM_SH_data, CM_SI_coeff, CM_SH_coeff = create_CM_data_ptr(
                    nonzeros,
                    cooL_rows,
                    cooL_cols,
                    cooL_data,
                    self.M,
                    yS,
                    yI,
                    yH,
                    S_ensemble_mean,
                    I_ensemble_mean,
                    H_ensemble_mean,
                    self.partial_transmission_rates,
                    ensemble_correction=self.ensemble_correction)

            self.CM_SI_coeff_history.append(CM_SI_coeff)
            self.CM_SH_coeff_history.append(CM_SH_coeff)

            CM_S = coo_matrix((np.array(CM_SI_data),(cooL.row,cooL.col)),shape=(self.N,self.N)).tocsr()
            CM_S @= y[:,iI].T
            self.closure[:] =  CM_S.T * self.ensemble_beta_infected

            CM_S = coo_matrix((np.array(CM_SH_data),(cooL.row,cooL.col)),shape=(self.N,self.N)).tocsr()
            CM_S @= y[:,iH].T
            self.closure[:] +=  CM_S.T * self.ensemble_beta_hospital

        elif closure_name == 'full':
            # XXX this only works for betas of shape (M, 1); untested for others
            ensemble_I_substate = self.y0[:, self.I_slice] # (M, N)
            ensemble_H_substate = self.y0[:, self.H_slice] # (M, N)

            closure_I = ensemble_I_substate @ self.L
            closure_H = ensemble_H_substate @ self.L
            self.closure[:] = (  closure_I * self.ensemble_beta_infected,
                               + closure_H * self.ensemble_beta_hospital)

        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this value of 'closure_name' is not supported: "
                    + closure_name)

    def simulate(
            self,
            time_window,
            min_steps=1,
            closure_name='independent',
            closure_flag=True):
        """
        Simulate master equations for the whole ensemble forward in time

        Input:
            time_window (float): duration of simulation
            min_steps (int): minimum number of timesteps
            closure_name (str): which closure to use; only 'independent' and
                                'full' are supported at this time
            closure_flag (bool): whether to evaluate closure during this
                                 simulation call
        Output:
            y0 (np.array): (M, 6*N) array of states at the end of time_window
        """
        stop_time = self.start_time + time_window
        maxdt = abs(time_window) / min_steps

        if closure_flag:
            timer_eval_closure = timer()
            self.eval_closure(closure_name)
            self.walltime_eval_closure += timer() - timer_eval_closure

        
        self.compute_prevalence_indep_exogenous_rates()

        if self.parallel_cpu:
            futures = []
            args = (self.start_time, stop_time, maxdt, self.exogenous_flag)

            for integrator in self.integrators:
                futures.append(integrator.integrate.remote(*args))

            for future in futures:
                ray.get(future)
        else:
            for j in range(self.M):
                ode_result = solve_ivp(
                        fun = lambda t, y: (
                            self.compute_rhs(j, y)
                            ),
                        t_span = [self.start_time, stop_time],
                        y0 = self.y0[j],
                        t_eval = [stop_time],
                        method = 'RK45',
                        max_step = maxdt)

                self.y0[j] = np.clip(np.squeeze(ode_result.y), 0, 1)

        self.start_time += time_window
        return self.y0

    def simulate_backwards(
            self,
            time_window,
            min_steps=1,
            closure_name='independent',
            closure_flag=True):

        """
        Simulate master equations for the whole ensemble backward in time

        Input:
            time_window (float): duration of simulation
            min_steps (int): minimum number of timesteps
            closure_name (str): which closure to use; only 'independent' and
                                'full' are supported at this time
            closure_flag (bool): whether to evaluate closure during this
                                 simulation call
        Output:
            y0 (np.array): (M, 6*N) array of states at the end of time_window
        """
        positive_time_window = abs(time_window)
        
        return self.simulate(-positive_time_window,
                             min_steps,
                             closure_name,
                             closure_flag)

    def reset_walltimes(self):
        """
        Reset walltimes to zero
        """
        self.walltime_eval_closure = 0.0

    def get_walltime_eval_closure(self):
        """
        Get walltime of the 'eval_closure' calls
        """
        return self.walltime_eval_closure

    def wrap_up(self):
        """
        Take care of the shared memory objects

        This should be called only once at the end of object's lifetime;
        unfortunately, destructor doesn't work for this (some memory is already
        being freed, so it throws an exception).

        Output:
            None
        """
        if self.parallel_cpu:
            self.y0_shm.close()
            self.closure_shm.close()
            self.coefficients_shm.close()
            self.exog_shm.close()
    
            self.y0_shm.unlink()
            self.closure_shm.unlink()
            self.coefficients_shm.unlink()
            self.exog_shm.unlink()

@ray.remote
class RemoteIntegrator:
    def __init__(
            self,
            members_to_compute,
            M,
            N,
            ensemble_state_shared_memory_name,
            coefficients_shared_memory_name,
            closure_shared_memory_name,
            exog_shared_memory_name):
        """
        Constructor

        Input:
            members_to_compute (np.array): indices of members that are assigned
                                           to this integrator
            M (int): total number of ensemble members
            N (int): total number of nodes (population)
            ensemble_state_shared_memory_name (str): shared memory name (state)
            coefficients_shared_memory_name (str): shared memory name (coeffs)
            closure_shared_memory_name (str): shared memory name (closure)
        """
        self.M = M
        self.N = N
        
        self.shared_memory_names = {
            'ensemble_state'                   : ensemble_state_shared_memory_name,
            'coefficients'                     : coefficients_shared_memory_name,
            'closure'                          : closure_shared_memory_name,
            'prevalence_indep_exogenous_rates' : exog_shared_memory_name
        }

        self.S_slice = slice(  0,   N)
        self.E_slice = slice(  N, 2*N)
        self.I_slice = slice(2*N, 3*N)
        self.H_slice = slice(3*N, 4*N)
        self.R_slice = slice(4*N, 5*N)
        self.D_slice = slice(5*N, 6*N)

        self.members_to_compute = members_to_compute

    def integrate(
            self,
            start_time,
            stop_time,
            maxdt,
            exogenous_flag):
        """
        Integrate members assigned to the current integrator

        All three arguments are arguments to scipy.integrate.solve_ivp

        Input:
            start_time (float): start time of the integration interval
            stop_time (float): stop time of the integration interval
            maxdt (float): maximum timestep
        Output:
            None
        """
        y0_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['ensemble_state'])
        coefficients_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['coefficients'])
        closure_shm = shared_memory.SharedMemory(
                name=self.shared_memory_names['closure'])
        exog_shm = shared_memory.SharedMemory(
            name=self.shared_memory_names['prevalence_indep_exogenous_rates'])
        
        ensemble_state = np.ndarray((self.M, 6*self.N),
                                    dtype=np.float_,
                                    buffer=y0_shm.buf)
        coefficients   = np.ndarray((self.M, 8*self.N),
                                    dtype=np.float_,
                                    buffer=coefficients_shm.buf)
        closure        = np.ndarray((self.M, self.N),
                                    dtype=np.float_,
                                    buffer=closure_shm.buf)
        prevalence_indep_exogenous_rates = np.ndarray(self.N,
                                                      dtype=np.float_,
                                                      buffer=exog_shm.buf)

        t_span = np.array([start_time, stop_time])
        t_eval = np.array([stop_time])

        for j in self.members_to_compute:
            member_state        = ensemble_state[j]
            member_coefficients = coefficients[j]
            member_closure      = closure[j]

            compute_rhs_member = lambda t, y: (
                    self.compute_rhs(y, member_coefficients, member_closure, prevalence_indep_exogenous_rates,exogenous_flag)
            )

            ode_result = solve_ivp(fun=compute_rhs_member,
                                   t_span=t_span,
                                   y0=member_state,
                                   t_eval=t_eval,
                                   method='RK45',
                                   max_step=maxdt)
            ensemble_state[j] = np.clip(np.squeeze(ode_result.y), 0.0, 1.0)

        y0_shm.close()
        coefficients_shm.close()
        closure_shm.close()
        exog_shm.close()

        return

    def compute_rhs(
            self,
            member_state,
            member_coefficients,
            member_closure,
            prevalence_indep_exogenous_rates,
            exogenous_flag):
        """
        Compute right-hand side of master equations

        Input:
            member_state (np.array): (6*N,) array of states
            member_coefficients (np.array): (8*N,) array of coefficients of the
                                            linear part of the RHS
            member_closure (np.array): (N,) array of coefficients for S_i's
            prevalence_indep_exogenous_rates (N,): when mult. by prevalence, gives the exogenous rate
            exogenous_flag (Bool) : whether to use exogenous rates
        Output:
            rhs (np.array): (6*N,) right-hand side of master equations
        """
        S_substate = member_state[self.S_slice]
        E_substate = member_state[self.E_slice]
        I_substate = member_state[self.I_slice]
        H_substate = member_state[self.H_slice]
       
        (sigma,
         gamma,
         delta,
         xi,
         mu,
         gammap,
         xip,
         mup
        ) = member_coefficients.reshape( (8, -1) )

        rhs = np.empty(6*self.N)

        if not exogenous_flag:
            rhs[self.S_slice] = -member_closure * S_substate 
            rhs[self.E_slice] =  member_closure * S_substate - sigma * E_substate
        else:
            
            prevalence = I_substate.mean()
            
            rhs[self.S_slice] = -(member_closure + prevalence * prevalence_indep_exogenous_rates) * S_substate
            rhs[self.E_slice] =  (member_closure + prevalence * prevalence_indep_exogenous_rates) * S_substate - sigma * E_substate
        
        rhs[self.I_slice] = sigma * E_substate - gamma  * I_substate
        rhs[self.H_slice] = delta * I_substate - gammap * H_substate
        rhs[self.R_slice] = xi    * I_substate + xip    * H_substate
        rhs[self.D_slice] = mu    * I_substate + mup    * H_substate

        return rhs


