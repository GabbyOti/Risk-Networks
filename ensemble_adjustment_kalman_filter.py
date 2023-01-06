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

import os
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import time
import warnings
 
class EnsembleAdjustmentKalmanFilter:

    def __init__(
            self,
            data_transform,
            elementwise_reg = False,
            joint_cov_noise = 1e-2,
            obs_cov_noise = 1e-2,
            inflate_states = False,
            inflate_reg = 0.1,
            additive_inflate=False,
            additive_inflate_factor=0.1,
            inflate_transmission_reg=1.0,
            mass_conservation_flag=True,
            output_path=None):
        '''
        Instantiate an object that implements an Ensemble Adjustment Kalman Filter.

        Flags:
            * inflate_states: enable the inflation of states if True

        Key functions:
            * eakf.obs
            * eakf.update
            * eakf.compute_error
        Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        '''
        
        self.data_transform = data_transform
        self.error = np.empty(0)
        self.elementwise_reg = elementwise_reg
        self.joint_cov_noise = joint_cov_noise
        self.obs_cov_noise = obs_cov_noise
        self.inflate_states = inflate_states
        self.inflate_reg = inflate_reg  # unit is in (%)
        self.additive_inflate = additive_inflate
        self.additive_inflate_factor = additive_inflate_factor
        self.inflate_transmission_reg = inflate_transmission_reg,
        self.mass_conservation_flag = mass_conservation_flag
        self.output_path = output_path

        # Compute error
    def compute_error(
            self,
            x,
            x_t,
            cov):
        diff = x_t - x.mean(0)
        error = diff.dot(np.linalg.solve(cov, diff))
        # Normalize error
        norm = x_t.dot(np.linalg.solve(cov, x_t))
        error = error/norm

        self.error = np.append(self.error, error)


    # x: forward evaluation of state, i.e. x(q), with shape (num_ensembles, num_elements)
    # q: model parameters, with shape (num_ensembles, num_elements)
    def update(
            self,
            ensemble_state,
            all_initial_ensemble_state,
            clinical_statistics,
            transmission_rates,
            truth,
            cov,
            H_obs,
            print_error=False,
            r=1.0,
            inflate_indices=None,
            save_matrices=False,
            save_matrices_name=None,
            verbose=False):

        '''
        - ensemble_state (np.array): J x M of update states for each of the J ensembles

        - clinical_statistics (np.array): transition rate model parameters for each of the J ensembles

        - transmission_rates (np.array): transmission rate of model parameters for each of the J ensembles

        - truth (np.array): M x 1 array of observed states.

        - cov (np.array): M x M array of covariances that represent observational uncertainty.
                          For example, an array of 0's represents perfect certainty.
                          Off-diagonal elements represent the fact that observations of state
                          i may not be independent from observations of state j. For example, this
                          can occur when a test applied to person ni alters the certainty of a subsequent
                          test to person nj.
                          We assume the covariance is diagonal in this code.

        #TODO: how to deal with no transition and/or transmission rates. i.e empty array input.
               (Could we just use an ensemble sized column of zeros? then output the empty array
                '''
        
        assert (truth.ndim == 1), 'EAKF init: truth must be 1d array'
        assert (cov.ndim == 2), 'EAKF init: covariance must be 2d array'
        assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
            'EAKF init: truth and cov are not the correct sizes'
        output_path = self.output_path


        # Observation data statistics at the observed nodes
        x_t = truth
        cov = r**2 * cov
    
        #add reg to observations too
        cov = cov + np.diag(self.obs_cov_noise*np.ones(cov.shape[0]))        

        # [0.] Process to include mass conservation in [X.] stages                
        # [1.] Augment states with sum_states
        x = self.data_transform.apply_transform(ensemble_state)
        if self.mass_conservation_flag:
            xall = all_initial_ensemble_state #do not transform the sum states            
            xsum = xall.sum(axis=1)[:,np.newaxis] #100 x 1
            x = np.hstack([x, xsum])
            
            # [2.] Augment the observations with the desired mass (1) for sum_states
            x_t = np.hstack([truth,1.]) 
            cov = np.diag(np.hstack([np.diag(cov),np.min(np.diag(cov)) ]))

        try:
            # We assume independent variances (i.e diagonal covariance)
            cov_inv = np.diag(1/np.diag(cov))
        except np.linalg.linalg.LinAlgError:
            print('cov not invertible')
            cov_inv = np.ones(cov.shape)

        if save_matrices:
            save_cov_file =os.path.join(output_path, 'cov_matrix'+save_matrices_name+'.npy')
            np.save(save_cov_file, cov)
        
        if self.mass_conservation_flag:
            # [3.] Augment H_obs
            Hsum_obs = np.zeros((H_obs.shape[0] + 1,H_obs.shape[1]+1))
            Hsum_obs[:-1,:-1] = H_obs
            Hsum_obs[-1,-1] = 1
            H_obs = Hsum_obs
        
        if not self.data_transform.name == "identity_clip":
            if verbose:
                print("mean state (transformed) pre DA", x.mean(axis=0))
        if verbose:
            print("mean state (untransformed) pre DA",ensemble_state.mean(axis=0))
            print("obs_var", np.diag(cov))
            #print("joint state cov", np.cov(x.T))
       
        # Stack parameters and states
        # the transition and transmission parameters act similarly in the algorithm
        p = clinical_statistics
        q = transmission_rates

        #if only 1 state is given
        if (ensemble_state.ndim == 1):
            x=x[np.newaxis].T

        # Whittaker Hamill 01 Inflation prior to update
        #x_bar = x.mean(axis=0)
        #x_inflated = self.inflate_reg * (x - x_bar) + x_bar
        #x[:,inflate_indices] = x_inflated[:,inflate_indices]
            
        # if all present we do [transition, transmission, joint_state, sum_state]
        if p.size>0 and q.size>0:
            zp = np.hstack([p, q, x])
        elif p.size>0 and q.size==0:
            zp = np.hstack([p, x])
        elif q.size>0 and p.size==0:
            zp = np.hstack([q, x])
        else:
            zp = x
        
        # Ensemble size
        J = x.shape[0]

        # Sizes of q and x
        pqs = q[0].size + p[0].size
        xs = x[0].size  
        xt = x_t.size #observations over all times + 1 
        
        zp_bar = np.mean(zp, 0)
        
        if save_matrices:
            save_state_file =os.path.join(output_path, 'state_matrix'+save_matrices_name+'.npy')
            np.save(save_state_file, (zp-zp_bar).T)

               
        # Follow Anderson 2001 Month. Weath. Rev. Appendix A.
        # Performing the first SVD of EAKF
        svd_failed = False
        num_svd_attempts = 0
        
        # build observation operators with parameters, states
        H = np.hstack([np.zeros((xt, pqs)), H_obs])
        Hpq = np.hstack([np.eye(pqs), np.zeros((pqs, xs))])
        Hs = np.hstack([np.zeros((xs, pqs)), np.eye(xs)])
        
        if save_matrices:
            save_H_file =os.path.join(output_path, 'H_matrix'+save_matrices_name+'.npy')
            np.save(save_H_file, H)
        

        # if ensemble_size < observations size, we pad the singular value matrix with added noise
        # unlikely but possible situation
        if zp.shape[0] < zp.shape[1]:    
            
            try:
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
            except:
                print("First SVD not converge!", flush=True)
                np.save(os.path.join(output_path, 'svd_matrix_1.npy'),
                        (zp-zp_bar).T)
                svd_failed = True
            while svd_failed == True:
                num_svd_attempts = num_svd_attempts+1
                np.random.seed(num_svd_attempts*100)
                try:
                    svd_failed = False
                    F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                except:
                    svd_failed = True 
                    print("First SVD not converge!",flush=True)
            F = F_full[:,:J-1]
            rtDp_vec = rtDp_vec[:-1]
            rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec
            rtDp_vec_full = np.zeros(zp.shape[1])
            rtDp_vec_full[:J-1] = rtDp_vec
            Dp_vec_full = rtDp_vec_full**2 + np.maximum(self.joint_cov_noise*(rtDp_vec_full[0]**2 - rtDp_vec_full[J-2]**2), np.mean(np.diag(cov))) # a little regularizations
            
            Dp = np.diag(Dp_vec_full)
        
        else:   
            try:
                F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
            except:
                print("First SVD not converge!", flush=True)
                np.save(os.path.join(output_path, 'svd_matrix_1.npy'),
                        (zp-zp_bar).T)
                svd_failed = True
            while svd_failed == True:
                num_svd_attempts = num_svd_attempts+1
                np.random.seed(num_svd_attempts*100)
                try:
                    svd_failed = False
                    F_full, rtDp_vec, _ = la.svd((zp-zp_bar).T, full_matrices=True)
                except:
                    svd_failed = True 
                    print("First SVD not converge!")
            F = F_full
           
            rtDp_vec = 1./np.sqrt(J-1) * rtDp_vec 
            if self.elementwise_reg:
                rtDp_vec = rtDp_vec + np.maximum(self.joint_cov_noise*(rtDp_vec), np.sqrt(np.mean(np.diag(cov))))
                Dp_vec_full = rtDp_vec**2
            else:
                Dp_vec_full = rtDp_vec**2 + np.maximum(self.joint_cov_noise*(rtDp_vec[0]**2 - rtDp_vec[-1]**2), np.mean(np.diag(cov))) # a little regularizations
            
            #print("Dp_vec_full",Dp_vec_full)
            Dp = np.diag(Dp_vec_full)

        # compute np.linalg.multi_dot([F_full, Dp, F_full.T])            
        #Sigma = np.linalg.multi_dot([np.multiply(F_full, np.diag(Dp)),F_full.T])
        
        G_full = np.diag(np.sqrt(Dp_vec_full))
        G_inv_full = np.diag(1./np.sqrt(Dp_vec_full))
            
        # Performing the second SVD of EAKF in the full space
        # computation of multidot([G_full.T, F_full.T, H.T, np.sqrt(cov_inv)])
        svd_failed = False
        num_svd_attempts = 0
        try:
            U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                   full_matrices=True)
        except:
            print("Second SVD not converge!", flush=True)
            np.save(os.path.join(output_path, 'svd_matrix_2.npy'), \
                    np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, \
                    np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]))
            svd_failed = True
        while svd_failed == True:
            num_svd_attempts = num_svd_attempts+1
            print(num_svd_attempts, flush=True)
            np.random.seed(num_svd_attempts*100)
            try:
                svd_failed = False
                U, rtD_vec, _ = la.svd(np.linalg.multi_dot([np.multiply(F_full,np.diag(G_full)).T, np.multiply(H.T, np.sqrt(np.diag(cov_inv)))]), \
                                   full_matrices=True)
            except:
                svd_failed = True 
                print("Second SVD not converge!", flush=True)
        D_vec = np.zeros(F_full.shape[0])
        D_vec[:cov_inv.shape[0]] = rtD_vec**2

        B = np.diag((1.0 + D_vec) ** (-1.0 / 2.0))
        #Computation of multi_dot([F_full, G.T,U,B.T,G_inv,F_full.T]) first by creating without F_full.T and multiply after by it.     
        AnoFt = np.linalg.multi_dot([np.multiply(F_full, np.diag(G_full)), np.multiply(np.multiply(U,np.diag(B)), np.diag(G_inv_full))])
        A = AnoFt.dot(F_full.T)
        # so overall: A = np.linalg.multi_dot([np.multiply(F_full, np.diag(G)), np.multiply(U,np.diag(B)), np.multiply(F_full,np.diag(G_inv)).T])
        Sigma_u = np.linalg.multi_dot([np.multiply(AnoFt,np.diag(Dp)),AnoFt.T])
        # so overall Sigma_u = np.linalg.multi_dot([A, Sigma, A.T])

        # compute np.linalg.multi_dot([F_full, inv(Dp), F_full.T])           
        Sigma_inv = np.linalg.multi_dot([np.multiply(F_full,1/np.diag(Dp)), F_full.T])
        
        zu_bar = np.dot(Sigma_u, \
                           (np.dot(Sigma_inv, zp_bar) + np.linalg.multi_dot([ np.multiply(H.T,np.diag(cov_inv)), x_t])))

        # Update parameters and state in `zu`
        zu = np.dot(zp - zp_bar, A.T) + zu_bar

        # Store updated parameters and states
        x_logit = np.dot(zu, Hs.T)

        # Avoid overflow for exp
        x_logit = np.minimum(x_logit, 1e2)
        
        # Applying Whittaker style inflation after update instead
        x_logit_bar = x_logit.mean(axis=0)
        x_logit_inflated = self.inflate_reg * (x_logit - x_logit_bar) + x_logit_bar
        x_logit[:,inflate_indices] = x_logit_inflated[:,inflate_indices]
        if self.additive_inflate == True:
            # Additional additive inflation
            x_logit[:,inflate_indices] = x_logit[:,inflate_indices] + \
                        np.random.normal(0*x_logit_bar, \
                        np.maximum(self.additive_inflate_factor*x_logit_bar,0.0), \
                        x_logit.shape)[:,inflate_indices]

        if not self.data_transform.name == "identity_clip":
            if verbose:
                print("mean joint-state (transformed) post DA", x_logit.mean(axis=0))

      # [4.] remove summed state
        if self.mass_conservation_flag:
            new_ensemble_state = self.data_transform.apply_inverse_transform(x_logit[:,:-1])
        else:
            new_ensemble_state = self.data_transform.apply_inverse_transform(x_logit)
        if verbose:
            print("mean state (untransformed) post DA",new_ensemble_state.mean(axis=0))
        
        if save_matrices:
            save_state_file =os.path.join(output_path, 'updated_state_matrix'+save_matrices_name+'.npy')
            np.save(save_state_file, (zu-zu_bar).T)

        pqout=np.dot(zu,Hpq.T)
        new_clinical_statistics = pqout[:, :clinical_statistics.shape[1]]
        new_transmission_rates  = pqout[:, clinical_statistics.shape[1]:]

        # Applying Whittaker style inflation after update to parameters
        new_transmission_rates_bar = new_transmission_rates.mean(axis=0)
        new_transmission_rates = self.inflate_transmission_reg * (new_transmission_rates - new_transmission_rates_bar) + new_transmission_rates_bar
        
        if (ensemble_state.ndim == 1):
            new_ensemble_state=new_ensemble_state.squeeze()

        # Compute error
        if print_error:
            self.compute_error(np.dot(x_logit, H_obs.T),x_t,cov)

        return new_ensemble_state, new_clinical_statistics, new_transmission_rates
