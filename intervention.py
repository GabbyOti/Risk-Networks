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
import networkx as nx

class Intervention:
  """
    Store intervention strategy and compute interventions from state

    Currently can only do a simple strategy with thresholds for E(xposed) and
    I(infected) in a "binary or" fashion:
        apply intervention to node[i] if (E[i] > E_thr) or (I[i] > I_thr)

    The class is aware of ensemble members, i.e. E[i] and I[i] in the above are
    computed as ensemble means of respective E[i]'s and I[i]'s

    Methods:
        find_sick

    Example:
        network = ContactNetwork.from_files(edges_filename, identifiers_filename)
        N = network.get_count_node()
        M = ensemble_size
        intervention = Intervention(N, M, compartment_index, E_thr=0.7, I_thr=0.5)

        for k in range(time_steps):
            # kinetic, master, DA
            sick_nodes = intervention.find_sick(ensemble_states)
            network.isolate(sick_nodes)
  """

  def __init__(self, N, M, compartment_index, E_thr=0.5, I_thr=0.5):
    """
      Constructor

      Args:
        N:                    number of nodes
        M:                    number of ensemble members
        compartment_index:    dictionary mapping letters to indices
        E_thr:                threshold of the E compartment
        I_thr:                threshold of the I compartment
    """

    self.N = N
    self.M = M

    E = compartment_index['E']
    I = compartment_index['I']

    if E == -1: # reduced model, need to compute E from the rest
      self.E_slice = None
    else:
      self.E_slice = np.s_[E * N : (E+1) * N]

    if I == -1:
      self.I_slice = None
    else:
      self.I_slice = np.s_[I * N : (I+1) * N]

    assert 0 < E_thr < 1
    assert 0 < I_thr < 1
    self.E_thr = E_thr
    self.I_thr = I_thr

    self.stored_nodes_to_intervene = {}

  def __get_complement_substate(self, ensemble_states):
    """
      Get the complement substate for the whole ensemble

      Args:
        ensemble_states: (M,c*N) np.array, where c is the number of compartments

      Example:
        if the model is reduced and one of the states is implicitly computed
        (say, E), then we obtain it by subtracting from 1:
          1 - (S + I + H + R + D)
    """

    return 1 - ensemble_states.reshape( (self.M, -1, self.N) ).sum(axis=1)

  def find_sick(self, ensemble_states, user_nodes, sum_EI=False):
    """
      Find node indices that are considered sick according to E and I thresholds

      Args:
        ensemble_states: (M,c*N) np.array, where c is the number of compartments
    """

    # both substates are (M, N) in shape
    if self.E_slice is None:
      E_substate = self.__get_complement_substate(ensemble_states)
    else:
      E_substate = ensemble_states[:, self.E_slice]

    if self.I_slice is None:
      I_substate = self.__get_complement_substate(ensemble_states)
    else:
      I_substate = ensemble_states[:, self.I_slice]

    # both means are (N,) in shape
    E_ensemble_mean = E_substate.mean(axis=0)
    I_ensemble_mean = I_substate.mean(axis=0)

    if sum_EI == False:
        sick_user_subset = np.where(
            (E_ensemble_mean > self.E_thr) | (I_ensemble_mean > self.I_thr)
            )[0]
    else:
        sick_user_subset = np.where(
            (E_ensemble_mean + I_ensemble_mean) > (self.E_thr + self.I_thr)
            )[0]

    return user_nodes[sick_user_subset]

  def save_nodes_to_intervene(self, current_time, nodes_to_intervene):
      """
        Save sick nodes for intervention

        Args:
            current_time: float number
            nodes_to_intervene: (N,) np.array, where N is the number of sick nodes 
      """

      self.stored_nodes_to_intervene[current_time] = nodes_to_intervene
