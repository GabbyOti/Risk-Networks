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

import warnings
import numpy as np

class EnsembleTimeSeries:
    """
    Store, add, read & process a time series composed of ensemble members

    An easy way to think about this, is a 3-tensor with:
        - 0th dimension equal to n_ensemble (size of the ensemble)
        - 1st dimension equal to n_vector (dimension of the stored quantity)
        - 2nd dimension equal to n_steps (total number of time steps)

    Once created, the timeseries cannot be changed in size (in any dimension).
    """

    def __init__(
            self,
            n_ensemble,
            n_vector,
            n_steps,
            n_roll_at_once=1):
        """
        Constructor

        Input:
            n_ensemble (int): ensemble size
            n_vector (int): dimension of the vector to store
            n_steps (int): number of time steps to store
            n_roll_at_once (int): discard this many timesteps from the start and
                                  shift along the 2nd dimension, creating space
                                  for new snapshots
        """
        assert n_steps > n_roll_at_once

        self.n_ensemble = n_ensemble
        self.n_vector   = n_vector
        self.n_steps    = n_steps

        self.container = np.empty( (n_ensemble, n_vector, n_steps) )
        self.n_roll_at_once = n_roll_at_once
        self.end = 0 # points to the past-the-end element

    def __getitem__(
            self,
            timestep):
        """
        A wrapper around get_snapshot; the same docstring applies
        """
        return self.get_snapshot(timestep)

    def get_snapshot(
            self,
            timestep):
        """
        Get the full ensemble snapshot at a specified timestep

        Input:
            timestep (int): timestep of a snapshot to return

        Output:
            snapshot (np.array): (n_ensemble, n_vector) array of values
        """
        if timestep >= self.end:
            raise ValueError(
                    self.__class__.__name__
                    + ": timestep is out of bounds, cannot get_snapshot"
                    + "; timestep: "
                    + str(timestep))

        return self.container[:,:,timestep]

    def get_snapshot_mean(
            self,
            timestep):
        """
        Get the ensemble mean of a snapshot at a specified timestep

        Input:
            timestep (int): timestep of a snapshot to return

        Output:
            snapshot_mean (np.array): (n_vector,) array of ensemble means
        """
        snapshot = self.get_snapshot(timestep)
        return snapshot.mean(axis=0)

    def get_mean(self):
        """
        Get the ensemble mean of the whole timeseries

        Output:
            timeseries_mean (np.array): (n_vector, n_steps) array of ensemble
                                        means
        """
        if self.end < self.n_steps:
            warnings.warn(
                    self.__class__.__name__
                    + ": mean of an incomplete container requested"
                    + "; the values starting from index "
                    + str(self.end)
                    + " are meaningless")

        return self.container.mean(axis=0)

    def push_back(
            self,
            snapshot):
        """
        Push an element to the back of time series

        If the container is full, shift the data along the 2nd dimension to
        discard the first `self.n_roll_at_once` elements.

        Input:
            snapshot (np.array): (n_ensemble, n_vector) array of values

        Output:
            None
        """
        # if we are at capacity, lose the first entry
        if self.end >= self.n_steps:
            self.container = np.roll(self.container, -self.n_roll_at_once, axis=2)
            self.end -= self.n_roll_at_once

        self.container[:,:,self.end] = snapshot
        self.end += 1


