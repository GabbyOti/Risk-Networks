import numpy as np
import networkx as nx

from .samplers import AgeDependentBetaSampler, AgeDependentConstant, BetaSampler, GammaSampler

class TransitionRates:
    """
    Container for clinical parameters and transition rates.

    For readability, long names are abbreviated using the following
    glossary:
        lp  :   latent periods
        cip :   community infection periods
        hip :   hospital infection periods
        hf  :   hospitalization fraction
        cmf :   community mortality fraction
        hmf :   hospital mortality fraction
    """
    # TODO extract clinical parameters into its own class

    CLINICAL_PARAMETER_NAMES = {
            'latent_periods':               0,
            'community_infection_periods':  1,
            'hospital_infection_periods':   2,
            'hospitalization_fraction':     3,
            'community_mortality_fraction': 4,
            'hospital_mortality_fraction':  5
    }

    @classmethod
    def from_samplers(
            cls,
            population,
            lp_sampler,
            cip_sampler,
            hip_sampler,
            hf_sampler,
            cmf_sampler,
            hmf_sampler,
            distributional_parameters=None,
            lp_transform=None,
            cip_transform=None,
            hip_transform=None,
            hf_transform=None,
            cmf_transform=None,
            hmf_transform=None):
        """
        Create an object from clinical parameter samplers

        Input:
            population (int): population count
            *_sampler (int),
                      (float)   : a constant value for a parameter
                      (list)    : a list of parameters of length population
                      (np.array): (population,) array of parameters
                      (BetaSampler),
                      (GammaSampler),
                      (AgeDependentConstant),
                      (AgeDependentBetaSampler): a sampler to use
            distributional_parameters (np.array): (population,) array
                                      (None): np.ones(population) is used in
                                              this case
            *_transform (str): a type of transform applied to a parameter
                        (None): no transform

        Output:
            transition_rates (TransitionRates): initialized object
        """
        clinical_samplers = {
                'lp' : lp_sampler,
                'cip': cip_sampler,
                'hip': hip_sampler,
                'hf' : hf_sampler,
                'cmf': cmf_sampler,
                'hmf': hmf_sampler
        }

        if distributional_parameters is None:
            distributional_parameters = np.ones(population, dtype=int)
        assert distributional_parameters.shape == (population,)

        clinical_parameters = cls.__draw_clinical_using(
                clinical_samplers,
                distributional_parameters)

        clinical_transforms = {
                'lp' : lp_transform,
                'cip': cip_transform,
                'hip': hip_transform,
                'hf' : hf_transform,
                'cmf': cmf_transform,
                'hmf': hmf_transform
        }

        return cls(population,
                   clinical_parameters,
                   clinical_transforms)

    def __init__(
            self,
            population,
            clinical_parameters,
            clinical_transforms):
        """
        Constructor

        Input:
            population (int): population count
            clinical_parameters (dict): mapping short_name -> values
            clinical_transforms (dict): mapping short_name -> transform
        """
        self.population  = population

        self.latent_periods               = clinical_parameters['lp']
        self.community_infection_periods  = clinical_parameters['cip']
        self.hospital_infection_periods   = clinical_parameters['hip']
        self.hospitalization_fraction     = clinical_parameters['hf']
        self.community_mortality_fraction = clinical_parameters['cmf']
        self.hospital_mortality_fraction  = clinical_parameters['hmf']

        self.lp_transform  = clinical_transforms['lp']
        self.cip_transform = clinical_transforms['cip']
        self.hip_transform = clinical_transforms['hip']
        self.hf_transform  = clinical_transforms['hf']
        self.cmf_transform = clinical_transforms['cmf']
        self.hmf_transform = clinical_transforms['hmf']

        self.exposed_to_infected       = None
        self.infected_to_resistant     = None
        self.infected_to_hospitalized  = None
        self.infected_to_deceased      = None
        self.hospitalized_to_resistant = None
        self.hospitalized_to_deceased  = None

    def __getitem__(
            self,
            user_nodes):
        """
        Get a slice of clinical parameters and transition rates for user nodes

        Input:
            user_nodes (np.array): (n_user_nodes,) array of user node indices

        Output:
            user_transition_rates (TransitionRates): a slice of self
        """
        n_user_nodes = user_nodes.size

        clinical_parameters = self.get_clinical_parameters_as_dict()
        user_clinical_parameters = {}
        for short_name, parameter in clinical_parameters.items():
            if isinstance(parameter, (int, float)):
                user_clinical_parameters[short_name] = parameter
            else:
                user_clinical_parameters[short_name] = parameter[user_nodes]

        clinical_transforms = self.get_clinical_transforms()

        return TransitionRates(n_user_nodes,
                               user_clinical_parameters,
                               clinical_transforms)

    def transform_clinical_parameter(
            self,
            clinical_parameter,
            transform_type):
        """
        Transforms the clinical parameter by the transform type and returns the output

        Args
        ----
        clinical_parameter(np.array): saved clinical parameter
        transform_type    (string): a string defining the transform implemented :
         - None = no transform required
         - 'log'  = clinical parameter is the logarithm of the desired object, so we exponentiate.
        """
        if transform_type is None:
            return clinical_parameter
        elif transform_type == 'log':
            return np.exp(clinical_parameter)
        else:
            raise ValueError("transform_type not recognised, choose from None (default) or 'log' ")

    def add_noise_to_clinical_parameters(
            self,
            parameter_str,
            noise_level):
        """
        Adds Gaussian Noise to the stored clinical_parameter (elementwise)

        Args
        ----
        noise_level (list of Floats): Size of standard deviation of the noise
        parameter_string (list of strings): the parameters to add noise too
        """
        for (lvl,par_str) in zip(noise_level,parameter_str):
            clinical_parameter = self.get_clinical_parameter(par_str)
            noise = np.random.normal(0,lvl,clinical_parameter.shape)
            setattr(self, par_str, clinical_parameter + noise)

    def calculate_from_clinical(self):
        """
        Calculate transition rates using the current clinical parameters

        Output:
            None
        """
        lp  = self.transform_clinical_parameter(self.latent_periods,              self.lp_transform)
        cif = self.transform_clinical_parameter(self.community_infection_periods, self.cip_transform)
        hip = self.transform_clinical_parameter(self.hospital_infection_periods,  self.hip_transform)
        hf  = self.transform_clinical_parameter(self.hospitalization_fraction,    self.hf_transform)
        cmf = self.transform_clinical_parameter(self.community_mortality_fraction,self.cmf_transform)
        hmf = self.transform_clinical_parameter(self.hospital_mortality_fraction, self.hmf_transform)

        σ       = self.__broadcast_to_array(1 / lp)
        γ       = self.__broadcast_to_array(1 / cif)
        γ_prime = self.__broadcast_to_array(1 / hip)
        h       = self.__broadcast_to_array(hf)
        d       = self.__broadcast_to_array(cmf)
        d_prime = self.__broadcast_to_array(hmf)

        self.exposed_to_infected      = dict(enumerate(σ))
        self.infected_to_resistant    = dict(enumerate((1 - h - d) * γ))
        self.infected_to_hospitalized = dict(enumerate(h * γ))
        self.infected_to_deceased     = dict(enumerate(d * γ))
        self.hospitalized_to_resistant= dict(enumerate((1 - d_prime) * γ_prime))
        self.hospitalized_to_deceased = dict(enumerate(d_prime * γ_prime))

    def get_transition_rate(
            self,
            name):
        """
        Get a transition rate by its name as np.array

        Input:
            name (str): rate name, like 'exposed_to_infected'
        Output:
            rate (np.array): (self.population,) array of values
        """
        rate_dict = getattr(self, name)
        return np.fromiter(rate_dict.values(), dtype=np.float_)

    def get_clinical_transforms(self):
        """
        Get clinical parameter transforms

        Output:
            clinical_transforms (dict): mapping short_name -> transform
        """
        clinical_transforms = {
                'lp' : self.lp_transform,
                'cip': self.cip_transform,
                'hip': self.hip_transform,
                'hf' : self.hf_transform,
                'cmf': self.cmf_transform,
                'hmf': self.hmf_transform
        }
        return clinical_transforms

    def get_clinical_parameters_total_count(self):
        """
        Get the total number of values of all clinical parameters

        Output:
            n_parameters (int): total number of clinical parameters values
        """
        n_parameters = 0
        for name in self.CLINICAL_PARAMETER_NAMES:
            n_parameters += self.get_clinical_parameter_count(name)

        return n_parameters

    def get_clinical_parameter_count(
            self,
            name):
        """
        Get the total number of values of a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            n_parameter (int): total number of the clinical parameter values
        """
        parameter = self.get_clinical_parameter(name)
        if isinstance(parameter, (int, float)):
            return 1
        elif isinstance(parameter, np.ndarray):
            return parameter.size
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": cannot infer the number of values of parameter: "
                    + name
                    + "; it is of the type: "
                    + values.__class__.__name__)

    def get_clinical_parameters_as_dict(self):
        """
        Get values of all clinical parameters as dict

        Output:
            clinical_parameters (dict): mapping short_name -> values
        """
        clinical_parameters = {
                'lp' : self.latent_periods,
                'cip': self.community_infection_periods,
                'hip': self.hospital_infection_periods,
                'hf' : self.hospitalization_fraction,
                'cmf': self.community_mortality_fraction,
                'hmf': self.hospital_mortality_fraction
        }
        return clinical_parameters

    def get_clinical_parameters_as_array(self):
        """
        Get values of all clinical parameters as np.array

        The order is the same as specified in CLINICAL_PARAMETER_NAMES.

        Output:
            clinical_parameters (np.array): (n_parameters,) array of values
        """
        clinical_parameters_list = []
        for name in self.CLINICAL_PARAMETER_NAMES:
            clinical_parameters_list.append(self.get_clinical_parameter(name))

        clinical_parameters = np.hstack(clinical_parameters_list)
        return clinical_parameters

    def get_clinical_parameter_indices(
            self,
            name):
        """
        Get indices of a clinical parameter by its name in concatenated array

        The indices are consistent with 'get_clinical_parameters_as_array'
        method, i.e. can be used to get slices of a particular parameter:
            clinical_array = transition_rates.get_clinical_parameters_as_array()
            lp_indices = transition_rates.get_clinical_parameter_indices(
                    'latent_periods')
            latent_periods = clinical_array[lp_indices]

        It is identical to calling:
            latent_periods = transition_rates.get_clinical_parameter(
                    'latent_periods')
        but provides more flexibility (e.g. when storing and accessing
        parameters as arrays)

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            indices (np.array): (k,) array of indices
        """
        start_index = 0
        for iteration_name in self.CLINICAL_PARAMETER_NAMES:
            if iteration_name == name:
                break
            start_index += self.get_clinical_parameter_count(iteration_name)

        end_index = start_index + self.get_clinical_parameter_count(name)
        return np.r_[start_index : end_index]

    def get_clinical_parameter(
            self,
            name):
        """
        Get a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
        Output:
            clinical_parameter (int),
                               (float):    constant value of a parameter
                               (np.array): (self.population,) array of values
        """
        return getattr(self, name)

    def set_clinical_parameter(
            self,
            name,
            value):
        """
        Set a clinical parameter by its name

        Input:
            name (str): parameter name, like 'latent_periods'
            value (int),
                  (float):    constant value for a parameter
                  (np.array): (self.population,) array of values
        Output:
            None
        """
        setattr(self, name, value)

    # TODO _maybe_ move this into utilities.py or something
    def __broadcast_to_array(
            self,
            values):
        """
        Broadcast values to numpy array if they are not already

        Input:
            values (int),
                   (float): constant value to be broadcasted
                   (np.array): (population,) array of values

        Output:
            values_array (np.array): (population,) array of values
        """
        if isinstance(values, (int, float)):
            return self.__broadcast_to_array_const(values)
        elif isinstance(values, np.ndarray):
            return self.__broadcast_to_array_array(values)
        else:
            raise ValueError(
                    self.__class__.__name__
                    + ": this type of argument is not supported: "
                    + values.__class__.__name__)

    def __broadcast_to_array_const(
            self,
            value):
        return np.full(self.population, value)

    def __broadcast_to_array_array(
            self,
            values):
        return values

    @classmethod
    def __draw_clinical_using(
            cls,
            samplers,
            distributional_parameters):
        """
        Draw clinical parameters using samplers and distributional parameters

        Samplers provided in the 'from_samplers' method for clinical parameters
        (periods and fractions) might be either distributions that depend on
        certain parameters (say, mean and variance in the Gaussian case), or
        arrays, or constant values.
        This method uses those samplers to draw from distributions according to
        the provided distributional parameters:
          - constant values and arrays are left unchanged;
          - samplers are used to draw an array of size
            'distributional_parameters.size'.

        Note that this method is NOT idempotent, i.e. in the following
            TransitionRates.__draw_clinical_using(samplers, dp)
            TransitionRates.__draw_clinical_using(samplers, dp)
        second call draws new samples (for those clinical parameters which had
        samplers specified in the argument).
        It leaves others unchanged though.

        Input:
            samplers (dict): mapping short_name -> sampler (see 'from_samplers')
            distributional_parameters (np.array): (population,) array

        Output:
            clinical_parameters (dict): mapping short_name -> values
        """
        lp  = cls.__draw_using(samplers['lp'],  distributional_parameters)
        cip = cls.__draw_using(samplers['cip'], distributional_parameters)
        hip = cls.__draw_using(samplers['hip'], distributional_parameters)
        hf  = cls.__draw_using(samplers['hf'],  distributional_parameters)
        cmf = cls.__draw_using(samplers['cmf'], distributional_parameters)
        hmf = cls.__draw_using(samplers['hmf'], distributional_parameters)

        clinical_parameters = {
                'lp' : lp,
                'cip': cip,
                'hip': hip,
                'hf' : hf,
                'cmf': cmf,
                'hmf': hmf
        }

        return clinical_parameters

    @classmethod
    def __draw_using(
            cls,
            sampler,
            distributional_parameters):
        """
        Draw samples using sampler and its distributional parameters

        Input:
            sampler (int),
                    (float),
                    (np.array): value(s) that are returned unchanged
                    (list): values; transformed into np.array
                    (BetaSampler),
                    (GammaSampler),
                    (AgeDependentBetaSampler),
                    (AgeDependentConstant): samplers to use for drawing
            distributional_parameters (iterable):
                an object used for sampling; redundant in (int), (float),
                (np.array), (list) cases
        Output:
            samples (int),
                    (float): same as `sampler` for (int), (float) cases
                    (np.array): array of samples
        """
        dp = distributional_parameters
        if isinstance(sampler, (int, float, np.ndarray)):
            return cls.__draw_using_const_array(sampler, dp)
        elif isinstance(sampler, list):
            return cls.__draw_using_list(sampler, dp)
        elif isinstance(sampler, (BetaSampler, GammaSampler)):
            return cls.__draw_using_sampler(sampler, dp)
        elif isinstance(sampler, AgeDependentBetaSampler):
            return cls.__draw_using_age_beta_sampler(sampler, dp)
        elif isinstance(sampler, AgeDependentConstant):
            return cls.__draw_using_age_const(sampler, dp)
        else:
            raise ValueError(
                    cls.__class__.__name__
                    + ": this type of argument is not supported: "
                    + sampler.__class__.__name__)

    @staticmethod
    def __draw_using_const_array(parameter, distributional_parameters):
        return parameter

    @staticmethod
    def __draw_using_list(parameter_list, distributional_parameters):
        return np.array(parameter_list)

    @staticmethod
    def __draw_using_sampler(sampler, distributional_parameters):
        return np.array([sampler.draw() for param in distributional_parameters])

    @staticmethod
    def __draw_using_age_beta_sampler(sampler, distributional_parameters):
        return np.array([sampler.draw(param)
                         for param in distributional_parameters])

    @staticmethod
    def __draw_using_age_const(sampler, distributional_parameters):
        return np.array([sampler.constants[param]
                         for param in distributional_parameters])

#take a mean over the network, to obtain an ensemble of parameters
def extract_ensemble_transition_rates(
        transition_rates_ensemble,
        num_params=6):
    ensemble_size = len(transition_rates_ensemble)
    ensemble_transition_rates_array = np.zeros((ensemble_size, num_params))
    for i in range(ensemble_size):
        ensemble_transition_rates_array[i,:] = np.mean(
                transition_rates_ensemble[i].get_clinical_parameters_as_array().reshape(num_params,-1),
                axis=1)
    return ensemble_transition_rates_array

#take a mean over the ensemble to obtain network parameters
def extract_network_transition_rates(
        transition_rates_ensemble,
        num_users,
        num_params=6):
    ensemble_size = len(transition_rates_ensemble)
    network_transition_rates_array = np.zeros((num_params,num_users))
    for i in range(ensemble_size):
        network_transition_rates_array += (1.0 / ensemble_size) * transition_rates_ensemble[i].get_clinical_parameters_as_array().reshape(num_params,-1)
    
    return network_transition_rates_array
