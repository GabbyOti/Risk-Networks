import os, sys; sys.path.append(os.path.join(".."))
from epiforecast.epiplots import plot_master_eqns

from epiforecast.contact_network import ContactNetwork
from epiforecast.populations import TransitionRates
from epiforecast.samplers import BetaSampler, GammaSampler
from epiforecast.risk_simulator import MasterEquationModelEnsemble
from epiforecast.measurements import TestMeasurement

import numpy as np
import networkx as nx

def random_state(population):
    """
    Returns a status dictionary associated with a random infection
    """
    status_catalog = ['S', 'E', 'I', 'H', 'R', 'D']
    status_weights = [0.8, 0.01, 0.15, 0.02, 0.01, 0.01]
    statuses = {node: status_catalog[np.random.choice(6, p = status_weights)] for node in range(population)}

    return statuses


np.random.seed(1)

minute = 1 / 60 / 24
hour = 60 * minute
day = 1.0

population = 1000
ensemble_size = 10

contact_graph = nx.watts_strogatz_graph(population, 12, 0.1, 1)
network = ContactNetwork.from_networkx_graph(contact_graph)


latent_periods              = GammaSampler(k=1.7, theta=2.0, minimum=2)
community_infection_periods = GammaSampler(k=1.5, theta=2.0, minimum=1)
hospital_infection_periods  = GammaSampler(k=1.5, theta=3.0, minimum=1)

hospitalization_fraction     = BetaSampler(mean=0.25, b=4)
community_mortality_fraction = BetaSampler(mean=0.02, b=4)
hospital_mortality_fraction  = BetaSampler(mean=0.04, b=4)

transition_rates = TransitionRates.from_samplers(
        network.get_node_count(),
        latent_periods,
        community_infection_periods,
        hospital_infection_periods,
        hospitalization_fraction,
        community_mortality_fraction,
        hospital_mortality_fraction)

transition_rates.calculate_from_clinical()
network.set_transition_rates_for_kinetic_model(transition_rates)

transmission_rate = 0.06 * np.ones(ensemble_size)

ensemble_model = MasterEquationModelEnsemble(
        network.get_node_count(),
        [transition_rates] * ensemble_size,
        transmission_rate,
        ensemble_size=ensemble_size)

np.random.seed(1)

I_fraction = 0.01
y0 = np.zeros([ensemble_size, 5 * population])

for mm, member in enumerate(ensemble_model.ensemble):
    infected = np.random.choice(population,
                                size=int(population * I_fraction),
                                replace=False)
    E, I, H, R, D = np.zeros([5, population])
    S = np.ones(population,)
    I[infected] = 1.
    S[infected] = 0.

    y0[mm, : ]  = np.hstack((S, I, H, R, D))

ensemble_model.set_states_ensemble(y0)
ensemble_model.set_mean_contact_duration(network.get_edge_weights())

static_contact_interval = 3 * hour
ode_states = ensemble_model.simulate(static_contact_interval, min_steps=100)

statuses = random_state(population)

print('\n0th Test: Probs in natural scale ----------------------------')

test = TestMeasurement('I')
test.update_prevalence(ensemble_model.states_trace[:,:,0], scale = None)
mean, var = test.take_measurements(statuses, scale = None)

print(np.vstack([np.array(list(statuses.values())), list(mean.values()), list(var.values())]).T[:5])

print('\n1st Test: Probs in natural scale ----------------------------')

test = TestMeasurement('I')
test.update_prevalence(ode_states, scale = None)
mean, var = test.take_measurements(statuses, scale = None)

print(np.vstack([np.array(list(statuses.values())), list(mean.values()), list(var.values())]).T[:5])

test = TestMeasurement('I')
test.update_prevalence(ode_states)
mean, var = test.take_measurements(statuses)

print('\n2nd Test: Probs in logit scale ------------------------------')
print(np.vstack([np.array(list(statuses.values())), list(mean.values()), list(var.values())]).T[:5])

print('\n3th Test: Hospitalized --------------------------------------')
test = TestMeasurement('H', specificity = .999, sensitivity = 0.999)
test.update_prevalence(ode_states, scale = None)
mean, var = test.take_measurements(statuses, scale = None)

print(np.vstack([np.array(list(statuses.values())), list(mean.values()), list(var.values())]).T[47:47+6])

print('\n4th Test: Hospitalized --------------------------------------')
test = TestMeasurement('H', specificity = .999, sensitivity = 0.999)
test.update_prevalence(ode_states)
mean, var = test.take_measurements(statuses)

print(np.vstack([np.array(list(statuses.values())), list(mean.values()), list(var.values())]).T[47:47+6])

print('\n4th Test: Noisy measurements for positive cases -------------')

test = TestMeasurement('I',noisy_measurement=True)
test.update_prevalence(ode_states, scale = None)
mean, var = test.take_measurements({node: 'I' for node in range(population)},
                                    scale = None)

positive_test, _ = test.take_measurements({0:'I'}, scale = None)
negative_test, _ = test.take_measurements({0:'S'}, scale = None)

positive_test = list(positive_test.values())[0]
negative_test = list(negative_test.values())[0]

print('Fraction of correct testing: %2.2f'%(np.array(list(mean.values())) == positive_test).mean())

print('\n5th Test: Noisy measurements for negative cases -------------')

mean, var = test.take_measurements({node: 'S' for node in range(population)},
                                    scale = None)

print('Fraction of correct testing: %2.2f'%(np.array(list(mean.values())) == negative_test).mean())


