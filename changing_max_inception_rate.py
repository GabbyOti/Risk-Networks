import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.autonotebook import tqdm

from epiforecast.contact_simulator import ContactSimulator
from epiforecast.contact_network import ContactNetwork

np.random.seed(1234)

################################################################################
# constants ####################################################################
################################################################################
minute = 1 / 60 / 24

dt = 0.1 / 24 # days
days = 10
steps = int(days / dt)

λ_min = 3  # minimum contact rate
λ_max = 50 # maximum contact rate

μ = 1.0 / minute
n_contacts_barabasi = 10000

################################################################################
# initialization ###############################################################
################################################################################
contact_graph = nx.barabasi_albert_graph(int(n_contacts_barabasi / 10), 10)
network = ContactNetwork.from_networkx_graph(contact_graph)
network.set_lambdas(λ_min, λ_max)
mean_degree = np.mean([d for n, d in network.get_graph().degree()])

simulator = ContactSimulator(network.get_edges(),
                             mean_degree,
                             day_inception_rate = λ_max,
                             night_inception_rate = λ_min,
                             mean_event_lifetime = 1 / μ,
                             start_time = -dt)

# Generate a time-series of contact durations and average number of active contacts
n_contacts = network.get_edge_count()
contact_durations = np.zeros((steps, n_contacts))
mean_contact_durations = np.zeros(steps)
measurement_times = np.arange(start=0.0, stop=(steps+1)*dt, step=dt)

(λ_min_nodal, λ_max_nodal) = network.get_lambdas()
simulator.run(stop_time = 0.0,
              current_edges = network.get_edges(),
              nodal_day_inception_rate = λ_max_nodal,
              nodal_night_inception_rate = λ_min_nodal)

edge_weights = simulator.compute_edge_weights()
network.set_edge_weights(edge_weights)

################################################################################
# main loop ####################################################################
################################################################################
start = timer()

for i in tqdm(range(steps), desc = 'Simulation', total = steps):

    stop_time = (i + 1) * dt

    if stop_time == 5:
        # this is for exposition only;
        # you can instead directly pass 5 as 'nodal_day_inception_rate' below
        network.set_lambda_max(5)

    (λ_min_nodal, λ_max_nodal) = network.get_lambdas()
    simulator.run(stop_time = stop_time,
                  current_edges = network.get_edges(),
                  nodal_day_inception_rate = λ_max_nodal,
                  nodal_night_inception_rate = λ_min_nodal)

    edge_weights = simulator.compute_edge_weights()
    network.set_edge_weights(edge_weights)

    mean_contact_durations[i] = simulator.contact_duration.mean()
    contact_durations[i,:]    = simulator.contact_duration

end = timer()

print("Simulated", network.get_edge_count(),
      "contacts in {:.3f} seconds".format(end - start))

################################################################################
# plot #########################################################################
################################################################################
fig, axs = plt.subplots(nrows=2, figsize=(10, 4), sharex=True)

plt.sca(axs[0])
for i in range(int(np.round(n_contacts/10))):
    plt.plot(measurement_times[1:],
             contact_durations[:, i] / dt,
             marker=".",
             mfc="xkcd:light pink",
             mec=None,
             alpha=0.02,
             markersize=0.1)

plt.plot(measurement_times[1:],
         mean_contact_durations / dt,
         linestyle="-",
         linewidth=3,
         color="xkcd:sky blue",
         label="Ensemble mean $ \\langle T_i \\rangle $")

plt.ylabel("Specific contact duration, $T_i$")
plt.legend(loc='upper right')

plt.sca(axs[1])
plt.plot(measurement_times[1:],
         mean_contact_durations / dt,
         linestyle="-",
         linewidth=1,
         color="xkcd:sky blue")

plt.xlabel("Time (days)")
plt.ylabel("$ \\langle T_i \\rangle $")

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axs[0].spines["bottom"].set_visible(False)

plt.show()
