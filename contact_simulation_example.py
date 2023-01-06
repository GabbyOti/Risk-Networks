import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.autonotebook import tqdm

from epiforecast.contact_simulator import ContactSimulator, diurnal_inception_rate
from epiforecast.contact_network import ContactNetwork

np.random.seed(1234)

################################################################################
# constants ####################################################################
################################################################################
minute = 1 / 60 / 24

dt = 1 / 24 # days
days = 7
steps = int(days / dt)

λ_min = 4  # minimum contact rate
λ_max = 84 # maximum contact rate

μ = 1.0 /(2.0 * minute)
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
contact_durations = np.zeros((steps, 4))
measured_inceptions = np.zeros((steps, 4))
mean_contact_durations = np.zeros(steps)
measurement_times = np.arange(start=0.0, stop=(steps+1)*dt, step=dt)

(λ_min_nodal, λ_max_nodal) = network.get_lambdas()
simulator.run(stop_time = 0.0,
              current_edges = network.get_edges(),
              nodal_day_inception_rate = λ_max_nodal,
              nodal_night_inception_rate = λ_min_nodal)

################################################################################
# main loop ####################################################################
################################################################################
start = timer()

for i in tqdm(range(steps), desc = 'Simulation', total = steps):
    stop_time = (i + 1) * dt

    (λ_min_nodal, λ_max_nodal) = network.get_lambdas()
    simulator.run(stop_time = stop_time,
                  current_edges = network.get_edges(),
                  nodal_day_inception_rate = λ_max_nodal,
                  nodal_night_inception_rate = λ_min_nodal)

    mean_contact_durations[i] = simulator.contact_duration.mean() / dt

    for j in range(contact_durations.shape[1]):
        contact_durations[i, j] = simulator.contact_duration[j] / dt

end = timer()

print("Simulated", network.get_edge_count(),
      "contacts in {:.3f} seconds".format(end - start))

################################################################################
# plot #########################################################################
################################################################################
fig, axs = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)

plt.sca(axs[0])
for j in range(contact_durations.shape[1]):
    plt.plot(measurement_times[1:],
             contact_durations[:,j],
             '.',
             alpha=0.6,
             label="Contact {}".format(j))

plt.ylabel("Mean contact durations, $T_i$")
plt.legend(loc='upper right')

plt.sca(axs[1])

t = measurement_times
λ = np.zeros(steps)
for i in range(steps):
    λ[i] = diurnal_inception_rate(λ_min, λ_max, mean_degree, 1/2 * (t[i] + t[i+1]))

plt.plot(t[1:],
         λ / (λ + μ),
         linestyle="-",
         linewidth=3,
         alpha=0.4,
         label="$ \lambda(t) / [ \mu + \lambda(t) ] $")

plt.plot(measurement_times[1:],
         mean_contact_durations,
         linestyle="--",
         color="k",
         linewidth=1,
         label="$ \\bar{T}_i(t) $")


plt.xlabel("Time (days)")
plt.ylabel("Ensemble-averaged $T_i$")
plt.legend(loc='upper right')

plt.show()
plt.savefig('simulated_contacts.png',
            rasterized=True,
            dpi=150)
