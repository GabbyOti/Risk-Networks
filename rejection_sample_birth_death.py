from numba import njit, prange, float64
import numpy as np
from timeit import default_timer as timer

from scipy.special import roots_legendre

import matplotlib.pyplot as plt

# See discussion in 
#
# Christian L. Vestergaard , Mathieu Génois, "Temporal Gillespie Algorithm: Fast Simulation 
# of Contagion Processes on Time-Varying Networks", PLOS Computational Biology (2015)
#
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004579

@njit
def diurnal_inception_rate(λmin, λmax, t):
    return np.maximum(λmin, λmax * (1 - np.cos(np.pi * t)**4)**4)

@njit
def simulate_contact(
                     time,
                     time_step,
                     n_steps,
                     contact_duration, 
                     contact,
                     min_inception_rate,
                     max_inception_rate,
                     mean_contact_lifetime
                    ):

    # Initialize
    contact_duration = 0.0
    inceptions = 0
    step = 0

    while step < n_steps:

        # Advance to the next time-step
        step += 1
        time += time_step

        # Determine whether an event has occurred
        r = np.random.random()

        if contact: # deactivate?
            contact_duration += time_step # accmulate contact duration

            if r < time_step / mean_contact_lifetime: # deactivation
                contact = False
    
        else: # inception?

            inception_rate = diurnal_inception_rate(min_inception_rate, max_inception_rate, time)

            if r < time_step * inception_rate: # inception
                contact = True
                inceptions += 1

    return contact, contact_duration, inceptions

@njit(parallel=True)
def simulate_contacts(
                      time,
                      stop_time,
                      time_step,
                      contact_duration,
                      contact,
                      min_inception_rate,
                      max_inception_rate,
                      mean_contact_lifetime,
                      inceptions
                     ):

    n_steps = int(np.round((stop_time - time) / time_step))

    # Parallel loop over contacts
    for i in prange(len(contact_duration)):

         contact[i], contact_duration[i], inceptions[i] = simulate_contact(
                                                                           time,
                                                                           time_step,
                                                                           n_steps,
                                                                           contact_duration[i],
                                                                           contact[i],
                                                                           min_inception_rate[i],
                                                                           max_inception_rate[i],
                                                                           mean_contact_lifetime[i]
                                                                          )
                         

if __name__ == "__main__":

    n = 10000
    second = 1 / 60 / 60 / 24
    minute = 60 * second

    λmin = 3
    λmax = 22
    μ = 1 / minute
    time_step = 10 * second

    εmin = λmin / (μ + λmin)
    εmax = λmax / (μ + λmax)

    print("Equilibrium solution:", εmin)

    min_inception_rate = λmin * np.ones(n)
    max_inception_rate = λmax * np.ones(n)
    mean_contact_lifetime = 1 / μ * np.ones(n)

    event_time = np.zeros(n)
    inceptions = np.zeros(n)
    contact_duration = np.zeros(n)
    overshoot_duration = np.zeros(n)
    contacts = np.random.choice([False, True], size = n, p = [1 - εmin, εmin])

    dt = 1 / 24
    steps = int(7 / dt)

    start = timer()

    contact_durations = np.zeros((steps, 4))
    measured_inceptions = np.zeros((steps, 4))

    mean_contact_durations = np.zeros(steps)
    measurement_times = np.arange(start=0.0, stop=(steps+1)*dt, step=dt)

    for i in range(steps):

        simulate_contacts(
                          i * dt,
                          (i + 1) * dt,
                          time_step,
                          contact_duration,
                          contacts,
                          min_inception_rate,
                          max_inception_rate,
                          mean_contact_lifetime,
                          inceptions
                         )

        mean_contact_durations[i] = contact_duration.mean() / dt

        for j in range(contact_durations.shape[1]):
            contact_durations[i, j] = contact_duration[j] / dt
            measured_inceptions[i, j] = inceptions[j]


    end = timer()

    print("Simulation time:", end - start)
    print("Mean contact duration:", contact_duration.mean())

    fig, axs = plt.subplots(nrows=3, figsize=(14, 8), sharex=True)

    plt.sca(axs[0])

    for j in range(measured_inceptions.shape[1]):
        plt.plot(measurement_times[1:], measured_inceptions[:, j] / dt, '.', alpha=0.4,
                 label="Contact {}".format(j))

    plt.plot(measurement_times, measured_inceptions.mean() * np.ones(steps+1) / dt,
             linewidth=3.0, alpha=0.6, linestyle='-', label="mean")
             
    plt.plot(measurement_times, 11 * np.ones(len(measurement_times)),
             linewidth=1.0, alpha=0.8, color='k', linestyle='--', label="11 $ \mathrm{day^{-1}} $")

    plt.ylabel("Inception rate $ \mathrm{(day^{-1})} $")
    plt.legend(loc='upper right')

    plt.sca(axs[1])
    for j in range(contact_durations.shape[1]):
        plt.plot(measurement_times[1:], contact_durations[:, j], '.', alpha=0.6,
                 label="Contact {}".format(j))

    plt.ylabel("Mean contact durations, $T_i$")
    plt.legend(loc='upper right')

    plt.sca(axs[2])

    
    t = measurement_times
    λ = np.zeros(steps)
    for i in range(steps):
        λ[i] = diurnal_inception_rate(λmin, λmax, 1/2 * (t[i] + t[i+1]))

    plt.plot(t[1:], λ / (λ + μ), linestyle="-", linewidth=3, alpha=0.4, label="$ \lambda(t) / [ \mu + \lambda(t) ] $")

    plt.plot(measurement_times[1:], mean_contact_durations, linestyle="--", color="k", linewidth=1,
             label="$ \\bar{T}_i(t) $")

    plt.xlabel("Time (days)")
    plt.ylabel("Ensemble-averaged $T_i$")
    plt.legend(loc='upper right')

    plt.show()
