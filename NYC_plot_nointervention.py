import os, sys; sys.path.append(os.path.join(".."))

from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import random 
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib import rcParams

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

from epiforecast.scenarios import load_edges

def simulation_average(model_data, times, sampling_time = 1):
    """
    Returns daily averages of simulation data.
    """
    
    simulation_data_average = {}
    daily_average = {}

    for key in model_data.keys():
        simulation_data_average[key] = []
        daily_average[key] = []
    
    tav = 0

    for i in range(len(times)):
        for key in model_data.keys():
            simulation_data_average[key].append(model_data[key][i])

        if times[i] >= tav:
            for key in model_data.keys():
                daily_average[key].append(np.mean(simulation_data_average[key]))
                simulation_data_average[key] = []

            tav += sampling_time

    return daily_average

edges = load_edges(os.path.join('..', 'data', 'networks', 'edge_list_SBM_1e4_nobeds.txt')) 

contact_network = nx.Graph()
contact_network.add_edges_from(edges)
contact_network = nx.convert_node_labels_to_integers(contact_network)
population = len(contact_network)

# plot simulation data and empirical data
simulation_data = np.loadtxt('simulation_data_nointervention.txt')

times = simulation_data[:,0]

kinetic_model = {'S': simulation_data[:,1], 'E': simulation_data[:,2], 'I': simulation_data[:,3], 'R': simulation_data[:,4], 'H': simulation_data[:,5], 'D': simulation_data[:,6]}

NYC_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'data_new_york.csv'))
NYC_cases = np.asarray([float(x) for x in NYC_data['Cases'].tolist()])
NYC_deaths =  np.asarray([float(x) for x in NYC_data['Deaths'].tolist()])
NYC_date_of_interest = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_data['DATE_OF_INTEREST'].tolist()])

# population of NYC
NYC_population = 8.399e6

# fraction reported cases
fraction_reported = 0.13

# cumulative cases NYC
cumulative_reported_cases_NYC = 1/fraction_reported*np.cumsum(NYC_cases)/NYC_population
cumulative_deaths_NYC = np.cumsum(NYC_deaths)/NYC_population*1e5

reported_cases_NYC = 1/fraction_reported*NYC_cases/NYC_population
reported_deaths_NYC = NYC_deaths/NYC_population*1e5

# daily averages of simulation data
# sampling_time = 1 means that we average over 1-day intervals
sampling_time = 2
daily_average = simulation_average(kinetic_model, times, sampling_time)
cumulative_cases_simulation = 1-np.asarray(daily_average['S'])/population
cumulative_deaths_simulation = np.asarray(daily_average['D'])/population*1e5

reported_deaths_simulation = np.asarray(daily_average['D'])/population*1e5
reported_deaths_simulation = np.ediff1d(reported_deaths_simulation)

reported_cases_simulation = np.ediff1d(cumulative_cases_simulation)

simulation_dates = np.asarray([NYC_date_of_interest[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_cases_simulation))])

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot(NYC_date_of_interest[::3], cumulative_reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None', label = r'total cases (NYC)')
ax.plot(simulation_dates+dt.timedelta(days = 17), cumulative_cases_simulation, 'k', label = 'total cases (simulation)')

ax2.plot(NYC_date_of_interest[::3], cumulative_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None', label = r'total deaths (NYC)')
ax2.plot(simulation_dates+dt.timedelta(days = 17), cumulative_deaths_simulation, 'darkred', label = 'total deaths (simulation)')

#ax.text(dt.date(2020, 3, 10), 0.13, 'no SD')
#ax.text(dt.date(2020, 4, 19), 0.03, 'SD intervention')
#ax.text(dt.date(2020, 6, 21), 0.13, 'loosening SD')
#
#
#ax.fill_between([dt.date(2020, 3, 1), dt.date(2020, 3, 26)], 1, 0, color = 'Salmon', alpha = 0.2)
#ax.fill_between([dt.date(2020, 3, 26), dt.date(2020, 6, 15)], 1, 0, color = 'orange', alpha = 0.2)
#ax.fill_between([dt.date(2020, 6, 15), dt.date(2020, 7, 26)], 1, 0, color = 'Salmon', alpha = 0.2)

ax.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 7, 26)])
ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(0,1)
ax.set_ylabel(r'proportion of total cases', labelpad = 3)

ax2.set_ylim(0,1200)
ax2.set_ylabel(r'total deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

ax.legend(frameon = True, loc = 1, fontsize = 7)
ax2.legend(frameon = True, loc = 4, fontsize = 7)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('new_york_cases_nointervention.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.plot(NYC_date_of_interest[::3], reported_cases_NYC[::3], marker = 'o', markersize = 3, color = 'k', ls = 'None', label = r'cases (NYC)')
ax.plot(simulation_dates[:-1]+dt.timedelta(days = 17), reported_cases_simulation, 'k', label = 'cases (simulation)')

ax2.plot(NYC_date_of_interest[::3], reported_deaths_NYC[::3], marker = 's', markersize = 4, color = 'darkred', markeredgecolor = 'Grey', ls = 'None', label = r'deaths (NYC)')
ax2.plot(simulation_dates[:-1]+dt.timedelta(days = 17), reported_deaths_simulation, 'darkred', label = 'deaths (simulation)')

#ax.text(dt.date(2020, 3, 10), 0.13, 'no SD')
#ax.text(dt.date(2020, 4, 19), 0.03, 'SD intervention')
#ax.text(dt.date(2020, 6, 21), 0.13, 'loosening SD')
#
#
#ax.fill_between([dt.date(2020, 3, 1), dt.date(2020, 3, 26)], 1, 0, color = 'Salmon', alpha = 0.2)
#ax.fill_between([dt.date(2020, 3, 26), dt.date(2020, 6, 15)], 1, 0, color = 'orange', alpha = 0.2)
#ax.fill_between([dt.date(2020, 6, 15), dt.date(2020, 7, 26)], 1, 0, color = 'Salmon', alpha = 0.2)

ax.set_xlim([dt.date(2020, 3, 8), dt.date(2020, 7, 26)])
ax.set_xticklabels(NYC_date_of_interest[::7], rotation = 45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

ax.set_ylim(0,0.05)
ax.set_ylabel(r'proportion of cases', labelpad = 3)

ax2.set_ylim(0,100)
ax2.set_ylabel(r'deaths/100,000', color = 'darkred')
ax2.tick_params(axis='y', labelcolor = 'darkred')   

ax.legend(frameon = True, loc = 1, fontsize = 7)
ax2.legend(frameon = True, loc = 4, fontsize = 7)

plt.tight_layout()
plt.margins(0,0)
plt.savefig('new_york_cases2_nointervention.png', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)
