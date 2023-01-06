import os, sys; sys.path.append(os.path.join(".."))

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from epiforecast.contact_network import ContactNetwork

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks")

from matplotlib import rcParams

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Helvetica',
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
fig_size = [1.5*fig_width, 2*fig_height]
rcParams.update({'figure.figsize': fig_size})

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

#%% load network data to get the population size
NETWORKS_PATH = os.path.join('..', 'data', 'networks')
edges_filename = os.path.join(NETWORKS_PATH, 'edge_list_SBM_1e5_nobeds.txt')
groups_filename = os.path.join(NETWORKS_PATH, 'node_groups_SBM_1e5_nobeds.json')

network = ContactNetwork.from_files(edges_filename, groups_filename)
population = len(network.graph)
#%%

#%% load NYC data
NYC_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'data_new_york2.csv'))
NYC_cases = np.asarray([float(x) for x in NYC_data['Cases'].tolist()])
NYC_date_of_interest_cases = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_data['DATE_OF_INTEREST'].tolist()])

NYC_death_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'death_data_new_york2.csv'))
#NYC_data_date_of_interest_deaths = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_death_data.columns.values[1:]])
#NYC_death_data = np.asarray(NYC_death_data.iloc[-1].tolist()[1:])+np.asarray(NYC_death_data.iloc[-2].tolist()[1:])
NYC_data_date_of_interest_deaths = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_death_data['DATE_OF_DEATH'].tolist()])
NYC_death_data = np.asarray(NYC_death_data['Total'].tolist())

# population of NYC
NYC_population = 8.399e6

# fraction reported cases
fraction_reported = 1

# cumulative cases NYC
reported_cases_NYC = 1/fraction_reported*NYC_cases/NYC_population*1e5
reported_deaths_NYC = NYC_death_data/NYC_population*1e5
cumulative_reported_cases_NYC = np.cumsum(reported_cases_NYC)
cumulative_deaths_NYC = np.cumsum(reported_deaths_NYC)

NYC_death_data_weekly = np.mean(np.append(reported_deaths_NYC, (7-len(reported_deaths_NYC)%7)*[reported_deaths_NYC[-1]]).reshape(-1, 7), axis=1)

NYC_cases_weekly = np.mean(np.append(reported_cases_NYC, (7-len(reported_cases_NYC)%7)*[reported_cases_NYC[-1]]).reshape(-1, 7), axis=1)


#%% plot definitions
fig, axs = plt.subplots(nrows = 2, ncols = 2)

# initial simulation data shift
dd = 6 # days

# cumulative death panel
ax00 = axs[0][0]

ax00_2 = axs[0][0].twinx()

ax00.set_title(r'Deaths per 100,000')

#ax00.text(dt.date(2020, 2, 26), 0.9*800, r'(a)')

ax00.bar(NYC_data_date_of_interest_deaths, cumulative_deaths_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.0001)
    
#ax00.text(dt.date(2020, 4, 29), 75, r'data')
#ax00.text(dt.date(2020, 6, 14), 350, r'model')
#ax00.text(dt.date(2020, 4, 28), 650, r'no SD')

ax00.set_ylabel("Cumulative")

ax00.set_ylim(0,800)
ax00.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax00.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax00.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax00_2.set_yticks([])

ax00.yaxis.grid(zorder=0)

# cumulative infection panel
ax01 = axs[0][1]
ax01.set_title(r'Infections per 100,000')

ax01.yaxis.grid(zorder=0)
ax01_2 = axs[0][1].twinx()
ax01_2.yaxis.grid(zorder=0)
#ax01.text(dt.date(2020, 2, 26), 0.9*60000, r'(b)')

ax01_2.bar(NYC_date_of_interest_cases, cumulative_reported_cases_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.00001, align = 'center', zorder = 10)
    
#ax01.text(dt.date(2020, 4, 29), 6000, r'data')
#ax01.text(dt.date(2020, 6, 11), 31000, r'model')
#ax01.text(dt.date(2020, 4, 10), 52000, r'no SD')
                           
ax01.set_ylim(0,80000)
ax01.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax01.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax01.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax01_2.set_ylim(0,8000)
ax01_2.set_yticks([0,2000,4000,6000,8000])
ax01_2.tick_params(axis='y', colors='indianred')
ax01_2.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
ax01.yaxis.grid(zorder=0)
#ax01_2.grid("off")

# death cases panel
ax10 = axs[1][0]

ax10_2 = axs[1][0].twinx()
ax10_2.yaxis.grid(zorder=0)
#ax10.text(dt.date(2020, 2, 26), 0.9*25, r'(c)')

ax10.fill_between(NYC_data_date_of_interest_deaths[::7]+dt.timedelta(days = 3.5), NYC_death_data_weekly, edgecolor = '#ED7B64', facecolor = '#ED7B64', alpha = 0.2, linewidth = 1.)
ax10.plot(NYC_data_date_of_interest_deaths[::7]+dt.timedelta(days = 3.5), NYC_death_data_weekly,  color = '#ED7B64',  linewidth = 1.)

ax10.bar(NYC_data_date_of_interest_deaths, reported_deaths_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.0001)

#ax10.text(dt.date(2020, 4, 7), 9.5, r'new deaths', fontsize = 7)
#ax10.text(dt.date(2020, 5, 18), 8, r'7-day average', fontsize = 7)
#ax10.plot([dt.date(2020, 5, 16), dt.date(2020, 4, 28)], [8.0, 3.2], color = 'k', linewidth = 0.5)
                 
ax10.set_ylabel("Daily")

ax10.set_ylim(0,25)
#ax10.set_yticks([0,10,20])
ax10.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax10.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax10.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax10.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax10_2.set_yticks([])

ax10.yaxis.grid()

# cases panel
ax11 = axs[1][1]

ax11_2 = axs[1][1].twinx()

#ax11.text(dt.date(2020, 2, 26), 0.9*2000, r'(d)')

ax11_2.fill_between(NYC_date_of_interest_cases[::7]+dt.timedelta(days = 3.5), NYC_cases_weekly, edgecolor = '#ED7B64', facecolor = '#ED7B64', alpha = 0.2, linewidth = 1.)
ax11_2.plot(NYC_date_of_interest_cases[::7]+dt.timedelta(days = 3.5), NYC_cases_weekly,  color = '#ED7B64',  linewidth = 1.)

ax11_2.bar(NYC_date_of_interest_cases, reported_cases_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.0001)

#ax11.text(dt.date(2020, 3, 29), 800, r'new cases', fontsize = 7)
#ax11.text(dt.date(2020, 5, 17), 400, r'7-day average', fontsize = 7)
#ax11.plot([dt.date(2020, 5, 16), dt.date(2020, 4, 28)], [420, 240], color = 'k', linewidth = 0.5)

ax11.set_ylim(0,3000)
ax11.set_yticks([0,1000,2000,3000])
ax11.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax11.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax11.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax11.get_yaxis().set_major_formatter(
ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax11_2.set_ylim(0,3000*6000/60000)
ax11_2.set_yticks([0,100,200,300])
ax11_2.tick_params(axis='y', colors='indianred')
ax11_2.get_yaxis().set_major_formatter(
ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax11.yaxis.grid(zorder=0)

cumulative_deaths_simulation_mean_arr = []
cumulative_deaths_simulation_nointervention_mean_arr = []
cumulative_cases_simulation_mean_arr = []
cumulative_cases_simulation_nointervention_mean_arr = []
deaths_simulation_mean_arr = []
deaths_simulation_nointervention_mean_arr = []
cases_simulation_mean_arr = []
cases_simulation_nointervention_mean_arr = []

infection_mean_arr = []

for i in range(20):
    #%% load simulation data (with and without interventions)
    simulation_data = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'NYC_interventions_1e5_%d.txt'%i))
    simulation_data_nointervention = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'NYC_nointerventions_1e5_%d.txt'%i))
    
    times = simulation_data[:,0]
    times_nointervention = simulation_data_nointervention[:,0]
    
    kinetic_model = {'S': simulation_data[:,1], 'E': simulation_data[:,2], 'I': simulation_data[:,3], 'H': simulation_data[:,4], 'R': simulation_data[:,5], 'D': simulation_data[:,6]}
    kinetic_model_nointervention = {'S': simulation_data_nointervention[:,1], 'E': simulation_data_nointervention[:,2], 'I': simulation_data_nointervention[:,3], 'H': simulation_data_nointervention[:,4], 'R': simulation_data_nointervention[:,5], 'D': simulation_data_nointervention[:,6]}
    
    #plt.figure()
    #plt.plot(times,simulation_data[:,3])
    #plt.plot(times_nointervention,simulation_data_nointervention[:,3])
    #plt.xlabel(r"$t$")
    #plt.ylabel(r"$I(t)$")
    #plt.show()
    
    #%% determine averages of simulation data
    # daily averages of simulation data
    # sampling_time = 1 means that we average over 1-day intervals
    sampling_time = 7
    daily_average = simulation_average(kinetic_model, times, sampling_time)
    infections_simulation = np.asarray(daily_average['I'])/population*100
    cumulative_cases_simulation = (1-np.asarray(daily_average['S'])/population)*1e5
    daily_average = simulation_average(kinetic_model, times, sampling_time)
    cumulative_deaths_simulation = np.asarray(daily_average['D'])/population*1e5
    
    daily_average_nointervention = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time)
    cumulative_cases_simulation_nointervention = (1-np.asarray(daily_average_nointervention['S'])/population)*1e5
    cumulative_deaths_simulation_nointervention = np.asarray(daily_average_nointervention['D'])/population*1e5

    
    cases_simulation = np.ediff1d(cumulative_cases_simulation)/sampling_time
    cases_simulation_nointervention = np.ediff1d(cumulative_cases_simulation_nointervention)/sampling_time
    
    simulation_dates = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_cases_simulation))])
    simulation_dates_nointervention = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time) for i in range(len(cumulative_deaths_simulation_nointervention))])
    
    sampling_time2 = 7
    daily_average2 = simulation_average(kinetic_model, times, sampling_time2)
    cumulative_cases_simulation2 = (1-np.asarray(daily_average2['S'])/population)*1e5
    daily_average2 = simulation_average(kinetic_model, times, sampling_time2)
    cumulative_deaths_simulation2 = np.asarray(daily_average2['D'])/population*1e5
    daily_average_nointervention2 = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time2)
    cumulative_cases_simulation_nointervention2 = (1-np.asarray(daily_average_nointervention2['S'])/population)*1e5
    cumulative_deaths_simulation_nointervention2 = np.asarray(daily_average_nointervention2['D'])/population*1e5
    
    deaths_simulation = np.ediff1d(cumulative_deaths_simulation2)/sampling_time2
    deaths_simulation_nointervention = np.ediff1d(cumulative_deaths_simulation_nointervention2)/sampling_time2
    
    simulation_dates2 = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time2) for i in range(len(deaths_simulation))])
    simulation_dates_nointervention2 = np.asarray([NYC_date_of_interest_cases[0] + i*dt.timedelta(days=sampling_time2) for i in range(len(deaths_simulation_nointervention))])
    
    #%% plot results
    
    ax00.plot(simulation_dates+dt.timedelta(days = dd), cumulative_deaths_simulation, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax00.plot(simulation_dates_nointervention+dt.timedelta(days = dd), cumulative_deaths_simulation_nointervention, '#EDBF64', linewidth = 0.8, alpha = 0.2)
    
    ax01.plot(simulation_dates+dt.timedelta(days = dd), cumulative_cases_simulation, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax01.plot(simulation_dates_nointervention+dt.timedelta(days = dd), cumulative_cases_simulation_nointervention, '#EDBF64', linewidth = 0.8, alpha = 0.2)
    
    ax10.plot(simulation_dates2+dt.timedelta(days = dd), deaths_simulation, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax10.plot(simulation_dates_nointervention2+dt.timedelta(days = dd), deaths_simulation_nointervention, '#EDBF64', linewidth = 0.8, alpha = 0.2)
    
    ax11.plot(simulation_dates2+dt.timedelta(days = dd), cases_simulation, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax11.plot(simulation_dates_nointervention2+dt.timedelta(days = dd), cases_simulation_nointervention, '#EDBF64', linewidth = 0.8, alpha = 0.2)

    cumulative_deaths_simulation_mean_arr.append(cumulative_deaths_simulation)
    cumulative_deaths_simulation_nointervention_mean_arr.append(cumulative_deaths_simulation_nointervention)
    cumulative_cases_simulation_mean_arr.append(cumulative_cases_simulation)
    cumulative_cases_simulation_nointervention_mean_arr.append(cumulative_cases_simulation_nointervention)
    deaths_simulation_mean_arr.append(deaths_simulation)
    deaths_simulation_nointervention_mean_arr.append(deaths_simulation_nointervention)
    cases_simulation_mean_arr.append(cases_simulation)
    cases_simulation_nointervention_mean_arr.append(cases_simulation_nointervention)

    infection_mean_arr.append(infections_simulation)
    
ax00.plot(simulation_dates+dt.timedelta(days = dd), np.mean(cumulative_deaths_simulation_mean_arr, axis = 0), 'cornflowerblue', linewidth = 1.5, zorder = 100)
ax00.plot(simulation_dates_nointervention+dt.timedelta(days = dd), np.mean(cumulative_deaths_simulation_nointervention_mean_arr, axis = 0), '#EDBF64', linewidth = 1.5, zorder = 100)

ax01.plot(simulation_dates+dt.timedelta(days = dd), np.mean(cumulative_cases_simulation_mean_arr, axis = 0), 'cornflowerblue', linewidth = 1.5, zorder = 100)
ax01.plot(simulation_dates_nointervention+dt.timedelta(days = dd), np.mean(cumulative_cases_simulation_nointervention_mean_arr, axis = 0), '#EDBF64', linewidth = 1.5, zorder = 100)

ax10.plot(simulation_dates2+dt.timedelta(days = dd), np.mean(deaths_simulation_mean_arr, axis = 0), 'cornflowerblue', linewidth = 1.5, zorder = 100)
ax10.plot(simulation_dates_nointervention2+dt.timedelta(days = dd), np.mean(deaths_simulation_nointervention_mean_arr, axis = 0), '#EDBF64', linewidth = 1.5, zorder = 100)

ax11.plot(simulation_dates2+dt.timedelta(days = dd), np.mean(cases_simulation_mean_arr, axis = 0), 'cornflowerblue', linewidth = 1.5, zorder = 100)
ax11.plot(simulation_dates_nointervention2+dt.timedelta(days = dd), np.mean(cases_simulation_nointervention_mean_arr, axis = 0), '#EDBF64', linewidth = 1.5, zorder = 100)

ax01.set_zorder(1)  
ax01.patch.set_visible(False)  
ax11.set_zorder(1)  
ax11.patch.set_visible(False)

plt.tight_layout()
plt.margins(0,0)
sns.despine(top=True, right=True, left=True)
plt.savefig('new_york_cases.pdf', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)

plt.figure()

plt.plot(simulation_dates+dt.timedelta(days = dd), np.mean(infection_mean_arr, axis = 0))

plt.show()

# set nice figure sizes
fig_width_pt = 368    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.75*fig_width*ratio  # height in inches
fig_size = [1.5*fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

# hospitalization plot
#%%

NYC_hosp_data = pd.read_csv(os.path.join('..', 'data', 'NYC_COVID_CASES', 'hospitalization_data_new_york.csv'))
NYC_data_date_of_interest_hosp = np.asarray([dt.datetime.strptime(x, "%m/%d/%Y") for x in NYC_hosp_data['DATE_OF_INTEREST'].tolist()])
NYC_hosp_data = np.asarray(NYC_hosp_data['Hospitalizations'].tolist())

reported_hosp_NYC = NYC_hosp_data/NYC_population*1e5
cumulative_hosp_NYC = np.cumsum(reported_hosp_NYC)

fig, axs = plt.subplots(ncols = 2)

# cumulative death panel
ax00 = axs[0]
ax00_2 = axs[0].twinx()

ax00.set_zorder(ax00_2.get_zorder()+1)
ax00.patch.set_visible(False)

ax00.set_ylabel("Cumulative")

#ax00.text(dt.date(2020, 3, 12), 0.9*1500, r'(a)')

#ax00.text(dt.date(2020, 5, 10), 1350, r'no SD')

ax00.set_ylim(0,1500)
ax00.set_yticks([0,500,1000,1500])
ax00.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax00.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax00.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax00.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax00.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax00_2.set_yticks([])

ax00.yaxis.grid(zorder=0)

# cumulative death panel
ax01 = axs[1]

ax01_2 = axs[1].twinx()

ax01.set_zorder(ax01_2.get_zorder()+1)
ax01.patch.set_visible(False)

ax01.set_ylabel("Daily")

#ax01.text(dt.date(2020, 3, 12), 0.9*40, r'(b)')

#ax01.text(dt.date(2020, 5, 2), 35, r'no SD')

ax01.set_ylim(0,50)
ax01.set_xlim([dt.date(2020, 3, 5), dt.date(2020, 7, 9)])
ax01.set_xticklabels(NYC_date_of_interest_cases[::14], rotation = 0)
ax01.xaxis.set_major_locator(ticker.MultipleLocator(14))
ax01.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax01.get_yaxis().set_major_formatter(
    ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax01_2.set_yticks([])

ax01.yaxis.grid(zorder=0)

fig.suptitle(r'Hospitalizations per 100,000', y = 0.92, fontsize=10)

hospitalizations_simulation_mean_arr = []
hospitalizations_simulation_nointervention_mean_arr = []
hospitalizations_simulation_daily_mean_arr = []
hospitalizations_simulation_nointervention_daily_mean_arr = []

for i in range(20):
    #%% load simulation data (with and without interventions)
    simulation_data = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'NYC_interventions_1e5_%d.txt'%i))
    simulation_data_nointervention = np.loadtxt(os.path.join('..', 'data', 'simulation_data', 'NYC_nointerventions_1e5_%d.txt'%i))
    
    times = simulation_data[:,0]
    times_nointervention = simulation_data_nointervention[:,0]
    
    kinetic_model = {'S': simulation_data[:,1], 'E': simulation_data[:,2], 'I': simulation_data[:,3], 'H': simulation_data[:,4], 'R': simulation_data[:,5], 'D': simulation_data[:,6]}
    kinetic_model_nointervention = {'S': simulation_data_nointervention[:,1], 'E': simulation_data_nointervention[:,2], 'I': simulation_data_nointervention[:,3], 'H': simulation_data_nointervention[:,4], 'R': simulation_data_nointervention[:,5], 'D': simulation_data_nointervention[:,6]}
    
    #%% determine averages of simulation data
    # daily averages of simulation data
    # sampling_time = 1 means that we average over 1-day intervals

    sampling_time = 7
    daily_average = simulation_average(kinetic_model, times, sampling_time)
    hospitalizations_simulation = np.cumsum(np.asarray(daily_average['H']))/population*1e5
    
    hospitalizations_simulation_daily = np.ediff1d(hospitalizations_simulation)/sampling_time
    
    daily_average_nointervention = simulation_average(kinetic_model_nointervention, times_nointervention, sampling_time)
    hospitalizations_simulation_nointervention = np.cumsum(np.asarray(daily_average_nointervention['H']))/population*1e5
    
    hospitalizations_simulation_nointervention_daily = np.ediff1d(hospitalizations_simulation_nointervention)/sampling_time
    
    ax00.plot(simulation_dates+dt.timedelta(days = dd), hospitalizations_simulation, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax00.plot(simulation_dates_nointervention+dt.timedelta(days = dd), hospitalizations_simulation_nointervention, '#EDBF64', linewidth = 0.8, alpha = 0.2)
    
    ax01.plot(simulation_dates[:-1]+dt.timedelta(days = dd), hospitalizations_simulation_daily, 'cornflowerblue', linewidth = 0.8, alpha = 0.2)
    ax01.plot(simulation_dates_nointervention[:-1]+dt.timedelta(days = dd), hospitalizations_simulation_nointervention_daily, '#EDBF64', linewidth = 0.8, alpha = 0.2)

    hospitalizations_simulation_mean_arr.append(hospitalizations_simulation)
    hospitalizations_simulation_nointervention_mean_arr.append(hospitalizations_simulation_nointervention)
    hospitalizations_simulation_daily_mean_arr.append(hospitalizations_simulation_daily)
    hospitalizations_simulation_nointervention_daily_mean_arr.append(hospitalizations_simulation_nointervention_daily)

ax00.plot(simulation_dates+dt.timedelta(days = dd), np.mean(hospitalizations_simulation_mean_arr, axis = 0), 'cornflowerblue')
ax00.plot(simulation_dates_nointervention+dt.timedelta(days = dd), np.mean(hospitalizations_simulation_nointervention_mean_arr, axis = 0), '#EDBF64')

ax01.plot(simulation_dates[:-1]+dt.timedelta(days = dd), np.mean(hospitalizations_simulation_daily_mean_arr, axis = 0), 'cornflowerblue')
ax01.plot(simulation_dates_nointervention[:-1]+dt.timedelta(days = dd), np.mean(hospitalizations_simulation_nointervention_daily_mean_arr, axis = 0), '#EDBF64')

ax00.bar(NYC_data_date_of_interest_hosp, cumulative_hosp_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.0001)
ax01.bar(NYC_data_date_of_interest_hosp, reported_hosp_NYC, facecolor='#ED7B64', edgecolor='#ED7B64', alpha = 1, width = 0.0001)

fig.tight_layout()
plt.margins(0,0)
sns.despine(top=True, right=True, left=True)
plt.savefig('new_york_hospitalizations.pdf', dpi=300, bbox_inches = 'tight',
    pad_inches = 0.05)
