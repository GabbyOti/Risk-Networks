import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


COLORS_OF_STATUSES = {
        'S': 'C0',
        'E': 'C3',
        'I': 'C1',
        'H': 'C2',
        'R': 'C4',
        'D': 'C6'
}


def plot_master_eqns(
        states,
        t,
        axes=None,
        xlims=None,
        leave=False,
        figsize=(15, 4),
        **kwargs):

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)
    N_eqns = 5
    S, I, H, R, D = np.arange(N_eqns)

    for mm in tqdm(range(states.shape[0]), desc = 'Plotting ODE', leave = leave):

        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[S], color = 'C0', linestyle = '--', linewidth = 2)
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[R], color = 'C4', linestyle = '--')
        axes[0].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'C1', linestyle = '--')

        axes[1].plot(t, (1 - states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 0)).sum(axis = 0), color = 'C3', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[I], color = 'C1', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[H], color = 'C2', linestyle = '--')
        axes[1].plot(t, states[mm].reshape(N_eqns, -1, len(t)).sum(axis = 1)[D], color = 'C6', linestyle = '--')

    axes[0].legend(['Susceptible', 'Resistant', 'Infected'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Exposed', 'Infected', 'Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.);

    for kk, ax in enumerate(axes):
        ax.set_xlim(xlims)

    lg = axes[0].get_legend()
    hl = {handle.get_label(): handle for handle in lg.legendHandles}
    hl['_line0'].set_color('C0')
    hl['_line1'].set_color('C4')
    hl['_line2'].set_color('C1')

    lg = axes[1].get_legend()
    hl = {handle.get_label(): handle for handle in lg.legendHandles}
    hl['_line0'].set_color('C3')
    hl['_line1'].set_color('C1')
    hl['_line2'].set_color('C2')
    hl['_line3'].set_color('C6')

    plt.tight_layout()

    return axes

#works in joint epidemic assimilation
def plot_ensemble_states(
        user_population,
        population,
        states_sum,
        t,
        axes=None,
        xlims=None,
        leave=False,
        figsize=(15, 4),
        a_min=None,
        a_max=None):

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = figsize)

    ensemble_size = states_sum.shape[0]
    N_eqns = 6
    statuses = np.arange(N_eqns)
    #user_population = int(states.shape[1]/N_eqns)
    #states_sum  = (states.reshape(ensemble_size, N_eqns, -1, len(t)).sum(axis = 2))/population
    states_perc = np.percentile(states_sum, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    if N_eqns == 6:
        statuses_colors = ['C0', 'C3', 'C1', 'C2', 'C4', 'C6']
        for status in statuses:
            if status in [0,1,4]:
                axes[0].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].plot(t, states_perc[3,status], color = statuses_colors[status])
                
            if status in [2]:
                axes[1].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].plot(t, states_perc[3,status], color = statuses_colors[status])
                
            if status in [3, 5]:
                axes[2].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].plot(t, states_perc[3,status], color = statuses_colors[status])
    
    if N_eqns == 5:
        statuses_colors = ['C3', 'C1', 'C2', 'C4', 'C6']
        for status in statuses:
            if status in [0,3]:
                axes[0].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[0].plot(t, states_perc[3,status], color = statuses_colors[status])
                
            if status in [1]:
                axes[1].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[1].plot(t, states_perc[3,status], color = statuses_colors[status])
                
            if status in [2, 4]:
                axes[2].fill_between(t, np.clip(states_perc[0,status], a_min, a_max), np.clip(states_perc[-1,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].fill_between(t, np.clip(states_perc[1,status], a_min, a_max), np.clip(states_perc[-2,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].fill_between(t, np.clip(states_perc[2,status], a_min, a_max), np.clip(states_perc[-3,status], a_min, a_max), alpha = .2, color = statuses_colors[status], linewidth = 0.)
                axes[2].plot(t, states_perc[3,status], color = statuses_colors[status])

        residual_state = 1.0 - states_sum.sum(axis = 1)
        residual_state = np.percentile(residual_state, q = [1, 10, 25, 50, 75, 90, 99], axis = 0)
        axes[0].fill_between(t, np.clip(residual_state[0], a_min, a_max), np.clip(residual_state[-1], a_min, a_max), alpha = .2, color = 'C0', linewidth = 0.)
        axes[0].fill_between(t, np.clip(residual_state[1], a_min, a_max), np.clip(residual_state[-2], a_min, a_max), alpha = .2, color = 'C0', linewidth = 0.)
        axes[0].fill_between(t, np.clip(residual_state[2], a_min, a_max), np.clip(residual_state[-3], a_min, a_max), alpha = .2, color = 'C0', linewidth = 0.)
        axes[0].plot(t, np.clip(residual_state[3], a_min, a_max), color = 'C0')
        
    axes[0].legend(['Susceptible','Exposed','Resistant'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=3, mode="expand", borderaxespad=0.);
    axes[1].legend(['Infected'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.);
    axes[2].legend(['Hospitalized', 'Death'],
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.);

    for kk, ax in enumerate(axes):
        ax.set_xlim(xlims)

    plt.tight_layout()

    return axes

def plot_epidemic_data(
        population,
        statuses_list,
        axes,
        plot_times):
    """
    Plot cumulative kinetic model states vs time

    Input:
        population (int): total population, used to normalize states to [0,1]
        statuses_list (list): timeseries of cumulative states; each element is a
                              6-tuple: (n_S, n_E, n_I, n_H, n_R, n_D)
        axes (np.array or list): (3,) array for plotting (S,E,R), (I) and (H,D)
        plot_times (np.array): (len(statuses_list),) array of times to plot
                               against
    Output:
        None
    """
    global COLORS_OF_STATUSES

    assert len(statuses_list) == len(plot_times)

    Sdata = [statuses_list[i][0]/population for i in range(len(plot_times))]
    Edata = [statuses_list[i][1]/population for i in range(len(plot_times))]
    Idata = [statuses_list[i][2]/population for i in range(len(plot_times))]
    Hdata = [statuses_list[i][3]/population for i in range(len(plot_times))]
    Rdata = [statuses_list[i][4]/population for i in range(len(plot_times))]
    Ddata = [statuses_list[i][5]/population for i in range(len(plot_times))]
    
    axes[0].scatter(plot_times, Sdata, c=COLORS_OF_STATUSES['S'], marker='x')
    axes[0].scatter(plot_times, Edata, c=COLORS_OF_STATUSES['E'], marker='x')
    axes[0].scatter(plot_times, Rdata, c=COLORS_OF_STATUSES['R'], marker='x')

    axes[1].scatter(plot_times, Idata, c=COLORS_OF_STATUSES['I'], marker='x')

    axes[2].scatter(plot_times, Hdata, c=COLORS_OF_STATUSES['H'], marker='x')
    axes[2].scatter(plot_times, Ddata, c=COLORS_OF_STATUSES['D'], marker='x')

    return axes

def plot_ensemble_transmission_latent_fraction(
        community_transmission_rate_trace,
        latent_periods_trace, time_horizon):

    transmission_perc = np.percentile(community_transmission_rate_trace, q = [1, 25, 50, 75, 99], axis = 0)
    latent_periods_perc = np.percentile(latent_periods_trace, q = [1, 25, 50, 75, 99], axis = 0)

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    axes[0].fill_between(time_horizon, transmission_perc[0], transmission_perc[-1], alpha = .2, color = 'C0')
    axes[0].fill_between(time_horizon, transmission_perc[1], transmission_perc[-2], alpha = .2, color = 'C0')
    axes[0].plot(time_horizon, transmission_perc[2])
    axes[0].set_title(r'Transmission rate: $\beta$');

    axes[1].fill_between(time_horizon, latent_periods_perc[0], latent_periods_perc[-1], alpha = .2, color = 'C0')
    axes[1].fill_between(time_horizon, latent_periods_perc[1], latent_periods_perc[-2], alpha = .2, color = 'C0')
    axes[1].plot(time_horizon, latent_periods_perc[2])
    axes[1].set_title(r'Latent period: $\gamma$');

    return axes

def plot_scalar_parameters(parameters, time_horizon, names):
    percentiles = {}
    fig, axes = plt.subplots(1, len(parameters), figsize = (4 * len(parameters), 4))

    for kk, parameter in enumerate(names):
        percentiles[parameter] = np.percentile(parameters[kk], q = [1, 25, 50, 75, 99], axis = 0)

        axes[kk].fill_between(time_horizon, percentiles[parameter][0], percentiles[parameter][-1], alpha = .2, color = 'C0')
        axes[kk].fill_between(time_horizon, percentiles[parameter][1], percentiles[parameter][-2], alpha = .2, color = 'C0')
        axes[kk].plot(time_horizon, percentiles[parameter][2])
        axes[kk].set_title(names[kk]);

    return axes


def plot_roc_curve(true_negative_rates,
                   true_positive_rates,
                   labels = None,
                   show = True,
                   fig_size=(10, 5)):
    """
    Plots an ROC (Receiver Operating Characteristics) curve. This requires many experiments to
    as each experiments will produce one TNR, TPR pair.
    The x-axis is the False Positive Rate = 1 - TNR = 1 - TN / (TN + FP) 
    The y-axis is the True Positive Rate = TPR = TP / (TP + FN) 

    One can obtain these quantities through the PerformanceMetrics object
    
    Args
    ----
    true_negative_rates(np.array): array of true_negative_rates
    true_positive_rates(np.array): array of true_positive_rates of the same dimensions
    show                   (bool): bool to display graph
    labels                 (list): list of labels for the line plots
    """
    if true_negative_rates.ndim == 1:
        fpr = 1 -  np.array([true_negative_rates])
    else:
        fpr = 1 - true_negative_rates
        
    if true_positive_rates.ndim == 1:
        tpr = np.array([true_positive_rates])
    else:
        tpr = true_positive_rates

    # fpr,tpr size num_line_plots x num_samples_per_plot 
    colors = ['C'+str(i) for i in range(tpr.shape[0])]

    if labels is None:
        labels = ['ROC_' + str(i) for i in range(tpr.shape[0])]
        
    fig, ax = plt.subplots(figsize=fig_size)
    for xrate,yrate,clr,lbl in zip(fpr,tpr,colors,labels):
        #plt.plot(xrate, yrate, color=clr, label=lbl , marker='|')
        plt.plot(xrate, yrate, color=clr, label=lbl)
            
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')

    if show:
        plt.show()
        
    return fig, ax


def plot_tpr_curve(predicted_positive_fraction,
                   true_positive_rates,
                   noda_flag=False,
                   test_and_isolate_flag=False,
                   tai_predicted_positive_fraction=None,
                   tai_true_positive_rates=None,
                   xmin=1e-3,
                   labels = None,
                   show = True,
                   fig_size=(10, 5)):
    """
    plots a curve of predicted_positive_fraction vs true_positive_rate
    as each experiments will produce one TNR, TPR pair.
    The x-axis is the Predicted Positive Fraction = PPF = (TP + FP) / Total 
    The y-axis is the True Positive Rate = TPR = TP / (TP + FN) 

    One can obtain these quantities through the PerformanceMetrics object
    
    Args
    ----
    predicted_positive_fraction     (np.array): array of predicted_positive_fraction
    true_positive_rates             (np.array): array of true_positive_rates of the same dimensions
    noda_flag                           (bool): bool for adding "prior" tprs to graph
    test_and_isolate_flag               (bool): bool for adding "test and isolate tprs to graph
    tai_predicted_positive_fraction (np.array): data for "test and isolate"
    tai_true_positive_rates         (np.array): data for "test and isolate"
   
    show                   (bool): bool to display graph
    labels                 (list): list of labels for the line plots
    """
    if predicted_positive_fraction.ndim == 1:
        ppf = np.array([predicted_positive_fraction])
    else:
        ppf = predicted_positive_fraction
        
    if true_positive_rates.ndim == 1:
        tpr = np.array([true_positive_rates])
    else:
        tpr = true_positive_rates
    
    #take out the prior
    if noda_flag:
        prior_ppf = ppf[0]
        prior_tpr = tpr[0]
        prior_label = labels[0]
        ppf = ppf[1:]
        tpr = tpr[1:]
        labels = labels[1:]
        

    # ppf,tpr size num_line_plots x num_samples_per_plot 
    #colors = ['C'+str(i) for i in range(tpr.shape[0])]
    #get colors from a color map
    colors = [plt.cm.OrRd(x) for x in np.linspace(0.2,1.0,tpr.shape[0])]

    if labels is None:
        labels = ['Curve_' + str(i) for i in range(tpr.shape[0])]
    
    # plot tpr curves in range
    fig, ax = plt.subplots(figsize=fig_size)
    for xrate,yrate,clr,lbl in list(zip(ppf,tpr,colors,labels))[::-1]:
        #first sort the lower bound with interpolation 
        # xrate,yrate are monotone DECREASING)
        idxabovemin = np.max(np.where(xrate>=xmin))
        xabovemin = xrate[idxabovemin]
        xbelowmin = xrate[idxabovemin+1]
        yabovemin = yrate[idxabovemin]
        ybelowmin = yrate[idxabovemin+1]
        yatmin = ybelowmin + (xmin - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)

        xplot = np.hstack((xrate[xrate>=xmin], xmin))
        yplot = np.hstack((yrate[xrate>=xmin], yatmin))
        # plt.plot(xrate, yrate, color=clr, label=lbl, marker='|')
        plt.plot(xplot, yplot, color=clr, label=lbl)
            
    #plot prior
    if noda_flag:
        idxabovemin = np.max(np.where(prior_ppf>=xmin))
        xabovemin = prior_ppf[idxabovemin]
        xbelowmin = prior_ppf[idxabovemin+1]
        yabovemin = prior_tpr[idxabovemin]
        ybelowmin = prior_tpr[idxabovemin+1]
        yatmin = ybelowmin + (xmin - xbelowmin) / (xabovemin - xbelowmin) * (yabovemin - ybelowmin)

        xplot = np.hstack((prior_ppf[prior_ppf>=xmin], xmin))
        yplot = np.hstack((prior_tpr[prior_ppf>=xmin], yatmin))
        # plt.plot(prior_ppf, prior_tpr, color=clr, label=lbl, marker='|')
        plt.plot(xplot, yplot, color="black", label=prior_label, linestyle=':')
        

    #plot test_and_isolate curves
    if test_and_isolate_flag:
        for (xplot,yplot,clr) in zip(tai_predicted_positive_fraction, tai_true_positive_rates, colors[-len(tai_true_positive_rates):]):
            plt.scatter([xplot],[yplot], color=[clr], marker='X')

    #plot random case
    #plt.plot([1e-3, 1], [1e-3, 1], color='darkblue', linestyle='--')
    plt.plot(np.logspace(np.log10(xmin),0,num=100),np.logspace(np.log10(xmin),0,num=100),color='black', linestyle='--')
    ax.set_xscale('log')
    plt.xlabel('PPF')#Predicted Positive Fraction')
    plt.ylabel('TPR') #True Positive Rate')
    #plt.title('PPF vs TPR Curve')
    plt.legend(loc='upper left')# 'lower right'

    if show:
        plt.show()
        
    return fig, ax
    
def plot_transmission_rate(transmission_rate_timeseries,
                           t,
                           color='b',
                           a_min=None,
                           a_max=None,
                           output_path='.',
                           output_name='transmission_rate'):

    rate_perc = np.percentile(transmission_rate_timeseries, 
            q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    plt.fill_between(t, np.clip(rate_perc[0,0], a_min, a_max), np.clip(rate_perc[-1,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    plt.fill_between(t, np.clip(rate_perc[1,0], a_min, a_max), np.clip(rate_perc[-2,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    plt.fill_between(t, np.clip(rate_perc[2,0], a_min, a_max), np.clip(rate_perc[-3,0], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
    plt.plot(t, rate_perc[3,0], color = color)
    plt.xlabel('Time (days)')
    plt.ylabel('1/community_transmission_rate')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,output_name+'.png'))
    plt.close()

def plot_network_averaged_clinical_parameters(
        transition_rates_timeseries,
        t,
        nodal_ages,
        age_indep_rates_true = None,
        age_dep_rates_true = None,
        color='b',
        a_min=None,
        a_max=None,
        num_rates=6,
        num_ages=5,
        age_dep_rates=[3,4,5],
        output_path='.',
        output_name=''):
    
    rate_timeseries = transition_rates_timeseries
    rate_perc = np.percentile(rate_timeseries, 
            q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    ylabel_list = ['latent_periods',
            'community_infection_periods',
            'hospital_infection_periods',
            'hospitalization_fraction',
            'community_mortality_fraction',
            'hospital_mortality_fraction']
    age_indep_rates = [i for i in range(num_rates) if i not in age_dep_rates]

    for k in age_indep_rates:
        plt.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.plot(t, rate_perc[3,k], color = color)
        if age_indep_rates_true is not None:
            plt.axhline(age_indep_rates_true[ylabel_list[k]], color = 'black', ls = '--')
            
        plt.xlabel('Time (days)')
        plt.ylabel(ylabel_list[k])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+output_name+'.png'))
        plt.close()
    
    #where the network truth is an age dependent object
    
    for k in age_dep_rates:
        plt.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.plot(t, rate_perc[3,k], color = color)
            
        plt.xlabel('Time (days)')
        plt.ylabel(ylabel_list[k])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+output_name+'.png'))
        plt.close()
    
def plot_ensemble_averaged_clinical_parameters(
        transition_rates_timeseries,
        t,
        nodal_ages,
        age_indep_rates_true = None,
        age_dep_rates_true = None,
        color='b',
        a_min=None,
        a_max=None,
        num_rates=6,
        num_ages=5,
        age_dep_rates=[3,4,5],
        output_path='.',
        output_name='',
        font_size=18,
        font_family='sans-serif'):
    
    matplotlib.rcParams.update({'font.size': font_size, 'font.family': font_family})
    rate_timeseries = transition_rates_timeseries
    rate_perc = np.percentile(rate_timeseries, 
            q = [5, 10, 25, 50, 75, 90, 95], axis = 0)

    ylabel_list = ['latent_periods',
            'community_infection_periods',
            'hospital_infection_periods',
            'hospitalization_fraction',
            'community_mortality_fraction',
            'hospital_mortality_fraction']
    age_indep_rates = [i for i in range(num_rates) if i not in age_dep_rates]

    for k in age_indep_rates:
        plt.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.plot(t, rate_perc[3,k], color = color)
        if age_indep_rates_true is not None:
            plt.axhline(age_indep_rates_true[ylabel_list[k]], color = 'black', ls = '--')
            
        plt.xlabel('Time (days)')
        ylabel_names = ylabel_list[k].split('_')
        ylabel_names[0] = ylabel_names[0].capitalize()
        plt.ylabel(' '.join(ylabel_names))
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+output_name+'.pdf'))
        plt.close()
    
    #where the network truth is an age dependent object
    age_colors = ['C'+ str(i) for i in np.arange(num_ages)]

    for k in age_dep_rates:
        for age_cat in range(num_ages):
            age_cat_nodes = [node for (node,cat) in enumerate(nodal_ages) if cat == age_cat]
            
            rate_perc = np.percentile(transition_rates_timeseries[age_cat_nodes,:], 
                                      q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

            plt.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = age_colors[age_cat], linewidth = 0.)
            plt.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = age_colors[age_cat], linewidth = 0.)
            plt.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = age_colors[age_cat], linewidth = 0.)
            plt.plot(t, rate_perc[3,k], color = age_colors[age_cat])
            if age_dep_rates_true is not None:
                plt.axhline(age_dep_rates_true[ylabel_list[k]][age_cat], color = age_colors[age_cat], ls = '--')

        plt.xlabel('Time (days)')
        plt.ylabel(ylabel_list[k])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+output_name+'.pdf'))
        plt.close()



def plot_transition_rates(transition_rates_obj_timeseries,
        t,
        color='b',
        a_min=None,
        a_max=None,
        num_rates=6,
        output_path='.'):

    num_time = len(transition_rates_obj_timeseries)
    num_ensemble = len(transition_rates_obj_timeseries[0])

    rate_timeseries = np.zeros((num_ensemble, num_rates, num_time))
    for i in range(num_time):
        for j in range(num_ensemble):
            rate_timeseries[j,0,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('exposed_to_infected')) 
            rate_timeseries[j,1,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('infected_to_hospitalized')) 
            rate_timeseries[j,2,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('infected_to_resistant')) 
            rate_timeseries[j,3,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('hospitalized_to_resistant')) 
            rate_timeseries[j,4,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('infected_to_deceased')) 
            rate_timeseries[j,5,i] = np.mean(transition_rates_obj_timeseries[i][j].get_transition_rate('hospitalized_to_deceased')) 

    rate_perc = np.percentile(rate_timeseries, 
            q = [1, 10, 25, 50, 75, 90, 99], axis = 0)

    ylabel_list = ['exposed_to_infected',
        'infected_to_hospitalized',
        'infected_to_resistant',
        'hospitalized_to_resistant',
        'infected_to_deceased',
        'hospitalized_to_deceased']

    for k in range(num_rates):
        plt.fill_between(t, np.clip(rate_perc[0,k], a_min, a_max), np.clip(rate_perc[-1,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[1,k], a_min, a_max), np.clip(rate_perc[-2,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.fill_between(t, np.clip(rate_perc[2,k], a_min, a_max), np.clip(rate_perc[-3,k], a_min, a_max), alpha = .2, color = color, linewidth = 0.)
        plt.plot(t, rate_perc[3,k], color = color)
        plt.xlabel('Time (days)')
        plt.ylabel(ylabel_list[k])
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,ylabel_list[k]+'.png'))
        plt.close()


