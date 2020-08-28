'''
Stanley Bak
Python code for F-16 animation video output
'''

import math
import os

import numpy as np
from numpy import rad2deg

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aerobench.util import get_state_names, StateIndex

def get_script_path(filename=__file__):
    '''get the path this script'''
    return os.path.dirname(os.path.realpath(filename))

def init_plot():
    'initialize plotting style'

    matplotlib.use('TkAgg') # set backend

    parent = get_script_path()
    p = os.path.join(parent, 'bak_matplotlib.mlpstyle')

    plt.style.use(['bmh', p])

def plot_overhead(run_sim_result, waypoints=None, llc=None):
    '''altitude over time plot from run_f16_sum result object

    note: call plt.show() afterwards to have plot show up
    '''

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)

    full_states = res['states']

    if llc is not None:
        num_vars = len(get_state_names()) + llc.get_num_integrators()
        num_aircraft = full_states[0, :].size // num_vars
    else:
        num_vars = full_states[0, :].size
        num_aircraft = 1

    for i in range(num_aircraft):
        states = full_states[:, i*num_vars:(i+1)*num_vars]

        ys = states[:, StateIndex.POSN] # 9: n/s position (ft)
        xs = states[:, StateIndex.POSE] # 10: e/w position (ft)

        ax.plot(xs, ys, '-')

        label = 'Start' if i == 0 else None
        ax.plot([xs[0]], [ys[1]], 'k*', ms=8, label=label)

    if waypoints is not None:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]

        ax.plot(xs, ys, 'ro', label='Waypoints')

    ax.set_ylabel('North / South Position (ft)')
    ax.set_xlabel('East / West Position (ft)')
    
    ax.set_title('Overhead Plot')

    ax.axis('equal')

    ax.legend()

    plt.tight_layout()

def plot_attitude(run_sim_result, title='Attitude History', skip_yaw=True, figsize=(7, 5)):
    'plot a single variable over time'

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['states']

    indices = [StateIndex.PHI, StateIndex.THETA, StateIndex.PSI, StateIndex.P, StateIndex.Q, StateIndex.R]
    labels = ['Roll (Phi)', 'Pitch (Theta)', 'Yaw (Psi)', 'Roll Rate (P)', 'Pitch Rate (Q)', 'Yaw Rate (R)']
    colors = ['r-', 'g-', 'b-', 'r--', 'g--', 'b--']

    rad_to_deg_factor = 180 / math.pi

    for index, label, color in zip(indices, labels, colors):
        if skip_yaw and index == StateIndex.PSI:
            continue
        
        ys = states[:, index] # 11: altitude (ft)

        ax.plot(times, ys * rad_to_deg_factor, color, label=label)

    ax.set_ylabel('Attitudes & Rates (deg, deg/s)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

def plot_outer_loop(run_sim_result, title='Outer Loop Controls'):
    'plot a single variable over time'

    init_plot()

    res = run_sim_result
    assert 'u_list' in res, "Simulation must be run with extended_states=True"
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    u_list = res['u_list']
    ps_list = res['ps_list']
    nz_list = res['Nz_list']
    ny_r_list = res['Ny_r_list']

    # u is: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    # u_ref is: Nz, ps, Ny + r, throttle
    ys_list = []

    ys_list.append(nz_list)
    ys_list.append([u[4] for u in u_list])

    ys_list.append(ps_list)
    ys_list.append([u[5] for u in u_list])

    ys_list.append(ny_r_list)
    ys_list.append([u[6] for u in u_list])

    # throttle reference is not included... although it's just a small offset so probably less important
    ys_list.append([u[0] for u in u_list])

    labels = ['N_z', 'N_z,ref', 'P_s', 'P_s,ref', 'N_yr', 'N_yr,ref', 'Throttle']
    colors = ['r', 'r', 'lime', 'lime', 'b', 'b', 'c']

    for i, (ys, label, color) in enumerate(zip(ys_list, labels, colors)):
        lt = '-' if i % 2 == 0 else ':'
        lw = 1 if i % 2 == 0 else 3

        ax.plot(times, ys, lt, lw=lw, color=color, label=label)

    ax.set_ylabel('Autopilot (deg & percent)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

def plot_inner_loop(run_sim_result, title='Inner Loop Controls'):
    'plot inner loop controls over time'

    init_plot()

    res = run_sim_result
    assert 'u_list' in res, "Simulation must be run with extended_states=True"
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    u_list = res['u_list']

    # u is throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    ys_list = []

    rad_to_deg_factor = 180 / math.pi

    for i in range(4):
        factor = 1.0 if i == 0 else rad_to_deg_factor
        ys_list.append([u[i] * factor for u in u_list])

    labels = ['Throttle', 'Elevator', 'Aileron', 'Rudder']
    colors = ['b-', 'r-', '#FFA500', 'm-']

    for ys, label, color in zip(ys_list, labels, colors):
        ax.plot(times, ys, color, label=label)

    ax.set_ylabel('Controls (deg & percent)')
    ax.set_xlabel('Time (sec)')

    if title is not None:
        ax.set_title(title)

    ax.legend()
    plt.tight_layout()

def plot_single(run_sim_result, state_name, title=None):
    'plot a single variable over time'

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['states']

    index = get_state_names().index(state_name)
    ys = states[:, index] # 11: altitude (ft)

    ax.plot(times, ys, '-')

    ax.set_ylabel(state_name)
    ax.set_xlabel('Time')

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

def plot_altitude(run_sim_result):
    '''altitude over time plot from run_f16_sum result object

    note: call plt.show() afterwards to have plot show up
    '''

    plot_single(run_sim_result, 'alt')
    

def plot2d(filename, times, plot_data_list):
    '''plot state variables in 2d

    plot data list of is a list of (values_list, var_data),
    where values_list is an 2-d array, the first is time step, the second is a state vector
    and each var_data is a list of tuples: (state_index, label)
    '''

    num_plots = sum([len(var_data) for _, var_data in plot_data_list])

    fig = plt.figure(figsize=(7, 5))

    for plot_index in range(num_plots):
        ax = fig.add_subplot(num_plots, 1, plot_index + 1)
        ax.tick_params(axis='both', which='major', labelsize=16)

        sum_plots = 0
        states = None
        state_var_data = None

        for values_list, var_data in plot_data_list:
            if plot_index < sum_plots + len(var_data):
                states = values_list
                state_var_data = var_data
                break

            sum_plots += len(var_data)

        state_index, label = state_var_data[plot_index - sum_plots]

        if state_index == 0 and isinstance(states[0], float): # state is just a single number
            ys = states
        else:
            ys = [state[state_index] for state in states]

        ax.plot(times, ys, '-')

        ax.set_ylabel(label, fontsize=16)

        # last one gets an x axis label
        if plot_index == num_plots - 1:
            ax.set_xlabel('Time', fontsize=16)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
