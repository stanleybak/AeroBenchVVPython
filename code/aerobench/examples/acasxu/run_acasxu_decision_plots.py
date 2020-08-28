'''
Stanley Bak

AcasXu f16 sim

This script makes the decision point plots for multiple simulations.
'''

import math
import random
import multiprocessing
import time
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import StateIndex
from aerobench.visualize import plot
from aerobench.lowlevel.low_level_controller import LowLevelController

from acasxu_autopilot import AcasXuAutopilot

def main(count):
    'main function to make plots'

    tmax = 20 # simulation time

    step = 1/30
    extended_states = False

    random.seed(0)

    params = []

    # load from cache
    cache_filename = "history_cache.pkl"
    history_list = []

    try:
        with open(cache_filename, "rb") as f:
            history_list = pickle.load(f)
            print(f"loaded {len(history_list)} sims from cache file {cache_filename}")
    except EnvironmentError:
        print(f"cache file {cache_filename} doesn't yet exist")

    start = time.perf_counter()
    llc = LowLevelController()
        
    for i in range(count):
        # still run these to get random number generator in same state
        init = intruder_initial_state(llc)
        init += ownship_initial_state(llc)

        if i < len(history_list):
            continue

        param = (i, count, init, llc, tmax, step, extended_states)
        params.append(param)

    if len(params) == 0:
        new_results = []
    else:
        with multiprocessing.Pool(processes=12) as pool:
            new_results = pool.starmap(sim_get_history, params)

    history_list += new_results

    diff = time.perf_counter() - start
    print(f"Total sim time: {round(diff, 2)} sec")

    # save cache
    if len(new_results) > 0:
        with open(cache_filename, "wb") as f:
            pickle.dump(history_list, f)
            print(f"saved {len(history_list)} sims to {cache_filename}")

    history_list = history_list[:count]
        
    plot_decisions(history_list)

def intruder_initial_state(llc):
    'returns intruder initial state'

    # use trim state
    init = list(llc.xequil)

    init[StateIndex.VT] = 807
    init[StateIndex.POSN] = 25000
    init[StateIndex.PSI] = math.pi

    init += [0] * llc.get_num_integrators()

    return init

def ownship_initial_state(llc):
    'returns intruder initial state'

    # use trim state
    init = list(llc.xequil)

    init[StateIndex.VT] = 807

    rad = 5000
    init[StateIndex.POS_E] += random.random() * 2*rad - rad

    y_rad = 200
    init[StateIndex.POS_N] += random.random() * 2*y_rad - y_rad

    init += [0] * llc.get_num_integrators()

    return init

def sim_get_history(i, count, init, llc, tmax, step, extended_states):
    'simulate and return autopilot history'

    ap = AcasXuAutopilot(init, llc)

    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states)
    t = res['runtime']
    print(f"Simulation {i+1}/{count} Completed in {round(t, 2)} seconds (extended_states={extended_states})")

    return ap.history

def plot_decisions(history_list):
    'plot commands with their points'

    plot.init_plot()

    count = len(history_list)

    if count <= 100:
        ms = 4.0
    elif count <= 1000:
        ms = 2.0
    elif count <= 5000:
        ms = 0.5
    else:
        ms = 0.2

    filename = f'decisions_{count}.png'
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)

    colors = ['grey', 'skyblue', 'lightcoral', 'deepskyblue', 'firebrick']

    xs_list = [[], [], [], [], []]
    ys_list = [[], [], [], [], []]

    for history in history_list:
        for command, state in history:

            y = state[StateIndex.POSN] # 9: n/s position (ft)
            x = state[StateIndex.POSE] # 10: e/w position (ft)

            xs_list[command].append(x)
            ys_list[command].append(y)

    for xs, ys, c in zip(xs_list, ys_list, colors):
        ax.plot(xs, ys, c, marker='o', ms=ms, lw=0)

    ax.set_ylabel('North / South Position (ft)', fontsize=24)
    ax.set_xlabel('East / West Position (ft)', fontsize=24)

    ax.set_title(f'Decision Points ({len(history_list)} Sims)', fontsize=30)

    ax.axis('equal')

    custom_lines = [Line2D([0], [0], color=colors[3], lw=0, marker='o', ms=4),
                    Line2D([0], [0], color=colors[1], lw=0, marker='o', ms=4),
                    Line2D([0], [0], color=colors[0], lw=0, marker='o', ms=4),
                    Line2D([0], [0], color=colors[2], lw=0, marker='o', ms=4),
                    Line2D([0], [0], color=colors[4], lw=0, marker='o', ms=4)]

    ax.legend(custom_lines, ['Strong Left', 'Weak Left', 'Clear', 'Weak Right', 'Strong Right'], \
            fontsize=16, loc='best')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Wrote {filename}")

if __name__ == '__main__':
    for count in [100, 1000, 10000]:
        main(count)
