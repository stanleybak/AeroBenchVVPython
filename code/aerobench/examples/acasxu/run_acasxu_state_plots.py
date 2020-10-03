'''
Stanley Bak

AcasXu f16 sim

This script makes the state history plots (attitude, inner loop contols, ect) for a single simulation.
'''

import math

import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import StateIndex, extract_single_result
from aerobench.visualize import plot
from aerobench.lowlevel.low_level_controller import LowLevelController

from acasxu_autopilot import AcasXuAutopilot

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

    init += [0] * llc.get_num_integrators()

    return init

def main():
    'main function'

    tmax = 20 # simulation time

    step = 1/30
    extended_states = True

    llc = LowLevelController()
        
    init = intruder_initial_state(llc)
    init += ownship_initial_state(llc)

    ap = AcasXuAutopilot(init, llc)

    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states)
    t = res['runtime']
    print(f"Sim Completed in {round(t, 2)} seconds (extended_states={extended_states})")

    plot_states(res, ap)

def plot_states(res, ap):
    'make traditional plots'

    plot.init_plot()

    plot.plot_overhead(res, llc=ap.llc)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    plt.close()

    labels = ['Intruder', 'Ownship']

    fig, axs = plt.subplots(4, 2, figsize=(14, 20))

    #plt.rcParams["font.size"] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 24
    plt.tick_params(labelsize=6)

    for col, label in enumerate(labels):
        res_single = extract_single_result(res, col, ap.llc)

        row = 0

        plot.plot_single(res_single, 'alt', title=f'{label} Altitude (ft)', ax=axs[row, col])
        row += 1

        plot.plot_attitude(res_single, title=f'{label} Attitude', ax=axs[row, col])
        row += 1

        # plot inner loop controls + referenceso
        plot.plot_inner_loop(res_single, title=f'{label} Inner Loop Controls', ax=axs[row, col])
        row += 1

        # plot outer loop controls + references
        plot.plot_outer_loop(res_single, title=f'{label} Outer Loop Controls', ax=axs[row, col])
        row += 1

    filename = 'states.png'
    fig.tight_layout()
    fig.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
