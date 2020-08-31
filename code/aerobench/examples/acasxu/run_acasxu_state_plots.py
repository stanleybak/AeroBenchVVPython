'''
Stanley Bak

AcasXu f16 sim

This script makes the state history plots (attitude, inner loop contols, ect) for a single simulation.
'''

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

    plot.plot_overhead(res, llc=ap.llc)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    labels = ['intruder', 'ownship']

    for i, l in enumerate(labels):
        res_single = extract_single_result(res, i, ap.llc)

        plot.plot_single(res_single, 'alt', title='Altitude (ft)')
        filename = f'alt_{l}.png'
        plt.savefig(filename)
        print(f"Made {filename}")

        plot.plot_attitude(res_single)
        filename = f'attitude_{l}.png'
        plt.savefig(filename)
        print(f"Made {filename}")

        # plot inner loop controls + references
        plot.plot_inner_loop(res_single)
        filename = f'inner_loop_{l}.png'
        plt.savefig(filename)
        print(f"Made {filename}")

        # plot outer loop controls + references
        plot.plot_outer_loop(res_single)
        filename = f'outer_loop_{l}.png'
        plt.savefig(filename)
        print(f"Made {filename}")

if __name__ == '__main__':
    main()
