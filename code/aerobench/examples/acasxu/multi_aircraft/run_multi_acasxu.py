'''
Stanley Bak

AcasXu f16 sim

This script simulates multiple aircraft.
'''

import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import StateIndex, get_state_names
from aerobench.visualize import plot, anim
from aerobench.lowlevel.low_level_controller import LowLevelController

from aerobench.examples.acasxu.acasxu_autopilot import AcasXuAutopilot

def make_init(llc, num_aircraft, diameter=25000):
    'returns combined initial state'

    rv = []

    # psi = 0 is facing up
    step = 2* math.pi / num_aircraft
    rad = diameter / 2
    center = (0, rad/2)

    theta_offset = -math.pi/2

    # use trim state

    for a in range(num_aircraft):
        #print("debug: only making 0 and 5")
        #if a != 0 and a != 5:
        #    continue

        theta = theta_offset + step * a
        y = rad * math.sin(theta)
        x = rad* math.cos(theta)

        psi = -(theta + theta_offset) + math.pi

        init = list(llc.xequil)
        init[StateIndex.VT] = 807
        init[StateIndex.POSE] = x + center[0]
        init[StateIndex.POSN] = y + center[1]
        init[StateIndex.PSI] = psi
        init += [0] * llc.get_num_integrators()

        rv += init

    return rv

def main():
    'main function to make plots'

    tmax = 100 # 3 # simulation time

    step = 1/30

    llc = LowLevelController()
    num_vars = len(get_state_names()) + llc.get_num_integrators()

    if len(sys.argv) > 1:
        num_aircraft = int(sys.argv[1])
    else:
        num_aircraft = 3

    if len(sys.argv) > 2:
        diameter = float(sys.argv[2])
    else:
        diameter = 25000

    print(f"Num Aircraft {num_aircraft}, Init Circle Diameter: {diameter}")

    init = make_init(llc, num_aircraft, diameter=diameter)

    # int case size changed in init
    num_aircraft = len(init) // num_vars

    num_aircraft_acasxu = num_aircraft
    stop_on_coc = True
    ap = AcasXuAutopilot(init, llc, num_aircraft_acasxu=num_aircraft_acasxu, stop_on_coc=stop_on_coc)

    ap.coc_stop_delay = 20

    #print("debug: initial command fixed")
    #ap.next_nn_update = ap.nn_update_rate
    #ap.commands[0] = 3
    #ap.commands[1] = 4

    res = run_f16_sim(init, tmax, ap, step=step)
    t = res['runtime']

    print(f"Simulation Completed in {round(t, 2)} seconds")

    plot.plot_overhead(res, llc=llc)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")
    plt.close()

    mp4_filename = ''

    lines = []
    labels = []

    def anim_init_extra(ax):
        'extra animation initialization'

        for _ in range(num_aircraft):
            l, = ax.plot([], [], color='k', lw=0.5, zorder=1)
            lines.append(l)

            t = ax.text(0, 0, "", fontsize=12, color='k', zorder=10,
                        horizontalalignment='center', verticalalignment='top')
            labels.append(t)

        return lines + labels

    def anim_update_extra(_ax, t, state, _mode):
        'extra animation update'

        closest_tuple = ap.full_history[0]

        for tup in ap.full_history:
            if tup[0] <= t:
                closest_tuple = tup

        for a in range(num_aircraft):
            # find the closest_plane_history for plane index 'a' with the latest time <= t

            label_text = f"{a}"

            s1 = state[a*num_vars:(a+1)*num_vars]
            
            x1 = s1[StateIndex.POS_E]
            y1 = s1[StateIndex.POS_N]

            if a < num_aircraft_acasxu:
                # set label
                command = 0
                all_command_str = ""

                for i, c in enumerate(closest_tuple[1][a]):
                    if i == closest_tuple[2][a]:
                        all_command_str += r"$\overline{" + f"{c}" + r"}$"
                        command = c
                    else:
                        all_command_str += f"${c}$"

                names = ['clear', 'weak-left', 'weak-right', 'strong-left', 'strong-right']
                label_text = f"#{a} ({names[command]}) {all_command_str}"

                b = closest_tuple[2][a]

                if b is None:
                    lines[a].set_visible(False)
                else:
                    lines[a].set_visible(True)

                    s2 = state[b*num_vars:(b+1)*num_vars]
                    x2 = s2[StateIndex.POS_E]
                    y2 = s2[StateIndex.POS_N]

                    endx = 0.9 * x2 + 0.1 * x1
                    endy = 0.9 * y2 + 0.1 * y1

                    lines[a].set_data([x1, endx], [y1, endy])
            else:
                lines[a].set_visible(False)
                
            labels[a].set_text(label_text)
            labels[a].set_x(x1)
            plane_size = 1200
            labels[a].set_y(y1 - plane_size/2)

    extra_info = True

    if not extra_info:
        anim_init_extra = None
        anim_update_extra = None

    skip_frames = 10
    show_closest = False
    print_frame = True

    #mp4_filename = f'acasxu{num_aircraft_acasxu}.mp4'
    mp4_filename = ''
    if len(mp4_filename) > 0:
        skip_frames = None
        
    anim.make_anim(res, llc, mp4_filename, show_closest=show_closest, print_frame=print_frame,
                   skip_frames=skip_frames,
                   init_extra=anim_init_extra, update_extra=anim_update_extra)

if __name__ == '__main__':
    main()
