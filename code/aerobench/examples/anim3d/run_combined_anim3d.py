'''
Stanley Bak

plots 3d animation for 'u_turn' scenario 
'''

import sys

import numpy as np

from aerobench.visualize import anim3d

from run_GCAS_anim3d import simulate as simulate_gcas
from run_u_turn_anim3d import simulate as simulate_waypoint

def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res_gcas = simulate_gcas()

    res_waypoint, init_extra, update_extra, skip_waypoint, _waypoints = simulate_waypoint(filename)

    res_list = [res_gcas, res_waypoint]
    scale_list = [None, 70]
    viewsize_list = [None, 5000]
    viewsize_z_list = [None, 4000]
    trail_pts_list = [None, np.inf]
    elev_list = [15, 27]
    azim_list = [-150, -107]

    skip_gcas = 3 if filename.endswith('.gif') else None
    
    skip_list = [skip_gcas, skip_waypoint]
    chase_list = [False, True]
    fixed_floor_list = [False, True]
    init_extra_list = [None, init_extra]
    update_extra_list = [None, update_extra]

    anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
                     trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
                     chase=chase_list, fixed_floor=fixed_floor_list,
                     init_extra=init_extra_list, update_extra=update_extra_list)

if __name__ == '__main__':
    main()
