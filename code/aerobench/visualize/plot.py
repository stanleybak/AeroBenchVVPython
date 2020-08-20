'''
Stanley Bak
Python code for F-16 animation video output
'''

import math
import time
import traceback
import os

import numpy as np
from numpy import rad2deg

# these imports are needed for 3d plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from scipy.io import loadmat

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

def scale3d(pts, scale_list):
    'scale a 3d ndarray of points, and return the new ndarray'

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv

def rotate3d(pts, theta, psi, phi):
    'rotates an ndarray of 3d points, returns new list'

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([ \
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta], \
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi, \
        -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi, \
        -cosTheta * sinPhi], \
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi, \
        sinPsi * sinTheta * cosPhi + cosPsi * sinPhi, \
        cosTheta * cosPhi]], dtype=float)

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv

def plot3d_anim(times, states, modes, ps_list, Nz_list, skip=1, filename=None):
    '''
    make a 3d plot of the GCAS maneuver
    '''

    full_plot = True

    if filename == '': # plot to the screen
        filename = None
        skip = 20
        full_plot = False
    elif filename.endswith('.gif'):
        skip = 5
    else:
        skip = 1 # plot every frame

    assert len(times) == len(states)

    start = time.time()

    times = times[0::skip]
    states = states[0::skip]
    modes = modes[0::skip]
    ps_list = ps_list[0::skip]
    Nz_list = Nz_list[0::skip]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)

    pos_xs = [pt[9] for pt in states]
    pos_ys = [pt[10] for pt in states]
    pos_zs = [pt[11] for pt in states]

    trail_line, = ax.plot([], [], [], color='r', lw=1)

    data = loadmat('f-16.mat')
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color=None if full_plot else 'k')
    ax.add_collection3d(plane_polys)

    ax.set_xlim([min(pos_xs), max(pos_xs)])
    ax.set_ylim([min(pos_ys), max(pos_xs)])
    ax.set_zlim([min(pos_zs), max(pos_zs)])

    ax.set_xlabel('X [ft]')
    ax.set_ylabel('Y [ft]')
    ax.set_zlabel('Altitude [ft] ')
    frames = len(times)

    # text
    fontsize = 14
    time_text = ax.text2D(0.05, 1.07, "", transform=ax.transAxes, fontsize=fontsize)
    mode_text = ax.text2D(0.95, 1.07, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alt_text = ax.text2D(0.05, 1.00, "", transform=ax.transAxes, fontsize=fontsize)
    v_text = ax.text2D(0.95, 1.00, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    alpha_text = ax.text2D(0.05, 0.93, "", transform=ax.transAxes, fontsize=fontsize)
    beta_text = ax.text2D(0.95, 0.93, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    nz_text = ax.text2D(0.05, 0.86, "", transform=ax.transAxes, fontsize=fontsize)
    ps_text = ax.text2D(0.95, 0.86, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='right')

    ang_text = ax.text2D(0.5, 0.79, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='center')

    def anim_func(frame):
        'updates for the animation frame'

        print("{}/{}".format(frame, frames))

        speed = states[frame][0]
        alpha = states[frame][1]
        beta = states[frame][2]
        alt = states[frame][11]
        phi = states[frame][3]
        theta = states[frame][4]
        psi = states[frame][5]
        dx = states[frame][9]
        dy = states[frame][10]
        dz = states[frame][11]

        time_text.set_text('t = {:.2f} sec'.format(times[frame]))

        colors = ['red', 'blue', 'green', 'magenta']
        
        mode = modes[frame]
        col = colors[mode % len(colors)]
        mode_text.set_color(col)
        mode_text.set_text('Mode: {}'.format(mode))

        alt_text.set_text('h = {:.2f} ft'.format(alt))
        v_text.set_text('V = {:.2f} ft/sec'.format(speed))

        alpha_text.set_text('$\\alpha$ = {:.2f} deg'.format(rad2deg(alpha)))
        beta_text.set_text('$\\beta$ = {:.2f} deg'.format(rad2deg(beta)))

        nz_text.set_text('$N_z$ = {:.2f} g'.format(Nz_list[frame]))
        ps_text.set_text('$p_s$ = {:.2f} deg/sec'.format(rad2deg(ps_list[frame])))

        ang_text.set_text('[$\\phi$, $\\theta$, $\\psi$] = [{:.2f}, {:.2f}, {:.2f}] deg'.format(\
            rad2deg(phi), rad2deg(theta), rad2deg(psi)))

        # do trail
        trail_len = 200 / skip
        start_index = max(0, frame-trail_len)
        trail_line.set_data(pos_xs[start_index:frame], pos_ys[start_index:frame])
        trail_line.set_3d_properties(pos_zs[start_index:frame])

        scale = 25
        pts = scale3d(f16_pts, [-scale, scale, scale])

        pts = rotate3d(pts, theta, -psi, phi)

        size = 1000
        minx = dx - size
        maxx = dx + size
        miny = dy - size
        maxy = dy + size
        minz = dz - size
        maxz = dz + size

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_zlim([minz, maxz])

        verts = []
        fc = []
        count = 0

        for face in f16_faces:
            face_pts = []

            count = count + 1

            if not full_plot and count % 10 != 0:
                continue

            for index in face:
                face_pts.append((pts[index-1][0] + dx, \
                    pts[index-1][1] + dy, \
                    pts[index-1][2] + dz))

            verts.append(face_pts)
            fc.append('k')

        # draw ground
        if minz <= 0 and maxz >= 0:
            z = 0
            verts.append([(minx, miny, z), (maxx, miny, z), (maxx, maxy, z), (minx, maxy, z)])
            fc.append('0.8')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolors(fc)

        return None

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=30, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=80, writer='imagemagick')
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 60
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
            except AttributeError:
                traceback.print_exc()
                print("\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        plt.show()

def plot_overhead(run_sim_result, waypoints=None):
    '''altitude over time plot from run_f16_sum result object

    note: call plt.show() afterwards to have plot show up
    '''

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)

    states = res['states']

    ys = states[:, 9] # 9: n/s position (ft)
    xs = states[:, 10] # 10: e/w position (ft)

    ax.plot(xs, ys, '-')

    ax.plot([xs[0]], [ys[1]], 'k*', ms=8, label='Start')

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

def plot_attitude(run_sim_result, title='Attitude History'):
    'plot a single variable over time'

    init_plot()

    res = run_sim_result
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1)
    ax.ticklabel_format(useOffset=False)

    times = res['times']
    states = res['states']

    indices = [StateIndex.PHI, StateIndex.THETA, StateIndex.PSI, StateIndex.P, StateIndex.Q, StateIndex.R]
    labels = ['Roll (Phi)', 'Pitch (Theta)', 'Yaw (Psi)', 'Roll Rate (P)', 'Pitch Rate (Q)', 'Yaw Rate (R)']
    colors = ['r-', 'g-', 'b-', 'r--', 'g--', 'b--']

    rad_to_deg_factor = 180 / math.pi

    for index, label, color in zip(indices, labels, colors):
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
