'''
2d animation utilities for aerobench
'''

import math
import time
import os
import traceback

from scipy import ndimage

from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from aerobench.visualize import plot
from aerobench.util import StateIndex, get_script_path, get_state_names


def make_anim(res, llc, filename, skip_frames=None, figsize=(7, 5), show_closest=True, print_frame=True,
              init_extra=None, update_extra=None):
    '''
    make a 2d animiation

    params can be a list for back-to-back animations
    '''

    plot.init_plot()
    start = time.time()

    if not isinstance(res, list):
        res = [res]

    if not isinstance(skip_frames, list):
        skip_frames = [skip_frames]

    if not isinstance(init_extra, list):
        init_extra = [init_extra]

    if not isinstance(update_extra, list):
        update_extra = [update_extra]

    #####
    # fill in defaults
    for i, skip in enumerate(skip_frames):
        if skip is not None:
            continue

        if filename == '': # plot to the screen
            skip_frames[i] = 5
        elif filename.endswith('.gif'):
            skip_frames[i] = 2
        else:
            skip_frames[i] = 1 # plot every frame

    if filename == '':
        filename = None

    ##
    all_times = []
    all_states = []
    all_modes = []

    for r, skip in zip(res, skip_frames):
        t = r['times']
        s = r['states']
        m = r['modes']

        t = t[0::skip]
        s = s[0::skip]
        m = m[0::skip]

        all_times.append(t)
        all_states.append(s)
        all_modes.append(m)
            
    ##

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ##

    num_planes_list = []
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    max_num_planes = 0

    for states in all_states:
        s0 = states[0]
        assert(s0.size % num_vars) == 0
        
        num_planes = s0.size // num_vars
        num_planes_list.append(num_planes)
        max_num_planes = max(max_num_planes, num_planes)

    ax.set_xlabel('X [ft]', fontsize=14)
    ax.set_ylabel('Y [ft]', fontsize=14)

    states = all_states[0]
    axis_limits = plot.set_axis_limits(ax, num_vars, states)

    parent = get_script_path(__file__)
    plane_path = os.path.join(parent, 'airplane.png')
    img = plt.imread(plane_path)

    planes = []
    plane_trails = []
    plane_previews = []
    plane_size_factor = 0.05 # percent of axis limits

    for i in range(max_num_planes):
        size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
        box = Bbox.from_bounds(0, 0, size, size)
        tbox = TransformedBbox(box, ax.transData)
        box_image = BboxImage(tbox, zorder=2)

        #theta_deg = (theta - math.pi / 2) / math.pi * 180 # original image is facing up, not right
        theta_deg = 0
        img_rotated = ndimage.rotate(img, theta_deg, order=1)

        box_image.set_data(img_rotated)
        ax.add_artist(box_image)

        planes.append(box_image)

        trail, = ax.plot([], [], lw=1, zorder=2)
        plane_trails.append(trail)

        preview, = ax.plot([], [], ':', color=trail.get_color(), lw=1, zorder=1)
        plane_previews.append(preview)

    # text
    fontsize = 13
    cur_min_text = ax.text(0.47, 0.96, "", transform=ax.transAxes, fontsize=fontsize, color='b',
                           horizontalalignment='center')
    
    acc_min_text = ax.text(0.97, 0.96, "", transform=ax.transAxes, fontsize=fontsize, color='r',
                           horizontalalignment='right')
    
    time_text = ax.text(0.03, 0.96, "", transform=ax.transAxes, fontsize=fontsize, horizontalalignment='left')

    l1, = ax.plot([], [], 'b:', lw=2, zorder=4)
    l2, = ax.plot([], [], 'r:', lw=2, zorder=5)
    min_lines = [l1, l2]

    if not show_closest:
        l1.set_visible(False)
        l2.set_visible(False)

    extra_lines = []

    for func in init_extra:
        if func is not None:
            extra_lines.append(func(ax))
        else:
            extra_lines.append([])

    first_frames = []
    frames = 0

    for t in all_times:
        first_frames.append(frames)
        frames += len(t)

    dist_data = [0, [[0], [0]]] # cur_global_min, cur_global_min_line (list of xs, ys)

    def anim_func(global_frame):
        'updates for the animation frame'

        index = 0
        first_frame = False

        for i, f in enumerate(first_frames):
            if global_frame >= f:
                index = i

                if global_frame == f:
                    first_frame = True
                    break

        frame = global_frame - first_frames[index]
        states = all_states[index]
        times = all_times[index]
        modes = all_modes[index]

        if print_frame:
            print(f"Frame: {global_frame}/{frames} - Index {index} frame {frame}/{len(times)}")

        time_text.set_text(f"Time: {round(times[frame], 1)} sec")

        mode = modes[frame]

        state = states[frame]
        num_planes = num_planes_list[index]
        
        cur_min = np.inf
        min_line = [[], []] # list of [xs, ys]
        
        for p1 in range(num_planes):
            s1 = state[p1*num_vars:(p1+1)*num_vars]
            
            x1 = s1[StateIndex.POS_E]
            y1 = s1[StateIndex.POS_N]
            
            for p2 in range(p1+1, num_planes):
                s2 = state[p2*num_vars:(p2+1)*num_vars]

                x2 = s2[StateIndex.POS_E]
                y2 = s2[StateIndex.POS_N]

                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

                if dist < cur_min:
                    cur_min = dist
                    min_line = [[x1, x2], [y1, y2]]

        cur_min_text.set_text(f"Cur Dist: {round(cur_min)} ft")

        if first_frame or cur_min < dist_data[0]:
            dist_data[0] = cur_min
            dist_data[1] = min_line

        acc_min_text.set_text(f"Min Dist: {round(dist_data[0])} ft")

        min_lines[0].set_data(min_line[0], min_line[1])
        min_lines[1].set_data(dist_data[1][0], dist_data[1][1])

        if first_frame:
            axis_limits[:] = plot.set_axis_limits(ax, num_vars, states)

            # initialize dist_stats (index 0: cur_min, index 1: total_min)
            for i, lines in enumerate(extra_lines):
                for line in lines:
                    line.set_visible(i == index)

            for i, (p, t) in enumerate(zip(planes, plane_trails)):
                is_vis = i < num_planes_list[index]

                p.set_visible(is_vis)
                t.set_visible(is_vis)

        # do trail
        for i in range(num_planes):
            # do trails
            offset = i * num_vars
            pos_xs = [pt[offset + StateIndex.POS_E] for pt in states]
            pos_ys = [pt[offset + StateIndex.POS_N] for pt in states]

            plane_trails[i].set_data(pos_xs[:frame], pos_ys[:frame])
            plane_previews[i].set_data(pos_xs, pos_ys)

            # do plane images
            s = state[i*num_vars:(i+1)*num_vars]
            psi = s[StateIndex.PSI]
            x = s[StateIndex.POS_E]
            y = s[StateIndex.POS_N]

            theta_deg = -psi * 180 / math.pi
            original_size = list(img.shape)
            img_rotated = ndimage.rotate(img, theta_deg, order=1)
            rotated_size = list(img_rotated.shape)
            ratios = [r / o for r, o in zip(rotated_size, original_size)]
            planes[i].set_data(img_rotated)

            size = (axis_limits[1] - axis_limits[0]) * plane_size_factor
            width = size * ratios[0]
            height = size * ratios[1]
            box = Bbox.from_bounds(x - width/2, y - height/2, width, height)
            tbox = TransformedBbox(box, ax.transData)
            planes[i].bbox = tbox

        if update_extra[index] is not None:
            update_extra[index](frame, times[frame], state, mode)

    plt.tight_layout()

    interval = 30

    if filename is not None and filename.endswith('.gif'):
        interval = 60

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=interval, \
        blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=60, writer='imagemagick') # dpi was 80
            print("Finished saving to {} in {:.1f} sec".format(filename, time.time() - start))
        else:
            fps = 40
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
