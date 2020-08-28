#!/bin/bash

time python3 run_combined_anim3d.py combined.gif
time python3 run_combined_anim3d.py combined.mp4

time python3 run_GCAS_anim3d.py gcas.mp4
time python3 run_GCAS_anim3d.py gcas.gif

time python3 run_u_turn_anim3d.py u_turn.mp4
time python3 run_u_turn_anim3d.py u_turn.gif
