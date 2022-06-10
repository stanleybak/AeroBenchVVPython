<p align="center"> <img src="anim3d.gif"/> </p>

Note: This is the v2 branch of the code, which is now a python3 project and includes more modularity and general simulation capabilities. For the original benchmark paper version see the v1 branch.

# AeroBenchVVPython Overview
This project contains a python version of models and controllers that test automated aircraft maneuvers by performing simulations. The hope is to provide a benchmark to motivate better verification and analysis methods, working beyond models based on Dubins car dynamics, towards the sorts of models used in aerospace engineering. Roughly speaking, the dynamics are nonlinear, have about 10-20 dimensions (continuous state variables), and hybrid in the sense of discontinuous ODEs, but not with jumps in the state. 

This is a python port of the original matlab version, which can can see for
more information: https://github.com/pheidlauf/AeroBenchVV

# Citation

For citation purposes, please use: "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers", P. Heidlauf, A. Collins, M. Bolender, S. Bak, 5th International Workshop on Applied Verification for Continuous and Hybrid Systems (ARCH 2018)

# Required Libraries 

The following Python libraries are required (can be installed using `sudo pip install <library>`):

`numpy` - for matrix operations

`scipy` - for simulation / numerical integration (RK45) and trim condition optimization 

`matplotlib` - for animation / plotting (requires `ffmpeg` for .mp4 output or `imagemagick` for .gif)

`slycot` - for control design (not needed for simulation)

`control` - for control design (not needed for simulation)

### Animation isuses
Use matplotlib version 3.1.1 if you get errors like: 
"art3d.py", line 175, in set_3d_properties
    zs = np.broadcast_to(zs, xs.shape)
AttributeError: 'list' object has no attribute 'shape'

### Release Documentation
Distribution A: Approved for Public Release (88ABW-2020-2188) (changes in this version)
    
Distribution A: Approved for Public Release (88ABW-2017-6379) (v1)
