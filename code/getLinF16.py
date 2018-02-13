'''
Stanley Bak
F16 GCAS in Python
Get linearized version of f16 model about a trim point
'''

def getLinF16( xequil, uequil, printOn, model='stevens'):
    'get the 4-tuple A, B, C, D for the linearized version of the F-16 about a setpoint'

#Given equilibrium trim and controls, returns a linearized state space 
# model of the F - 16.
#
#   '''
Stanley Bak
F16 GCAS in Python
'''

def  getLinF16( xequil, uequil):
'desc'

#       lin_f16 = getLinF16( xequil, uequil, printOn )
#
#   Inputs:
#       xequil  -   Equilibrium states (13x1)
#       uequil  -   Equilibrium control (4x1)
#       printOn -   If true, prints intermediate data
#
#   Outputs:
#       lin_f16 -   labeled state space model of f16 
#                   (13 state, 4 control, 10 output) 
#
#   x_f16 states:
#       x_f16(1)  = air speed, VT                           (ft / s)
#       x_f16(2)  = angle of attack, alpha                  (rad)
#       x_f16(3)  = angle of sideslip, beta                 (rad)
#       x_f16(4)  = roll angle, phi                         (rad)
#       x_f16(5)  = pitch angle, theta                      (rad)
#       x_f16(6)  = yaw angle, psi                          (rad)
#       x_f16(7)  = roll rate, P                            (rad / s)
#       x_f16(8)  = pitch rate, Q                           (rad / s)
#       x_f16(9)  = yaw rate, R                             (rad / s)
#       x_f16(10) = northward horizontal displacement, pn   (ft)
#       x_f16(11) = eastward horizontal displacement, pe    (ft)
#       x_f16(12) = altitude, h                             (ft)
#       x_f16(13) = engine thrust dynamics lag state, pow   (lbs)
#
#   x_f16 controls:
#       u(1) = throttle                                     (0 to 1)
#       u(2) = elevator                                     (rad?) 
#       u(3) = aileron                                      (rad?)
#       u(4) = rudder                                       (rad?)
#
# <a href = "https: / /github.com / pheidlauf / AeroBenchVV">AeroBenchVV< / a>
# Copyright: GNU General Public License 2017
#
# See also: TRIMMERFUN, JACOBFUN

if(nargin = =0)   
    printOn = true
 
    # SET THESE VALUES MANUALLY
    hg = 0
         # Altitude guess (ft msl)
    Vtg = 502
      # Velocity guess (ft / sec)
    phig = 0
       # Roll angle from horizontal guess (deg)
    thetag = 0
     # Pitch angle guess (deg)
    xguess = [Vtg 0 0 phig thetag 0 0 0 0 0 0 hg 0]

    # u = [throttle elevator aileron rudder]
    uguess = [.2 0 0 0]

    # Orientation for Linearization
    # 1:    Wings Level (gamma = 0)
    # 2:    Wings Level (gamma <> 0)
    # 3:    Constant Altitude Turn
    # 4:    Steady Pull Up
    orient = 4
    inputs = [xguess(1), xguess(12), 0, 0, 0]
   
    [xequil, uequil] = trimmerFun(xguess, uguess, orient, inputs, printOn)
end
if(nargin = =2)
    printOn = true
end

[A, B, C, D] = jacobFun(xequil, uequil, printOn)

# y = [ Az q alpha theta Vt Ay p r beta phi ]T
C([2:4 7:10], :) = deg2rad(C([2:4 7:10], :))
D([2:4 7:10], :) = deg2rad(D([2:4 7:10], :))

# Build Default Linear SS Model
lin_f16 = ss(A, B,C, D)
lin_f16.stateName = {'Vt', 'alpha', 'beta', ...
    'phi', 'theta', 'psi', ...
    'p', 'q', 'r', ....
    'pn', 'pe', 'alt', ...
    'pow'}
lin_f16.stateUnit = {'ft / s', 'rad', 'rad', ...
    'rad', 'rad', 'rad', ...
    'rad', 'rad', 'rad', ...
    'ft', 'ft', 'ft', 'lbs'}
lin_f16.inputName = {'Throttle', 'Elevator', 'Aileron', 'Rudder'}
lin_f16.inputUnit = {'percent', 'rad', 'rad', 'rad'}
lin_f16.outputName = {'Az', 'q', 'alpha', 'theta', 'Vt', 'Ay', ...
    'p', 'r', 'beta', 'phi'}
lin_f16.outputUnit = {'g''s', 'rad / s', 'rad', 'rad', 'ft / s', ...
    'g''s', 'rad / s', 'rad / s', 'rad', 'rad'}
lin_f16.name = 'Linearized F - 16 SS Model'

if(printOn)
    disp('Linearized F - 16 SS Model')
    lin_f16
end

end
