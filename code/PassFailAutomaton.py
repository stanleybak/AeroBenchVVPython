'''
Stanley Bak
Pass / Fail Conditions State-Machine Logic

This class has a method, .advance() that gets called at each step with the current state and time.
A second method .result() returns True iff all conditions passed (and may print a summary as well)
'''

import numpy as np

from util import Freezable

class PassFailAutomaton(Freezable):
    '''An abstract class for a pass fail automaton'''

    def __init__(self, printOn):

        self.freeze_attrs()

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''
        advance the pass/fail automaton state given the current state and time
        '''

        return False

    @abc.abstractmethod
    def result(self):
        '''
        returns True iff all conditions passed... may print a summary as well
        '''
        return

class GcasPassFailAutomaton(PassFailAutomaton):
    '''The ground-collision avoidance system pass / fail logic'''

    AIRSPEED = 0
    ALPHA_LIMIT = 1
    BETA_LIMIT = 2
    NZ_LIMIT = 3
    PS_RATE_LIMIT = 4
    ALTITUDE_LIMIT = 5
    MANEUVER_TIME = 6 # was maneuver completed on time?
    NUM_CONDITIONS = 7

    def __init__(self, xequil, uequil, flightLimits, ctrlLimits):

        self.passed = [True] * GcasPassFailAutomaton.NUM_CONDITIONS
        self.passed[MANEUVER_TIME] = False

        PassFailAutomaton.__init__(self)

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''advance the automaton state based on the current aircraft state'''

    # Check airspeed limits
    if(maxAirspeed > flightLimits.vMax || minAirspeed < flightLimits.vMin)
        passFail.airspeed = false;
        passFail.stable = false;
    else
        passFail.airspeed = true;
    end

    % Check alpha limits
    if(max(x_f16_hist(2,:)) > deg2rad(flightLimits.alphaMaxDeg) || ...
            min(x_f16_hist(2,:)) < deg2rad(flightLimits.alphaMinDeg))
        passFail.alpha = false;
        passFail.stable = false;
    else
        passFail.alpha = true;
    end

    % Check beta limits
    if(abs(x_f16_hist(3,:)) > deg2rad(flightLimits.betaMaxDeg))
        passFail.beta = false;
        passFail.stable = false;
    else
        passFail.beta = true;
    end

    % Check Nz limits
    if(minNz < flightLimits.NzMin || maxNz > flightLimits.NzMax)
        passFail.Nz = false;
    else
        passFail.Nz = true;
    end

    % Check Ps_rate limits
    if(max_ps_accel > flightLimits.psMaxAccelDeg)
        passFail.psMaxAccelDeg = false;
    else
        passFail.psMaxAccelDeg = true;
    end

    % Check altitude limits
    if(min(x_f16_hist(12,:)) < flightLimits.altitudeMin ||...
            max(x_f16_hist(12,:)) > flightLimits.altitudeMax)
        passFail.altitude = false;
    else
        passFail.altitude = true;
    end

    % Check maneuver time limits (GCAS?)
    if(t_maneuver(2)-t_maneuver(1) > flightLimits.maneuverTime ||...
            t_maneuver(2) < 0)
        passFail.maneuverTime = false;
    else
        passFail.maneuverTime = true;
    end

    @abc.abstractmethod
    def result(self):
        '''returns True if all conditions passed'''

        return np.all(passed)

