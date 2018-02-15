'''
Stanley Bak
Pass / Fail Conditions State-Machine Logic

This class has a method, .advance() that gets called at each step with the current state and time.
A second method .result() returns True iff all conditions passed
'''

import abc

from numpy import rad2deg # pylint: disable=E0611

from util import Freezable

class FlightLimits(Freezable):
    'Flight Limits (for pass-fail conditions)'

    def __init__(self):
        self.altitudeMin = 0 # ft AGL
        self.altitudeMax = 10000 #ft AGL
        self.NzMax = 9 # G's
        self.NzMin = -2 #G's
        self.psMaxAccelDeg = 500 # deg/s/s

        # Note: Alpha, Beta, Vt are hard-coded limits. DO NOT CHANGE
        self.vMin = 300 # ft/s
        self.vMax = 900 # ft/s
        self.alphaMinDeg = -10 # deg
        self.alphaMaxDeg = 45 # deg
        self.betaMaxDeg = 30 # deg

        self.freeze_attrs()

class PassFailAutomaton(Freezable):
    '''The parent class for a pass fail automaton... checks each state against the flight envelope limits'''

    def __init__(self, printOn):
        self.printOn = printOn

        self.freeze_attrs()

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''
        advance the pass/fail automaton state given the current state and time.
        '''

        return

    @abc.abstractmethod
    def result(self):
        '''
        returns True iff all conditions passed
        '''
        return False

class FlightLimitsPFA(PassFailAutomaton):
    '''An automaton that checks the flight limits at each step'''

    AIRSPEED = 0
    ALPHA_LIMIT = 1
    BETA_LIMIT = 2
    NZ_LIMIT = 3
    PS_RATE_LIMIT = 4
    ALTITUDE_LIMIT = 5
    NUM = 6

    def __init__(self, printOn, flightLimits):
        self.flightLimits = flightLimits
        self.passed = True

        self.last_ps = None
        self.last_time = None

        PassFailAutomaton.__init__(self, printOn)

    def check(self, label, time, value, minVal, maxVal):
        'check if a value was out of bounds'

        assert minVal <= maxVal, "{} limits given in wrong order".format(label)

        if value < minVal:
            self.passed = False

            if self.printOn:
                print "Min {} limit exceeded ({} < {}) at time {:.3f}".format(label, value, minVal, time)
        elif value > maxVal:
            self.passed = False

            if self.printOn:
                print "Max {} limit exceeded ({} > {}) at time {:.3f}".format(label, value, maxVal, time)

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''advance the automaton state based on the current aircraft state'''

        airspeed = x_f16[0]
        self.check('airspeed', t, airspeed, self.flightLimits.vMin, self.flightLimits.vMax)

        alpha = rad2deg(x_f16[1])
        self.check('alpha', t, alpha, self.flightLimits.alphaMinDeg, self.flightLimits.alphaMaxDeg)

        beta = rad2deg(x_f16[2])
        self.check('beta', t, beta, -self.flightLimits.betaMaxDeg, self.flightLimits.betaMaxDeg)

        self.check('Nz', t, Nz, self.flightLimits.NzMin, self.flightLimits.NzMax)

        altitude = x_f16[11]
        self.check('altitude', t, altitude, self.flightLimits.altitudeMin, self.flightLimits.altitudeMax)

        # check ps rate limits using numerical derivative (requires two samples)
        if self.last_ps is not None:
            ps_rate_rad = (ps - self.last_ps) / (t - self.last_time)
            ps_rate_deg = rad2deg(ps_rate_rad)

            self.check('ps_rate', t, ps_rate_deg, -self.flightLimits.psMaxAccelDeg, self.flightLimits.psMaxAccelDeg)

        # update previous sample for next step
        self.last_ps = ps
        self.last_time = t

    @abc.abstractmethod
    def result(self):
        '''returns True if all conditions passed'''

        return self.passed
