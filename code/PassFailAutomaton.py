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
        self.altitudeMax = 45000 #ft AGL
        self.NzMax = 9 # G's
        self.NzMin = -2 #G's
        self.psMaxAccelDeg = 500 # deg/s/s

        self.vMin = 300 # ft/s
        self.vMax = 2500 # ft/s
        self.alphaMinDeg = -10 # deg
        self.alphaMaxDeg = 45 # deg
        self.betaMaxDeg = 30 # deg

        self.check()

        self.freeze_attrs()

    def check(self):
        'check that flight limits are within model bounds'

        flightLimits = self

        assert not (flightLimits.vMin < 300 or flightLimits.vMax > 2500), \
            'flightLimits: Airspeed limits outside model limits (300 to 2500)'

        assert not (flightLimits.alphaMinDeg < -10 or flightLimits.alphaMaxDeg > 45), \
            'flightLimits: Alpha limits outside model limits (-10 to 45)'

        assert not (abs(flightLimits.betaMaxDeg) > 30), 'flightLimits: Beta limit outside model limits (30 deg)'

class PassFailAutomaton(Freezable):
    '''The parent class for a pass fail automaton... checks each state against the flight envelope limits'''

    def __init__(self):
        self.break_on_error = True

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

class MultiplePFA(PassFailAutomaton):
    '''A pass-fail automaton that combines multiple checks'''

    def __init__(self, pfa_list, break_on_error):
        self.pfa_list = pfa_list

        for pfa in pfa_list:
            pfa.break_on_error = break_on_error

        PassFailAutomaton.__init__(self)
        self.break_on_error = break_on_error

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''
        advance the pass/fail automaton state given the current state and time
        '''

        for pfa in self.pfa_list:
            pfa.advance(t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r)

    @abc.abstractmethod
    def result(self):
        '''
        returns True iff all conditions passed
        '''

        return all([pfa.result() for pfa in self.pfa_list])

class FlightLimitsPFA(PassFailAutomaton):
    '''An automaton that checks the flight limits at each step'''

    AIRSPEED = 0
    ALPHA_LIMIT = 1
    BETA_LIMIT = 2
    NZ_LIMIT = 3
    PS_RATE_LIMIT = 4
    ALTITUDE_LIMIT = 5
    NUM = 6

    def __init__(self, flightLimits, print_error=True):
        self.flightLimits = flightLimits
        self.passed = True

        self.last_ps = None
        self.last_time = None

        self.print_error = print_error

        PassFailAutomaton.__init__(self)

    def check(self, label, time, value, minVal, maxVal):
        'check if a value was out of bounds'

        assert minVal <= maxVal, "{} limits given in wrong order".format(label)

        if value < minVal:
            self.passed = False

            if self.print_error:
                self.print_error = False

                print "{} limit violated at time {:.2f} sec. {} < minimum ({})".format(
                    label, time, value, minVal)

        elif value > maxVal:
            self.passed = False

            if self.print_error:
                self.print_error = False

                print "{} limit violated at time {:.2f} sec. {} > maximum ({})".format(
                    label, time, value, maxVal)

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

class AirspeedPFA(PassFailAutomaton):
    '''An automaton that checks that airspeed is within some bound after some desired settling time'''

    def __init__(self, settling_time, setpoint, percent, print_error=True):
        self.settling_time = settling_time
        self.min = setpoint - setpoint * (percent / 100.0)
        self.max = setpoint + setpoint * (percent / 100.0)

        self.print_error = print_error
        self.passed = True

        PassFailAutomaton.__init__(self)

    def check(self, label, time, value, minVal, maxVal):
        'check if a value was out of bounds'

        assert minVal <= maxVal, "{} limits given in wrong order".format(label)

        if value < minVal:
            self.passed = False

            if self.print_error:
                self.print_error = False

                print "{} limit violated at time {:.2f} sec. {} < minimum ({})".format(
                    label, time, value, minVal)

        elif value > maxVal:
            self.passed = False

            if self.print_error:
                self.print_error = False

                print "{} limit violated at time {:.2f} sec. {} > maximum ({})".format(
                    label, time, value, maxVal)

    @abc.abstractmethod
    def advance(self, t, x_f16, autopilot_state, xd, u, Nz, ps, Ny_r):
        '''advance the automaton state based on the current aircraft state'''

        if t >= self.settling_time:
            airspeed = x_f16[0]

            self.check('airspeed', t, airspeed, self.min, self.max)

    @abc.abstractmethod
    def result(self):
        '''returns True if all conditions passed'''

        return self.passed
