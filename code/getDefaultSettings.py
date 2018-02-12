'''
Stanley Bak
F-16 GCAS get_default_settings
'''

from util import Freezable

def getDefaultSettings():
    'return a tuple of settings'

    return FlightLimits(), CtrlLimits()

class FlightLimits(Freezable):
    'Flight Limits (for pass-fail conditions)'

    def __init__(self):
        self.altitudeMin = 0 # ft AGL
        self.altitudeMax = 10000 #ft AGL
        self.maneuverTime = 15 # seconds
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

class CtrlLimits(Freezable):
    'Control Limits'

    def __init__(self):
        self.ThrottleMax = 1 # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        self.MaxBankDeg = 60 # For turning maneuvers
        self.NzMax = 6
        self.NzMin = -1

        self.freeze_attrs()
