"""
x8_sequencer.py — Test sequences and abort safety checker.

TestSequencer  generates attitude commands for standard test manoeuvres.
AbortChecker   monitors envelope and parameter health each tick.
"""

import math
import time
from typing import Optional

import numpy as np

from x8_mavlink import VehicleState
from x8_params  import X8Params


# ---------------------------------------------------------------------------
# Test sequences
# ---------------------------------------------------------------------------

class TestSequencer:
    """
    Returns desired [phi_d, theta_d, psi_d] in radians for each tick.

    Sequences:
        doublet        ±20° roll steps — first thing to run in SITL
        pitch_doublet  ±8°  pitch steps
        chirp          frequency sweep 0.1→2 Hz — needed for theta convergence
        cruise         wings-level hold — baseline comparison
    """

    SEQUENCES = ('doublet', 'pitch_doublet', 'chirp', 'cruise')

    def __init__(self, kind: str = 'doublet',  trim_theta_rad: float = 0.0):
        if kind not in self.SEQUENCES:
            raise ValueError(f"Unknown sequence '{kind}'. Choose: {self.SEQUENCES}")
        self.kind = kind
        self.trim_theta = trim_theta_rad
        self._t0: Optional[float] = None

    def start(self):
        self._t0 = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self._t0 is None:
            self.start()
        return time.monotonic() - self._t0

    def get_command(self, state: VehicleState) -> np.ndarray:
        """Returns [phi_d, theta_d, psi_d] in radians."""
        t = self.elapsed
        r = math.radians

        if self.kind == 'doublet':
            # ±20° roll doublet — standard step response
            if   t < 10.0:  phi_d = 0.0
            elif t < 25.0:  phi_d = r(20)
            elif t < 40.0:  phi_d = r(-20)
            else:           phi_d = 0.0
            return np.array([phi_d, self.trim_theta, state.psi])

        elif self.kind == 'pitch_doublet':
            # ±8° pitch doublet
            if   t < 10.0:  th_d = self.trim_theta
            elif t < 25.0:  th_d = self.trim_theta + r(8)
            elif t < 40.0:  th_d = self.trim_theta - r(4)
            else:           th_d = self.trim_theta
            return np.array([0.0, th_d, state.psi])

        elif self.kind == 'chirp':
            # Roll frequency sweep — drives persistent excitation for adaptation
            A        = r(15)
            f0, f1   = 0.1, 2.0
            T        = 30.0
            f        = f0 + (f1 - f0) * min(t / T, 1.0)
            phi_d    = A * math.sin(2 * math.pi * f * t)
            return np.array([phi_d, self.trim_theta, state.psi])

        else:  # cruise
            return np.array([0.0, self.trim_theta, state.psi])


    # need to add a lawnmower pattern here

# ---------------------------------------------------------------------------
# Abort checker
# ---------------------------------------------------------------------------

class AbortChecker:
    """
    Checks envelope and parameter health every tick.
    Returns None if safe, or a string reason if the controller should abort.

    Abort conditions:
        - |phi|   > roll_limit
        - |theta| > pitch_limit
        - |s|_∞   > s_limit  (sliding surface runaway)
        - airspeed out of [ias_min, ias_max]
        - any theta_hat element > theta_mult × nominal  (parameter blowup)
    """

    def __init__(self,
                 roll_limit_deg:  float = 55.0,
                 pitch_limit_deg: float = 35.0,
                 s_limit_deg_s:   float = 200.0,
                 ias_min:         float = 9.0,
                 ias_max:         float = 28.0,
                 theta_mult:      float = 2.0):
        self.roll_lim   = math.radians(roll_limit_deg)
        self.pitch_lim  = math.radians(pitch_limit_deg)
        self.s_lim      = math.radians(s_limit_deg_s)
        self.ias_min    = ias_min
        self.ias_max    = ias_max
        self.theta_mult = theta_mult

    def check(self,
              state:      VehicleState,
              s:          np.ndarray,
              theta_hat:  np.ndarray,
              theta_nom:  np.ndarray) -> Optional[str]:

        if abs(state.phi) > self.roll_lim:
            return f"Roll limit  φ={math.degrees(state.phi):.1f}°"

        if abs(state.theta) > self.pitch_lim:
            return f"Pitch limit θ={math.degrees(state.theta):.1f}°"

        if np.max(np.abs(s)) > self.s_lim:
            return f"Sliding surface |s|={math.degrees(float(np.max(np.abs(s)))):.1f} °/s"

        if state.airspeed < self.ias_min:
            return f"Airspeed low  {state.airspeed:.1f} m/s"

        if state.airspeed > self.ias_max:
            return f"Airspeed high {state.airspeed:.1f} m/s"

        drift = np.max(np.abs(theta_hat) / (np.abs(theta_nom) + 1e-9))
        if drift > self.theta_mult:
            return f"Parameter divergence {drift:.2f}× nominal"

        return None
