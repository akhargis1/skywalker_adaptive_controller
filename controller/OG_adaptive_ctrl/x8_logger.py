"""
x8_logger.py — Flight data logger.

Accumulates one row per control tick into memory.
Call save() on exit → compressed .npz file.
Use x8_plot.py to visualise the results.

Row schema (39 columns):
    t                        wall-clock time, s
    phi theta psi            measured attitude, rad
    p q r                    measured body rates, rad/s
    phi_d theta_d psi_d      reference attitude, rad
    s0 s1 s2                 sliding surface vector, rad/s
    e0 e1 e2                 attitude error, rad
    tau0 tau1 tau2           torque command, N·m
    delta_L delta_R          elevon deflections, rad
    th0..th9                 theta_hat (10 parameters)
    V_lyap                   Lyapunov proxy
    airspeed                 m/s
    alpha beta               rad
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from x8_mavlink import VehicleState


FIELDS = [
    't',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'phi_d', 'theta_d', 'psi_d',
    's0', 's1', 's2',
    'e0', 'e1', 'e2',
    'tau0', 'tau1', 'tau2',
    'delta_L', 'delta_R',
    'th0', 'th1', 'th2', 'th3', 'th4',
    'th5', 'th6', 'th7', 'th8', 'th9',
    'V_lyap',
    'airspeed',
    'alpha', 'beta',
]


class FlightLogger:

    def __init__(self, prefix: str = 'sitl_run'):
        self._rows:  list = []
        self._prefix = prefix
        self._t0     = time.monotonic()

    def record(self,
               state:   VehicleState,
               out:     dict):
        """Append one row. Call once per control tick."""
        t   = time.monotonic() - self._t0
        s   = out['s']
        e   = out['e']
        tau = out['tau']
        th  = out['theta_hat']

        self._rows.append([
            t,
            state.phi, state.theta, state.psi,
            state.p,   state.q,     state.r,
            out['q_d'][0], out['q_d'][1], out['q_d'][2],
            s[0],  s[1],  s[2],
            e[0],  e[1],  e[2],
            tau[0], tau[1], tau[2],
            out['delta_L'], out['delta_R'],
            *th[:10],
            out['V'],
            state.airspeed,
            state.alpha, state.beta,
        ])

    def save(self) -> Optional[Path]:
        if not self._rows:
            print("[LOG] No data to save.")
            return None
        arr  = np.array(self._rows, dtype=np.float64)
        path = Path(f"{self._prefix}_{int(time.time())}.npz")
        np.savez_compressed(path, data=arr, fields=np.array(FIELDS))
        print(f"[LOG] {len(self._rows)} rows → {path}")
        return path

    def __len__(self):
        return len(self._rows)
