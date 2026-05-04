"""
x8_controller.py — CASMC controller core.

Contains only math — no MAVLink, no I/O, no threading.
Import this wherever you need the control law (runner, tests, HIL, etc.).

Call pattern (50 Hz loop):
    ctrl = X8Controller(params, gains)
    out  = ctrl.update(q_att, dq, q_d, dq_d, ddq_d, alpha, beta, airspeed)
    use  out['delta_L'], out['delta_R']  to command actuators
"""

import math
import numpy as np
from x8_params import X8Params, CASMCGains


# ---------------------------------------------------------------------------
# Parameter index map  (theta has 10 elements)
# ---------------------------------------------------------------------------
# 0  Ixx          roll inertia           kg·m²
# 1  Iyy          pitch inertia          kg·m²
# 2  Izz          yaw inertia            kg·m²
# 3  Ixz          cross inertia          kg·m²
# 4  Cl_α·qSb     effective roll-AoA     N·m/rad
# 5  Cm_α·qSc     effective pitch-stiff  N·m/rad
# 6  Cl_δ·qSb     effective roll ctrl    N·m/rad
# 7  Cm_δ·qSc     effective pitch ctrl   N·m/rad
# 8  Cn_β·qSb     effective yaw-slip     N·m/rad
# 9  K_drag·qSb   effective drag couple  N·m/rad


def _sat(s: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Element-wise saturation  sat(s / phi)."""
    return np.clip(s / np.where(phi > 1e-9, phi, 1e-9), -1.0, 1.0)


def _project(theta_hat: np.ndarray,
             theta_nom:  np.ndarray,
             frac: float) -> np.ndarray:
    """Hard-clamp each parameter to nominal ± frac, sign-safe."""
    lo = np.minimum(theta_nom * (1 - frac), theta_nom * (1 + frac))
    hi = np.maximum(theta_nom * (1 - frac), theta_nom * (1 + frac))
    return np.clip(theta_hat, lo, hi)


def build_regressor(q_att:    np.ndarray,
                    dq:       np.ndarray,
                    dq_r:     np.ndarray,
                    ddq_r:    np.ndarray,
                    alpha:    float,
                    beta:     float,
                    airspeed: float,
                    params:   X8Params) -> np.ndarray:
    """
    Build 3×10 regressor matrix Y such that:
        M * ddq_r + C * dq_r  ≈  Y(q, dq, dq_r, ddq_r, α, β, V) @ theta

    Rows:  [roll, pitch, yaw]
    Cols:  parameter index map above.
    """
    pr, qr, rr   = dq_r
    dpr, dqr, drr = ddq_r

    # Dynamic pressure scale relative to nominal
    q_dyn  = 0.5 * params.rho * max(airspeed, 8.0) ** 2
    sc     = q_dyn / params.q_nom          # rescales effective aero columns

    # --- Row 0: roll (L) ---
    y0     = np.zeros(10)
    y0[0]  =  dpr                          # Ixx * dp_r
    y0[3]  = -drr                          # -Ixz * dr_r
    y0[4]  =  sc * alpha                   # Cl_α effective
    #y0[6]  =  sc                           # Cl_δ effective (delta injected by mixer)
    y0[6]  =  0.0
                        
    # --- Row 1: pitch (M) ---
    y1     = np.zeros(10)
    y1[1]  =  dqr                          # Iyy * dq_r
    y1[3]  =  pr ** 2 - rr ** 2            # Ixz * (p² - r²)
    y1[5]  =  sc * alpha                   # Cm_α effective
    #y1[7]  =  sc                           # Cm_δ effective
    y1[7]  =  0.0

    # --- Row 2: yaw (N) ---
    y2     = np.zeros(10)
    y2[2]  =  drr                          # Izz * dr_r
    y2[3]  = -dpr                          # -Ixz * dp_r
    y2[8]  =  sc * beta                    # Cn_β effective
    y2[9]  =  sc * (pr - rr)              # K_drag coupling

    return np.vstack([y0, y1, y2])


def elevon_mixer(tau:      np.ndarray,
                 airspeed: float,
                 params:   X8Params) -> tuple:
    """
    Invert control effectiveness B(airspeed) to get physical elevon deflections.

    X8 elevon convention:
        delta_sym  = (L + R) / 2  →  pitch moment
        delta_diff = (L - R) / 2  →  roll moment

    Returns (delta_L, delta_R) in radians, clamped to ±elevon_limit.
    Yaw is not directly actuated (drag differential handled by adaptive law).
    """
    q_dyn = 0.5 * params.rho * max(airspeed, 8.0) ** 2
    S, b, c = params.Sref, params.wingspan, params.mac

    eff_roll  = q_dyn * S * b * params.Cl_delta
    eff_pitch = q_dyn * S * c * params.Cm_delta

    # Guard against near-zero effectiveness (e.g. very low airspeed)
    eff_roll  = eff_roll  if abs(eff_roll)  > 0.01 else math.copysign(0.01, eff_roll)
    eff_pitch = eff_pitch if abs(eff_pitch) > 0.01 else math.copysign(0.01, eff_pitch)

    delta_diff = tau[0] / eff_roll
    delta_sym  = tau[1] / eff_pitch

    lim = math.radians(params.elevon_limit_deg)
    #delta_L = float(np.clip(delta_sym + delta_diff, -lim, lim))
    #delta_R = float(np.clip(delta_sym - delta_diff, -lim, lim))
    #return delta_L, delta_R

    
    # Compute raw
    delta_L = delta_sym + delta_diff
    delta_R = delta_sym - delta_diff
    
    # Find max usage
    max_mag = max(abs(delta_L), abs(delta_R))
    
    # If exceeding limits → scale BOTH
    if max_mag > lim:
        scale = lim / max_mag
        delta_L *= scale
        delta_R *= scale
    
    return float(delta_L), float(delta_R)


class ReferenceModel:
    """2nd-order filter on raw attitude command → smooth q_d, dq_d, ddq_d."""

    def __init__(self, omega_n: float = 3.0, zeta: float = 0.9):
        self.omega_n = omega_n
        self.zeta    = zeta
        self.q_d     = np.zeros(3)
        self.dq_d    = np.zeros(3)

    def reset(self, q_init: np.ndarray):
        self.q_d  = q_init.copy()
        self.dq_d = np.zeros(3)

    def step(self, q_cmd: np.ndarray, dt: float):
        ddq_d     = (self.omega_n ** 2 * (q_cmd - self.q_d)
                     - 2 * self.zeta * self.omega_n * self.dq_d)
        self.dq_d = self.dq_d + ddq_d * dt
        self.q_d  = self.q_d  + self.dq_d * dt
        return self.q_d.copy(), self.dq_d.copy(), ddq_d.copy()


class X8Controller:
    """
    Composite Adaptive Sliding Mode Controller.

    Public API:
        ctrl = X8Controller(params, gains)
        ctrl.reset(q_init)          # call once before first update
        out  = ctrl.update(...)     # call at fixed rate (50 Hz recommended)
    """

    def __init__(self, params: X8Params = None, gains: CASMCGains = None):
        self.p  = params or X8Params()
        self.g  = gains  or CASMCGains()
        self.dt = 0.02

        self.theta_hat = self.p.theta_nominal.copy()
        self.refmod    = ReferenceModel(self.g.omega_n, self.g.zeta)

        self.trim_delta_sym = 0.0   # rad, set before first tick 
        self._prev_e   = np.zeros(3)
        self._ready    = False

    def reset(self, q_init: np.ndarray):
        """Seed reference model at current vehicle attitude before first tick."""
        self.refmod.reset(q_init)
        self._prev_e = np.zeros(3)
        self._ready  = True

    def update(self,
               q_att:    np.ndarray,   # measured [φ, θ, ψ]    rad
               dq:       np.ndarray,   # measured [p, q, r]    rad/s
               q_cmd:    np.ndarray,   # pilot/autopilot cmd   rad
               alpha:    float,        # angle of attack       rad
               beta:     float,        # sideslip              rad
               airspeed: float,        # true airspeed         m/s
               ) -> dict:

        if not self._ready:
            self.reset(q_att)

        dt = self.dt
        g  = self.g
        p  = self.p

        # 1. Reference trajectory
        q_d, dq_d, ddq_d = self.refmod.step(q_cmd, dt)

        # 2. Error
        e      = q_att - q_d
        e_dot  = (e - self._prev_e) / dt
        self._prev_e = e.copy()

        # 3. Sliding surface
        s = e_dot + g.Lambda @ e

        # 4. Reference body rates
        dq_r  = dq_d  - g.Lambda @ e
        ddq_r = ddq_d - g.Lambda @ e_dot

        # 5. Regressor
        Y = build_regressor(q_att, dq, dq_r, ddq_r, alpha, beta, airspeed, p)

        # 6. Control law
        tau_ff = Y @ self.theta_hat
        tau_sw = -(g.K @ _sat(s, g.phi))
        tau    = tau_ff + tau_sw

        # 7. Elevon mixer
        delta_L, delta_R = elevon_mixer(tau, airspeed, p)        
        lim = math.radians(p.elevon_limit_deg)
        delta_L = float(np.clip(delta_L + self.trim_delta_sym, -lim, lim))
        delta_R = float(np.clip(delta_R + self.trim_delta_sym, -lim, lim))
                   
        # 8. Adaptation  (σ-modification + projection)
        dtheta         = -g.Gamma @ (Y.T @ s) - g.sigma * self.theta_hat
        self.theta_hat = self.theta_hat + dtheta * dt
        self.theta_hat = _project(self.theta_hat, p.theta_nominal, g.proj_frac)

        # 9. Lyapunov proxy
        V = 0.5 * float(s @ s)

        return {
            'delta_L':   delta_L,
            'delta_R':   delta_R,
            'tau':       tau,
            's':         s,
            'e':         e,
            'q_d':       q_d,
            'dq_d':      dq_d,
            'theta_hat': self.theta_hat.copy(),
            'Y':         Y,
            'V':         V,
        }
