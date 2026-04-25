"""
x8_params.py — Skywalker X8 airframe parameters and controller gains.

Single source of truth. Every other file imports from here.
Swap in your real values from ArduPilot params / CAD / wind tunnel.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class X8Params:
    # --- Mass & geometry ---
    mass:      float = 4.5       # kg
    wingspan:  float = 2.12      # m
    mac:       float = 0.3571      # mean aerodynamic chord, m
    Sref:      float = 0.75      # wing reference area, m²

    # --- Inertia tensor (body frame, nominal CG) ---
    Ixx: float = 0.45           # kg·m²  roll
    Iyy: float = 0.325           # kg·m²  pitch
    Izz: float = 0.75          # kg·m²  yaw
    Ixz: float = 0.06           # kg·m²  flying-wing cross term

    # TODO - UNSURE ON BELOW VALUES - may be able to extract from logs or the paper 
    # these may be in the paper, just need to be lined up correctly potentiallly
    # --- Stability derivatives (per radian) ---
    #Cl_alpha: float =  0.00      # roll due to AoA (symmetric X8 ≈ 0)
    #Cm_alpha: float = -0.38      # pitch stiffness  (negative = stable)
    #Cl_delta: float =  0.22      # roll per differential elevon
    #Cm_delta: float = -0.65      # pitch per symmetric elevon
    #Cn_beta:  float =  0.048     # yaw due to sideslip (drag asymmetry)
    #K_drag:   float =  0.031     # roll-yaw drag coupling

    Cl_alpha: float =  0.000     # roll due to AoA — symmetric X8, effectively zero
    Cm_alpha: float = -0.380     # pitch stiffness (estimated — cma=0 in SDF)
    Cl_delta: float =  0.625     # roll per rad differential elevon  [SDF-derived]
    #Cl_delta: float =  0.0     # roll per rad differential elevon  [SDF-derived]
    Cm_delta: float = -1.339     # pitch per rad symmetric elevon    [SDF-derived]
    #Cm_delta: float =  0.0     # pitch per rad symmetric elevon    [SDF-derived]
    Cn_beta:  float =  0.342     # yaw per rad sideslip via winglets  [SDF-derived]
    K_drag:   float =  0.031     # roll-yaw drag coupling (retain estimate)

    #Cl_alpha: float =  0.000     # roll due to AoA — symmetric X8, effectively zero
    #Cm_alpha: float =  0.000    # pitch stiffness (estimated — cma=0 in SDF)
    #Cl_delta: float = 0.000     # roll per rad differential elevon  [SDF-derived]
    #Cl_delta: float =  0.0     # roll per rad differential elevon  [SDF-derived]
    #Cm_delta: float = -1.339     # pitch per rad symmetric elevon    [SDF-derived]
    #Cm_delta: float =  0.0     # pitch per rad symmetric elevon    [SDF-derived]
    #Cn_beta:  float =  0.000     # yaw per rad sideslip via winglets  [SDF-derived]
    #K_drag:   float =  0.000     # roll-yaw drag coupling (retain estimate)
    
    # --- Nominal flight condition ---
    rho:   float = 1.225         # air density, kg/m³
    V_nom: float = 17.0          # cruise airspeed, m/s

    # --- Actuator limits ---
    elevon_limit_deg: float = 30.0   # ± physical travel

    @property
    def q_nom(self) -> float:
        """Nominal dynamic pressure."""
        return 0.5 * self.rho * self.V_nom ** 2

    @property
    def theta_nominal(self) -> np.ndarray:
        """
        Ground-truth parameter vector at nominal flight condition.
        10 elements — see controller for index map.
        """
        q = self.q_nom
        S, b, c = self.Sref, self.wingspan, self.mac
        return np.array([
            self.Ixx,
            self.Iyy,
            self.Izz,
            self.Ixz,
            q * S * b * self.Cl_alpha,
            q * S * c * self.Cm_alpha,
            q * S * b * self.Cl_delta,
            q * S * c * self.Cm_delta,
            q * S * b * self.Cn_beta,
            q * S * b * self.K_drag,
        ])


@dataclass
class CASMCGains:
    # Sliding surface bandwidth  s = ė + Λ e
    Lambda: np.ndarray = field(
        default_factory=lambda: np.diag([2.0, 1.2, 0.8])
    )

    # Adaptive learning rates
    Gamma: np.ndarray = field(
        default_factory=lambda: np.diag([
            0.05, 0.05, 0.05, 0.02,   # inertia terms
            0.10, 0.10,                # aero stiffness
            0.15, 0.15,                # control effectiveness
            0.08, 0.04,                # yaw / drag
        ])
    )

    # Sliding robustness gain  (must exceed worst-case disturbance)
    K: np.ndarray = field(
        default_factory=lambda: np.diag([0.4, 0.3, 0.2])
    )

    # Boundary layer  (replaces sgn with sat — kills chatter)
    phi: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.04, 0.03])
    )

    # σ-modification  (prevents parameter drift in low-excitation flight)
    sigma: float = 0.005

    # Projection bound  (keep θ̂ within ± frac of nominal)
    proj_frac: float = 0.60

    # Reference model  (2nd-order filter on attitude command)
    omega_n: float = 3.0    # rad/s
    zeta:    float = 0.9
