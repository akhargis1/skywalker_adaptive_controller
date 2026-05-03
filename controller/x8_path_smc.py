"""
x8_path_smc.py — Path-following Sliding Mode Controller for Skywalker X8.

Implements the three-surface SMC derived from Frenet-Serret kinematics:

    s1 (cross-track)   → φ_cmd   (bank angle)
    s2 (along-track)   → T_cmd   (throttle 0–1)
    s3 (altitude)      → θ_cmd   (pitch angle, integrated)

All three are sent in a single SET_ATTITUDE_TARGET message (type_mask=0b00000111).
No TECS conflict. No RC override.

Coordinate convention (NED):
    x = North (m),  y = East (m),  z = Down (m, positive = below home)
    vx, vy, vz — NED velocities; vz positive = descending
    psi_r — path heading (0=North, CW positive)
    chi   — course angle from (vx, vy)
    kappa — signed curvature (rad/m): +1/R CW turn, -1/R CCW turn
    gamma — flight path angle (rad): positive = climbing

Error conventions:
    e_n — cross-track (m): positive = left of path (CCW from tangent)
    e_t — along-track (m): positive = ahead of virtual particle
    e_z — altitude (m): positive = ABOVE reference   [e_z = h - h_ref = -z_NED - h_ref]

Sliding surfaces:
    s1 = v_g sin χ̃ − κ v_d e_t + λ_n e_n
    s2 = v_g cos χ̃ − v_d(1 − κ e_n) + λ_t e_t
    s3 = v_g sin γ + λ_z e_z

Control laws:
    φ_cmd  = arctan[(−η_n sat(s1/Φ_n) + κv_d² cos χ̃ + κv_d ė_t − λ_n ė_n) / (g cos χ̃)]
    θ_cmd  = θ_cmd[k−1] + [(−η_z sat(s3/Φ_z) − λ_z ė_z) / (v_g cos γ)] · Δt
    T_cmd  = T_trim + K_s(−η_t sat(s2/Φ_t) + v_g sin χ̃ · χ̃̇ − κv_d ė_n − λ_t ė_t)

Stability guarantee (UUB):
    |e_n|_∞ ≤ Φ_n w̄_n / (η_n λ_n)
    |e_t|_∞ ≤ Φ_t w̄_t / (η_t λ_t)
    |e_z|_∞ ≤ Φ_z w̄_z / (η_z λ_z)

Usage
-----
    from trajectory_generation import LawnmowerTrajectory
    from x8_path_smc import PathSMC, PathSMCGains

    traj = LawnmowerTrajectory(altitude=100, airspeed=17, leg_length=400,
                               turn_radius=100, num_legs=4)
    smc  = PathSMC(traj)

    # inside 50 Hz loop (x, y origin-relative):
    out = smc.update(x, y, z, vx, vy, vz, phi=state.phi, t=t_ref)
    send_attitude_target(conn, roll_d=out.phi_cmd, pitch_d=out.theta_cmd,
                         yaw_d=state.psi, thrust=out.T_cmd, type_mask=0b00000111)
"""

import math
from dataclasses import dataclass

from trajectory_generation import LawnmowerTrajectory

G = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

@dataclass
class PathSMCGains:
    """
    Tuning parameters for the three-surface path-following SMC.

    Stability conditions:
        η_i > w̄_i    (switching gain must exceed disturbance bound)
        Φ_i > 0       (boundary layer must be positive)
        λ_i > 0       (convergence rate must be positive)

    Steady-state error bounds (UUB):
        |e_n| ≤ Φ_n w̄_n / (η_n λ_n)
        |e_t| ≤ Φ_t w̄_t / (η_t λ_t)
        |e_z| ≤ Φ_z w̄_z / (η_z λ_z)
    """
    # Convergence rates (1/s)
    lambda_n:    float = 0.5
    lambda_t:    float = 0.3
    lambda_z:    float = 0.5

    # Disturbance-rejection gains
    eta_n:       float = 1.0
    eta_t:       float = 1.0
    eta_z:       float = 1.0

    # Boundary layer thicknesses
    phi_n:       float = 2.0
    phi_t:       float = 0.5
    phi_z:       float = 0.3

    # Throttle parameters (Option A — parameter-free)
    T_trim:      float = 0.75              # cruise throttle fraction (tune to match actual cruise)
    K_scale:     float = 0.15             # throttle sensitivity (tune empirically)

    # Pitch trim and limits
    theta_trim:  float = math.radians(3)  # trim pitch for level flight
    theta_max:   float = math.radians(20)
    theta_min:   float = math.radians(-15)

    # Throttle limits
    T_min:       float = 0.1
    T_max:       float = 0.9

    # Output bank angle limit
    phi_max:     float = math.radians(45)

    # IIR derivative filter coefficient (0 = no update, 1 = no filtering)
    deriv_alpha: float = 0.3

    # e_t cap for s1 coupling term (m): prevents large schedule gaps from destabilising bank cmd
    e_t_cap:     float = 30.0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SMCOutput:
    """One-tick output from PathSMC.update()."""
    # Commands (feed directly into send_attitude_target)
    phi_cmd:   float   # bank angle (rad)
    theta_cmd: float   # pitch angle (rad, integrated state)
    T_cmd:     float   # throttle 0–1

    # Sliding surfaces (m/s) — for logging and tuning
    s1: float
    s2: float
    s3: float

    # Errors
    e_n:     float   # cross-track (m), positive = left of path
    e_t:     float   # along-track (m), positive = ahead of particle
    e_z:     float   # altitude (m), positive = above reference
    chi_err: float   # course error χ − ψ_r (rad)
    gamma:   float   # flight path angle (rad)

    # Reference trajectory (for logging)
    psi_r: float
    kappa: float
    x_r:   float
    y_r:   float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PathSMC:
    """
    Three-surface SMC for lawnmower path following.

    Instantiate once, then call update() at each control tick.
    Maintains state between ticks: integrated θ_cmd and filtered derivatives.
    """

    def __init__(self, trajectory: LawnmowerTrajectory,
                 gains: PathSMCGains = None):
        self.traj  = trajectory
        self.gains = gains if gains is not None else PathSMCGains()

        # Persistent state
        self._theta_cmd = self.gains.theta_trim
        self._en_dot_f  = 0.0
        self._et_dot_f  = 0.0
        self._ez_dot_f  = 0.0
        self._prev_t    = None

    def reset(self):
        """Reset integrator and filter state (call before a new mission)."""
        self._theta_cmd = self.gains.theta_trim
        self._en_dot_f  = 0.0
        self._et_dot_f  = 0.0
        self._ez_dot_f  = 0.0
        self._prev_t    = None

    def update(self,
               x: float,  y: float,  z: float,
               vx: float, vy: float, vz: float,
               phi: float,
               t: float) -> SMCOutput:
        """
        Compute SMC commands from current vehicle state.

        Parameters (NED, SI, origin-relative x/y)
        ----------
        x, y   : NED position relative to mission origin (m)
        z      : NED down (m), positive = below home
        vx, vy : NED horizontal velocity (m/s)
        vz     : NED vertical velocity (m/s), positive = descending
        phi    : current bank angle (rad)
        t      : trajectory reference time from nearest_t() (s)
        """
        g = self.gains

        # dt for θ integration
        if self._prev_t is None:
            dt = 0.02
        else:
            dt = _clamp(t - self._prev_t, 0.001, 0.1)
        self._prev_t = t

        # ------------------------------------------------------------------
        # 1. Reference trajectory
        # ------------------------------------------------------------------
        ref   = self.traj.query(t)
        x_r   = ref.x_ref
        y_r   = ref.y_ref
        h_ref = self.traj.altitude        # reference altitude AGL (m)
        psi_r = ref.psi_ref
        v_d   = ref.v_ref

        kappa = ref.psi_dot_ref / v_d if abs(v_d) > 0.1 else 0.0

        # ------------------------------------------------------------------
        # 2. Position errors (Frenet frame)
        # ------------------------------------------------------------------
        cpsi = math.cos(psi_r)
        spsi = math.sin(psi_r)
        dx   = x - x_r
        dy   = y - y_r

        e_t =  dx * cpsi + dy * spsi   # along-track
        e_n = -dx * spsi + dy * cpsi   # cross-track (CCW positive)

        h   = -z                        # altitude AGL (positive = above home)
        e_z = h - h_ref                 # altitude error (positive = above reference)

        # ------------------------------------------------------------------
        # 3. Kinematics
        # ------------------------------------------------------------------
        v_g   = max(math.hypot(vx, vy), 0.5)
        chi   = math.atan2(vy, vx)
        chi_e = _wrap_pi(chi - psi_r)
        gamma = math.atan2(-vz, v_g)   # positive = climbing

        # Velocity projections onto Frenet axes (direct, no finite diff)
        e_n_dot_raw = -spsi * vx + cpsi * vy          # normal component of v
        e_t_dot_raw =  cpsi * vx + spsi * vy - v_d  # tangential component - v_d
        e_z_dot_raw = -vz                              # altitude rate (positive = climbing)

        # IIR filter
        a = g.deriv_alpha
        self._en_dot_f = a * e_n_dot_raw + (1.0 - a) * self._en_dot_f
        self._et_dot_f = a * e_t_dot_raw + (1.0 - a) * self._et_dot_f
        self._ez_dot_f = a * e_z_dot_raw + (1.0 - a) * self._ez_dot_f
        e_n_dot = self._en_dot_f
        e_t_dot = self._et_dot_f
        e_z_dot = self._ez_dot_f

        # ------------------------------------------------------------------
        # 4. Sliding surfaces
        # ------------------------------------------------------------------
        # Cap e_t contribution to s1/phi_cmd: the coupling term -κv_d·e_t assumes
        # e_t is small; a large schedule gap (slow aircraft) otherwise saturates phi_cmd.
        e_t_s1 = _clamp(e_t, -g.e_t_cap, g.e_t_cap)

        s1 = v_g * math.sin(chi_e) - kappa * v_d * e_t_s1 + g.lambda_n * e_n
        s2 = v_g * math.cos(chi_e) - v_d * (1.0 - kappa * e_n) + g.lambda_t * e_t
        s3 = v_g * math.sin(gamma) + g.lambda_z * e_z

        # ------------------------------------------------------------------
        # 5. φ_cmd — bank angle (cross-track surface s1)
        # ------------------------------------------------------------------
        cos_chi = math.cos(chi_e)
        if abs(cos_chi) < 0.1:
            cos_chi = math.copysign(0.1, cos_chi)

        phi_num = (-g.eta_n * _sat(s1, g.phi_n)
                   + kappa * v_d ** 2 * cos_chi   # curvature feedforward
                   + kappa * v_d * e_t_dot        # coupling compensation
                   - g.lambda_n * e_n_dot)          # damping

        phi_cmd = math.atan2(phi_num, G * cos_chi)
        phi_cmd = _clamp(phi_cmd, -g.phi_max, g.phi_max)

        # ------------------------------------------------------------------
        # 6. θ_cmd — pitch angle (altitude surface s3, integrated)
        # ------------------------------------------------------------------
        cos_gam = max(math.cos(gamma), 0.1)
        theta_dot = (-g.eta_z * _sat(s3, g.phi_z) - g.lambda_z * e_z_dot) / (v_g * cos_gam)
        self._theta_cmd = _clamp(self._theta_cmd + theta_dot * dt,
                                 g.theta_min, g.theta_max)

        # ------------------------------------------------------------------
        # 7. T_cmd — throttle (along-track surface s2, Option A)
        # ------------------------------------------------------------------
        chi_e_dot = G / v_g * math.tan(phi) - kappa * v_d
        T_cmd = g.T_trim + g.K_scale * (
            -g.eta_t * _sat(s2, g.phi_t)
            + v_g * math.sin(chi_e) * chi_e_dot
            - kappa * v_d * e_n_dot
            - g.lambda_t * e_t_dot
        )
        T_cmd = _clamp(T_cmd, g.T_min, g.T_max)

        # ------------------------------------------------------------------
        # 8. Return
        # ------------------------------------------------------------------
        return SMCOutput(
            phi_cmd=phi_cmd, theta_cmd=self._theta_cmd, T_cmd=T_cmd,
            s1=s1, s2=s2, s3=s3,
            e_n=e_n, e_t=e_t, e_z=e_z, chi_err=chi_e, gamma=gamma,
            psi_r=psi_r, kappa=kappa, x_r=x_r, y_r=y_r,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sat(x: float, phi: float) -> float:
    if phi <= 0.0:
        return math.copysign(1.0, x)
    r = x / phi
    return r if abs(r) <= 1.0 else math.copysign(1.0, r)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Quick self-test  (python3 x8_path_smc.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    traj = LawnmowerTrajectory(altitude=100, airspeed=17, leg_length=400,
                               turn_radius=100, num_legs=4)
    smc  = PathSMC(traj)

    print("PathSMC self-test — on-trajectory snapshots\n")
    print(f"{'t(s)':>6}  {'e_n':>7}  {'e_t':>7}  {'e_z':>7}  "
          f"{'s1':>7}  {'s3':>7}  {'φ_cmd°':>8}  {'θ_cmd°':>8}  {'T_cmd':>6}")
    print("-" * 80)

    for tick in range(0, int(traj.total_time), 5):
        t     = float(tick)
        ref   = traj.query(t)
        psi_r = ref.psi_ref
        x, y  = ref.x_ref, ref.y_ref
        z     = ref.z_ref          # NED down = -100 m
        vx    = ref.v_ref * math.cos(psi_r)
        vy    = ref.v_ref * math.sin(psi_r)
        vz    = 0.0

        out = smc.update(x, y, z, vx, vy, vz, phi=0.0, t=t)

        print(f"{t:6.1f}  {out.e_n:+7.3f}  {out.e_t:+7.3f}  {out.e_z:+7.3f}  "
              f"{out.s1:+7.3f}  {out.s3:+7.3f}  "
              f"{math.degrees(out.phi_cmd):+8.2f}  "
              f"{math.degrees(out.theta_cmd):+8.2f}  "
              f"{out.T_cmd:6.3f}")

    print("\nExpect: e_n≈0, e_t≈0, e_z≈0, s1≈0, s3≈0 on straight legs.")
    print("φ_cmd non-zero on turns (curvature feedforward).")
    print("θ_cmd ≈ theta_trim (3°). T_cmd ≈ T_trim (0.75).")
