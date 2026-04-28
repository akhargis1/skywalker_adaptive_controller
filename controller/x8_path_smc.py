"""
x8_path_smc.py — Path-following Sliding Mode Controller for Skywalker X8.

Implements the three-surface SMC derived from Frenet-Serret kinematics:

    s1 (cross-track)   → φ_cmd  (bank angle, via SET_ATTITUDE_TARGET)
    s2 (along-track)   → v_cmd  (airspeed, via DO_CHANGE_SPEED)
    s3 (vertical)      → z_cmd  (NED altitude, via SET_POSITION_TARGET_LOCAL_NED)

Coordinate convention (NED throughout):
    x = North (m),  y = East (m),  z = Down (m)   — positive z = below home
    vx = North vel, vy = East vel, vz = Down vel   — positive vz = descending
    psi_r = path heading (rad, 0=North, increasing CW)
    chi   = course angle computed from (vx, vy)
    kappa = signed path curvature (rad/m): +1/R = CW turn, -1/R = CCW turn

Error conventions (Frenet frame):
    e_t — along-track: positive when UAV is AHEAD of virtual particle
    e_n — cross-track: positive in the n̂ direction = 90° CCW from path tangent
          (at psi_r=0: n̂ points East, so e_n>0 when UAV is East of northward path)
    e_z — vertical: e_z = z_NED - z_r, positive when UAV is BELOW reference

Equations of motion (disturbance-free, Frenet-Serret):
    ė_n = v_g sin(χ̃) + κ v_d e_t
    ė_t = v_g cos(χ̃) - v_d(1 - κ e_n)
    ė_z = vz               (= -v_g sin γ, when ż_r = 0)

Control laws:
    φ_cmd = arctan[(−η₁ sat(s₁/Φ₁) + κ v_d v_g cos χ̃ − κ v_d ė_t − λ₁ ė_n)
                   / (g cos χ̃)]
    v_cmd = [v_d(1 − κ e_n) − λ₂ e_t − η₂ sat(s₂/Φ₂)] / cos χ̃  / cos φ
    z_cmd = z_r − [η₃ sat(s₃/Φ₃) + λ₃ e_z] / λ₃     (NED down coord)

Stability guarantee (Lyapunov):
    |e_n|_∞ ≤ Φ₁ w̄_n / (η₁ λ₁)
    |e_t|_∞ ≤ Φ₂ w̄_t / (η₂ λ₂)
    |e_z|_∞ ≤ Φ₃ w̄_z / (η₃ λ₃)

Usage
-----
    from trajectory_generation import LawnmowerTrajectory
    from x8_path_smc import PathSMC, PathSMCGains

    traj = LawnmowerTrajectory(altitude=100, airspeed=17, leg_length=200,
                               turn_radius=40, num_legs=4)
    smc  = PathSMC(traj)                    # default gains
    # or:
    g = PathSMCGains(lambda_n=0.6, eta_n=3.0)
    smc = PathSMC(traj, gains=g)

    # inside 50 Hz loop:
    out = smc.update(x, y, z, vx, vy, vz, phi=state.phi, t=elapsed)
    send_attitude_target(conn, roll_d=out.phi_cmd, pitch_d=0.0, yaw_d=state.psi)
    send_airspeed_command(conn, out.v_cmd)
    send_altitude_command(conn, out.z_cmd)
"""

import math
from dataclasses import dataclass, field

from trajectory_generation import LawnmowerTrajectory

G = 9.81   # m/s²


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

@dataclass
class PathSMCGains:
    """
    Tuning parameters for the three-surface path-following SMC.

    Stability conditions:
        η_i > w̄_i         (switching gain must exceed disturbance bound)
        Φ_i > 0            (boundary layer must be positive)
        λ_i > 0            (convergence rate must be positive)

    Steady-state error bounds (UUB):
        |e_n| ≤ Φ_n w̄_n / (η_n λ_n)
        |e_t| ≤ Φ_t w̄_t / (η_t λ_t)
        |e_z| ≤ Φ_z w̄_z / (η_z λ_z)
    """
    # Desired airspeed (m/s) — virtual particle speed; matches LawnmowerTrajectory
    v_d:      float = 17.0

    # Convergence rates (1/s) — larger = faster convergence, more aggressive
    lambda_n: float = 0.5    # cross-track surface s1
    lambda_t: float = 0.3    # along-track surface s2
    lambda_z: float = 0.5    # vertical surface s3

    # Disturbance-rejection gains (m/s²) — must exceed bounded wind disturbances
    eta_n:    float = 1.0    # cross-track (raised from 2.0: max φ_cmd ≈ 39° on straights)
    eta_t:    float = 1.0    # along-track (reduced: less critical with nearest_t reference)
    eta_z:    float = 1.0    # vertical

    # Boundary layer thicknesses (m/s) — larger = smoother but more steady-state error
    phi_n:    float = 2.0    # s1 boundary layer (scaled with eta_n to preserve UUB bound)
    phi_t:    float = 0.5    # s2 boundary layer
    phi_z:    float = 0.3    # s3 boundary layer

    # Output saturation limits
    phi_max:  float = math.radians(45)   # max bank angle magnitude (rad)
    v_min:    float = 12.0               # min airspeed command (m/s)
    v_max:    float = 25.0               # max airspeed command (m/s)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SMCOutput:
    """One-tick output from PathSMC.update()."""
    # Commands to send to ArduPilot
    phi_cmd:   float    # bank angle command (rad); feed to send_attitude_target
    v_cmd:     float    # airspeed command (m/s); feed to send_airspeed_command
    z_cmd:     float    # NED down coordinate target (m); feed to send_altitude_command

    # Sliding surface values (m/s) — useful for logging and gain tuning
    s1:        float    # cross-track surface
    s2:        float    # along-track surface
    s3:        float    # vertical surface

    # Errors (m, m, m, rad)
    e_n:       float    # cross-track (m), positive = East of northward path
    e_t:       float    # along-track (m), positive = ahead of virtual particle
    e_z:       float    # vertical NED (m), positive = below reference
    chi_err:   float    # course error χ − ψ_r (rad)

    # Reference trajectory quantities (for logging)
    psi_r:     float    # reference heading (rad)
    kappa:     float    # path curvature (rad/m)
    x_r:       float    # reference North (m)
    y_r:       float    # reference East (m)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PathSMC:
    """
    Three-surface Sliding Mode Controller for lawnmower path following.

    Instantiate once, then call update() at each control tick.
    Contains no state between ticks (surfaces are computed, not integrated).
    """

    def __init__(self, trajectory: LawnmowerTrajectory,
                 gains: PathSMCGains = None):
        self.traj  = trajectory
        self.gains = gains if gains is not None else PathSMCGains()

    def update(self,
               x: float,  y: float,  z: float,
               vx: float, vy: float, vz: float,
               phi: float,
               t: float) -> SMCOutput:
        """
        Compute SMC commands from current vehicle state and elapsed time.

        Parameters (NED, SI)
        ----------
        x, y, z    : NED position (m); z positive = below home
        vx, vy, vz : NED velocity (m/s); vz positive = descending
        phi        : current bank angle (rad)
        t          : elapsed mission time (s), same clock as trajectory.query()

        Returns
        -------
        SMCOutput with phi_cmd, v_cmd, z_cmd, surfaces, and errors.
        """
        g = self.gains

        # ------------------------------------------------------------------
        # 1. Reference trajectory
        # ------------------------------------------------------------------
        ref   = self.traj.query(t)
        x_r   = ref.x_ref
        y_r   = ref.y_ref
        z_r   = ref.z_ref        # NED down = -altitude, negative number
        psi_r = ref.psi_ref      # path heading (rad)

        # Signed curvature: ψ̇_ref = ±V/R; κ = ψ̇_ref / v_d
        kappa = ref.psi_dot_ref / g.v_d if abs(g.v_d) > 0.1 else 0.0

        # ------------------------------------------------------------------
        # 2. Frenet-frame errors
        # ------------------------------------------------------------------
        # Tangent unit vector: t̂ = (cos ψ_r, sin ψ_r) in NED (N, E)
        # Normal unit vector:  n̂ = (-sin ψ_r, cos ψ_r) — 90° CCW from t̂
        cpsi, spsi = math.cos(psi_r), math.sin(psi_r)
        dx = x - x_r
        dy = y - y_r

        e_t =  dx * cpsi + dy * spsi    # along-track (ahead = positive)
        e_n = -dx * spsi + dy * cpsi    # cross-track (n̂ direction = positive)
        e_z =  z - z_r                  # vertical NED (below reference = positive)

        # ------------------------------------------------------------------
        # 3. Derived kinematics
        # ------------------------------------------------------------------
        v_g   = max(math.hypot(vx, vy), 0.5)   # horizontal ground speed (m/s)
        chi   = math.atan2(vy, vx)             # course angle (NED, 0=North, CW+)
        chi_e = _wrap_pi(chi - psi_r)          # course error χ̃

        # Nominal (disturbance-free) error rates from Frenet-Serret kinematics
        e_n_dot = v_g * math.sin(chi_e) + kappa * g.v_d * e_t
        e_t_dot = v_g * math.cos(chi_e) - g.v_d * (1.0 - kappa * e_n)

        # ------------------------------------------------------------------
        # 4. Sliding surfaces
        # ------------------------------------------------------------------
        s1 = e_n_dot + g.lambda_n * e_n   # cross-track (m/s)
        s2 = e_t_dot + g.lambda_t * e_t   # along-track (m/s)
        s3 = vz      + g.lambda_z * e_z   # vertical (m/s); ż_r = 0 (const alt)

        # ------------------------------------------------------------------
        # 5. Bank angle command (φ_cmd) — primary lateral control
        # ------------------------------------------------------------------
        # Guard cos(χ̃): equation degrades near ±90° cross-track
        cos_chi = math.cos(chi_e)
        if abs(cos_chi) < 0.087:                       # 5° guard
            cos_chi = math.copysign(0.087, cos_chi)

        # Curvature feedforward + coupling compensation + SMC term
        phi_num = (-g.eta_n * _sat(s1, g.phi_n)
                   + kappa * g.v_d * v_g * cos_chi     # curvature feedforward
                   - kappa * g.v_d * e_t_dot           # coupling compensation
                   - g.lambda_n * e_n_dot)             # damping

        phi_cmd = math.atan2(phi_num, G * cos_chi)
        phi_cmd = _clamp(phi_cmd, -g.phi_max, g.phi_max)

        # ------------------------------------------------------------------
        # 6. Airspeed command (v_cmd) — along-track control
        # ------------------------------------------------------------------
        v_cmd_raw = (g.v_d * (1.0 - kappa * e_n)
                     - g.lambda_t * e_t
                     - g.eta_t * _sat(s2, g.phi_t)) / cos_chi

        # Lift-loss compensation: in a banked turn, load factor = 1/cos(φ)
        cos_phi = max(abs(math.cos(phi)), 0.087)
        v_cmd   = _clamp(v_cmd_raw / cos_phi, g.v_min, g.v_max)

        # ------------------------------------------------------------------
        # 7. Altitude command (z_cmd) — vertical control
        # ------------------------------------------------------------------
        # z_cmd is a NED down coordinate (m); lower value = higher altitude.
        # Derived by setting ṡ₃ = −η₃ sat(s₃/Φ₃) and solving for the altitude
        # setpoint that drives TECS to achieve that surface dynamics.
        # See module docstring for stability analysis.
        z_cmd = z_r - (g.eta_z * _sat(s3, g.phi_z) + g.lambda_z * e_z) / g.lambda_z

        # ------------------------------------------------------------------
        # 8. Return
        # ------------------------------------------------------------------
        return SMCOutput(
            phi_cmd=phi_cmd, v_cmd=v_cmd, z_cmd=z_cmd,
            s1=s1, s2=s2, s3=s3,
            e_n=e_n, e_t=e_t, e_z=e_z, chi_err=chi_e,
            psi_r=psi_r, kappa=kappa, x_r=x_r, y_r=y_r,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sat(x: float, phi: float) -> float:
    """Scalar saturation function sat(x / phi)."""
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
    import numpy as np

    traj = LawnmowerTrajectory(altitude=100, airspeed=17, leg_length=200,
                               turn_radius=40, num_legs=4)
    smc  = PathSMC(traj)

    print("PathSMC self-test — simulating 1 Hz snapshots along nominal trajectory\n")
    print(f"{'t(s)':>6}  {'e_n':>7}  {'e_t':>7}  {'e_z':>7}  "
          f"{'s1':>7}  {'s2':>7}  {'phi_cmd_deg':>12}  {'v_cmd':>7}")
    print("-" * 75)

    for tick in range(0, int(traj.total_time), 5):
        t = float(tick)
        ref = traj.query(t)
        # Nominal state: vehicle exactly on trajectory, no error
        x, y = ref.x_ref, ref.y_ref
        z    = ref.z_ref
        psi_r = ref.psi_ref
        # Velocity from heading + nominal airspeed
        vx   = 17.0 * math.cos(psi_r)
        vy   = 17.0 * math.sin(psi_r)
        vz   = 0.0

        out = smc.update(x, y, z, vx, vy, vz, phi=0.0, t=t)

        print(f"{t:6.1f}  {out.e_n:+7.3f}  {out.e_t:+7.3f}  {out.e_z:+7.3f}  "
              f"{out.s1:+7.3f}  {out.s2:+7.3f}  "
              f"{math.degrees(out.phi_cmd):+12.2f}  {out.v_cmd:7.2f}")

    print("\nOn-trajectory: e_n, e_t, e_z should be ≈ 0.")
    print("phi_cmd on turns should be non-zero (curvature feedforward).")
    print("v_cmd should be close to 17.0 m/s.")
