"""
trajectory_generation.py — Lawnmower coverage trajectory for Skywalker X8.

LawnmowerTrajectory generates a time-parameterized reference path of alternating
North/South straight legs connected by 180-degree coordinated turns.  Given elapsed
time it returns the reference position, heading, and curvature feedforward needed by
the outer-loop path-following SMC.

All positions are in a local NED frame (x=North, y=East, z=Down) with the origin at
the trajectory start point.  Headings follow NED convention: 0=North, increasing CW.

Turn geometry
-------------
The turn center is always R to the East of the leg endpoint.  The entry point is
therefore directly South of the center (theta = -pi/2 in NED position angle).

After a North leg → CW turn  (psi: 0 → pi,  sign = +1)
After a South leg → CCW turn (psi: pi → 0,   sign = -1)

Position on circle parametrised by NED position angle theta (CW from North):
    x = cx + R * cos(theta)
    y = cy + R * sin(theta)
    theta(dt) = -pi/2 + sign * omega * dt   where omega = V/R
    psi(dt)   = psi_start + sign * omega * dt
"""

import argparse
import bisect
import math
from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    x_ref:       float  # North (m)
    y_ref:       float  # East (m)
    z_ref:       float  # Down (m), = -altitude
    psi_ref:     float  # heading (rad), NED convention
    psi_dot_ref: float  # curvature feedforward (rad/s): signed V/R on turns, 0 on straights
    v_ref:       float  # reference airspeed (m/s)
    segment:     str    # 'straight' | 'turn' | 'done'


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LawnmowerTrajectory:
    """
    Inputs
    ------
    altitude   : float  — flight altitude AGL (m)
    airspeed   : float  — nominal cruise airspeed (m/s)
    leg_length : float  — length of each straight leg (m)
    turn_radius: float  — radius of each 180-degree turn (m)
    num_legs   : int    — number of straight legs (default 4); must be >= 1
    loop       : bool  — if True, query() wraps t modulo total_time so the pattern
                         repeats indefinitely; segment is tagged 'straight'/'turn' as normal.
                         Use t_end externally to stop.

    Pattern (NED local frame, x=North, y=East)
    ------------------------------------------
        (0,  0) ──── North ────► (L,  0)
                                       ) CW 180°, centre (L, R)
        (0, 2R) ◄─── South ──── (L, 2R)
        (                              centre (0, 3R), CCW 180°
        (0, 4R) ──── North ────► (L, 4R)  ...

    Strip spacing = 2R (turn diameter).

    When loop=True the trajectory restarts from (0,0) heading North after each full
    pass.  The position is discontinuous at the wrap boundary; the outer-loop SMC
    will naturally start tracking the first segment again.
    """

    def __init__(self,
                 altitude:       float,
                 airspeed:       float,
                 leg_length:     float,
                 turn_radius:    float,
                 num_legs:       int   = 4,
                 loop:           bool  = False,
                 runway_length:  float = 0.0):

        if num_legs < 1:
            raise ValueError("num_legs must be >= 1")

        self.altitude       = altitude
        self.airspeed       = airspeed
        self.leg_length     = leg_length
        self.turn_radius    = turn_radius
        self.num_legs       = num_legs
        self.loop           = loop
        self.runway_length  = runway_length

        self._z_ref        = -altitude
        self._t_leg        = leg_length / airspeed
        self._t_turn       = math.pi * turn_radius / airspeed
        self._omega        = airspeed / turn_radius  # unsigned angular rate (rad/s)

        self._segments: List[dict] = []
        self._t_starts: List[float] = []
        self._build_segments()

    # ------------------------------------------------------------------
    # Segment precomputation
    # ------------------------------------------------------------------

    def _build_segments(self):
        t = 0.0
        x, y = 0.0, 0.0

        if self.runway_length > 0.0:
            t_run = self.runway_length / self.airspeed
            self._t_starts.append(t)
            self._segments.append({
                'type':    'straight',
                't_start': t,
                't_end':   t + t_run,
                'x0': x, 'y0': y,
                'psi': 0.0,
                'dx':  1.0,
                'length': self.runway_length,
            })
            t += t_run
            x += self.runway_length

        for leg_idx in range(self.num_legs):
            heading_north = (leg_idx % 2 == 0)
            psi = 0.0 if heading_north else math.pi
            dx  = 1.0 if heading_north else -1.0

            # straight leg
            self._t_starts.append(t)
            self._segments.append({
                'type':    'straight',
                't_start': t,
                't_end':   t + self._t_leg,
                'x0': x, 'y0': y,
                'psi': psi,
                'dx':  dx,
                'length': self.leg_length,
            })
            t += self._t_leg
            x += self.leg_length * dx

            # 180-degree turn (skip after last leg)
            if leg_idx < self.num_legs - 1:
                # center is always R to the East of the leg endpoint
                cx = x
                cy = y + self.turn_radius
                # CW after North legs, CCW after South legs
                sign = 1 if heading_north else -1

                self._t_starts.append(t)
                self._segments.append({
                    'type':      'turn',
                    't_start':   t,
                    't_end':     t + self._t_turn,
                    'cx': cx, 'cy': cy,
                    'R':         self.turn_radius,
                    'psi_start': psi,
                    'sign':      sign,
                })
                t += self._t_turn
                y += 2.0 * self.turn_radius

        self._t_total = t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, t: float) -> TrajectoryPoint:
        """Return the reference trajectory point at elapsed time t (seconds).

        When loop=True the trajectory repeats indefinitely; t is wrapped modulo
        total_time so the pattern restarts from (0,0) heading North each cycle.
        When loop=False and t >= total_time, segment is 'done'.
        """
        if t >= self._t_total:
            if self.loop:
                t = t % self._t_total
            else:
                last = self._segments[-1]
                if last['type'] == 'straight':
                    return TrajectoryPoint(
                        x_ref=last['x0'] + self.leg_length * last['dx'],
                        y_ref=last['y0'],
                        z_ref=self._z_ref,
                        psi_ref=last['psi'],
                        psi_dot_ref=0.0,
                        v_ref=self.airspeed,
                        segment='done',
                    )
                else:
                    return TrajectoryPoint(
                        x_ref=last['cx'],
                        y_ref=last['cy'] + self.turn_radius,
                        z_ref=self._z_ref,
                        psi_ref=_wrap_to_pi(last['psi_start'] + last['sign'] * math.pi),
                        psi_dot_ref=0.0,
                        v_ref=self.airspeed,
                        segment='done',
                    )

        idx = max(0, bisect.bisect_right(self._t_starts, t) - 1)
        seg = self._segments[idx]
        dt  = t - seg['t_start']

        if seg['type'] == 'straight':
            frac = dt / (seg['t_end'] - seg['t_start'])
            return TrajectoryPoint(
                x_ref=seg['x0'] + frac * seg['length'] * seg['dx'],
                y_ref=seg['y0'],
                z_ref=self._z_ref,
                psi_ref=seg['psi'],
                psi_dot_ref=0.0,
                v_ref=self.airspeed,
                segment='straight',
            )

        else:  # turn
            sign  = seg['sign']
            # theta: NED position angle (CW from North), entry always at -pi/2
            theta = -math.pi / 2.0 + sign * self._omega * dt
            x_ref = seg['cx'] + self.turn_radius * math.cos(theta)
            y_ref = seg['cy'] + self.turn_radius * math.sin(theta)
            psi   = _wrap_to_pi(seg['psi_start'] + sign * self._omega * dt)
            return TrajectoryPoint(
                x_ref=x_ref,
                y_ref=y_ref,
                z_ref=self._z_ref,
                psi_ref=psi,
                psi_dot_ref=sign * self._omega,
                v_ref=self.airspeed,
                segment='turn',
            )

    def cross_track_error(self, x: float, y: float, psi: float, t: float):
        """
        Signed cross-track error and heading error.

        Returns
        -------
        e_ct  : float  — lateral deviation (m); positive = right of reference path
        e_psi : float  — wrap_to_pi(psi - psi_ref) (rad)
        """
        ref = self.query(t)
        if ref.segment == 'done':
            return 0.0, 0.0

        t_eff = (t % self._t_total) if self.loop else t
        idx = max(0, bisect.bisect_right(self._t_starts, t_eff) - 1)
        seg = self._segments[idx]

        if seg['type'] == 'straight':
            # perpendicular (East) offset, signed by heading direction
            # North leg (dx=+1): right = East → positive e_ct when y > y0
            # South leg (dx=-1): right = West → positive e_ct when y < y0
            e_ct = (y - seg['y0']) * seg['dx']
        else:
            # signed radial error: positive = outside the arc
            dist = math.hypot(x - seg['cx'], y - seg['cy'])
            e_ct = dist - seg['R']

        e_psi = _wrap_to_pi(psi - ref.psi_ref)
        return e_ct, e_psi

    def nearest_t(self, x: float, y: float) -> float:
        """
        Return the trajectory time t* of the point on the path nearest to (x, y).

        Projects (x, y) onto every segment and returns the t with minimum
        squared distance. Handles straight legs (perpendicular projection) and
        semicircular arcs (angular projection) correctly.

        Used in place of wall-clock elapsed time so the virtual particle always
        stays near the aircraft rather than running away.
        """
        best_t, best_d2 = 0.0, float('inf')
        for seg in self._segments:
            t_proj, d2 = self._project_onto_segment(seg, x, y)
            if d2 < best_d2:
                best_t, best_d2 = t_proj, d2
        return best_t

    def _project_onto_segment(self, seg: dict, x: float, y: float):
        """Return (t_proj, squared_distance) for the projection of (x,y) onto seg."""
        if seg['type'] == 'straight':
            return self._project_straight(seg, x, y)
        else:
            return self._project_turn(seg, x, y)

    def _project_straight(self, seg: dict, x: float, y: float):
        # Leg travels in direction (dx, 0): dx=+1 for North, -1 for South
        dx   = seg['dx']
        x0   = seg['x0']
        y0   = seg['y0']
        # Along-track distance from leg start (dot product with unit direction)
        s = (x - x0) * dx          # (y component contributes 0)
        s = max(0.0, min(seg['length'], s))
        t_proj = seg['t_start'] + s / self.airspeed
        # Nearest point on segment
        cx = x0 + s * dx
        cy = y0
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        return t_proj, d2

    def _project_turn(self, seg: dict, x: float, y: float):
        cx_c = seg['cx']
        cy_c = seg['cy']
        R    = seg['R']
        sign = seg['sign']
        THETA_ENTRY = -math.pi / 2.0    # arc always enters due South of center

        # Angular position of (x, y) relative to arc center
        theta = math.atan2(y - cy_c, x - cx_c)

        # Arc parameter: how far along the arc (in radians) is theta?
        # Positive = in the direction of travel (sign).
        delta = (sign * (theta - THETA_ENTRY)) % (2 * math.pi)

        # The arc spans exactly pi radians; clamp delta to [0, pi].
        # For points past the exit (delta > pi), choose the nearer endpoint.
        if delta <= math.pi:
            delta_clamped = delta
        else:
            # past exit: compare distance-to-exit vs distance-to-entry
            dist_to_exit  = delta - math.pi
            dist_to_entry = 2 * math.pi - delta
            delta_clamped = math.pi if dist_to_exit <= dist_to_entry else 0.0

        t_proj = seg['t_start'] + delta_clamped / self._omega
        t_proj = max(seg['t_start'], min(seg['t_end'], t_proj))

        # Closest point on arc at clamped parameter
        theta_c = THETA_ENTRY + sign * self._omega * (t_proj - seg['t_start'])
        near_x  = cx_c + R * math.cos(theta_c)
        near_y  = cy_c + R * math.sin(theta_c)
        d2 = (x - near_x) ** 2 + (y - near_y) ** 2
        return t_proj, d2

    @property
    def total_time(self) -> float:
        return self._t_total

    @property
    def strip_width(self) -> float:
        return 2.0 * self.turn_radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Standalone plot / CLI
# ---------------------------------------------------------------------------

def _plot(altitude, airspeed, leg_length, turn_radius, num_legs, loop=False, t_end=None, runway_length=0.0):
    import matplotlib.pyplot as plt

    traj = LawnmowerTrajectory(altitude, airspeed, leg_length, turn_radius, num_legs,
                               loop=loop, runway_length=runway_length)
    dt   = 0.05
    stop = t_end if t_end is not None else traj.total_time
    ts   = np.arange(0, stop + dt, dt)

    xs, ys, psis, segs = [], [], [], []
    for t in ts:
        pt = traj.query(t)
        xs.append(pt.x_ref)
        ys.append(pt.y_ref)
        psis.append(math.degrees(pt.psi_ref))
        segs.append(pt.segment)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colours = {'straight': 'steelblue', 'turn': 'tomato', 'done': 'gray'}
    for i in range(len(ts) - 1):
        ax1.plot([ys[i], ys[i+1]], [xs[i], xs[i+1]],
                 color=colours.get(segs[i], 'black'), linewidth=2)

    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('Lawnmower Trajectory (NED, blue=straight red=turn)')
    ax1.set_aspect('equal')
    ax1.plot(ys[0], xs[0], 'go', markersize=8, label='start')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(ts, psis)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('psi_ref (deg)')
    ax2.set_title('Reference Heading vs Time')
    ax2.grid(True, linestyle='--', alpha=0.4)

    loop_str = f'  loop  T_end={stop:.0f} s' if loop else f'  T={traj.total_time:.0f} s'
    fig.suptitle(
        f'alt={altitude} m  V={airspeed} m/s  L={leg_length} m  '
        f'R={turn_radius} m  legs={num_legs}{loop_str}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lawnmower trajectory — plot mode')
    parser.add_argument('--alt',      type=float, default=50.0,  help='Altitude AGL (m)')
    parser.add_argument('--airspeed', type=float, default=17.0,  help='Airspeed (m/s)')
    parser.add_argument('--leg',      type=float, default=200.0, help='Straight leg length (m)')
    parser.add_argument('--radius',   type=float, default=40.0,  help='Turn radius (m)')
    parser.add_argument('--legs',     type=int,   default=4,     help='Number of legs')
    parser.add_argument('--loop',     action='store_true',       help='Repeat pattern indefinitely')
    parser.add_argument('--t_end',    type=float, default=None,  help='Plot duration (s); required with --loop')
    parser.add_argument('--runway',   type=float, default=0.0,   help='North lead-in runway before lawnmower (m)')
    args = parser.parse_args()

    if args.loop and args.t_end is None:
        parser.error('--t_end is required when --loop is set')

    _plot(args.alt, args.airspeed, args.leg, args.radius, args.legs,
          loop=args.loop, t_end=args.t_end, runway_length=args.runway)
