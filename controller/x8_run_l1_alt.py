"""
x8_run_l1.py — ArduPlane AUTO-mode L1 baseline runner for Skywalker X8 SITL.

Uploads the same lawnmower coverage trajectory as x8_run_smc.py as a MAVLink
mission and flies it using ArduPlane's built-in L1 lateral guidance + TECS
longitudinal control.  Logs the same 29-column NPZ as x8_run_smc.py so that
x8_plot_smc.py can overlay both runs for direct comparison.

Usage
-----
    python3 x8_run_l1.py --alt 100 --leg 200 --radius 40 --legs 4

    python3 x8_run_l1.py --alt 100 --leg 200 --radius 40 --legs 4 \\
        --connect tcp:127.0.0.1:5762 --log l1_run

Workflow
--------
    Terminal 1:  sim_vehicle.py -v ArduPlane --model JSON ...  (SITL)
    Terminal 2:  gz sim -v4 -r skywalker_x8_runway.sdf         (Gazebo)
    Terminal 3:  (arm + takeoff to cruise alt, then:)
                 python3 x8_run_l1.py --alt 100 --leg 200 --radius 40

The script requests HOME_POSITION to anchor the local NED frame, converts
trajectory waypoints to global lat/lon, uploads the mission, switches to AUTO,
then monitors and logs until the trajectory is complete or an abort limit fires.

Compared to x8_run_smc.py fields:
  s1/s2/s3    — always 0.0  (no SMC surfaces)
  phi_cmd     — always 0.0  (ArduPlane commands internally)
  theta_cmd   — always 0.0
  T_cmd       — VFR_HUD throttle / 100 (0–1)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

from trajectory_generation import LawnmowerTrajectory
from x8_mavlink import (
    connect, wait_for_state, set_mode,
    StateBuffer, MAVReceiver,
    AUTO_MODE, FBWA_MODE,
)


# ---------------------------------------------------------------------------
# Safety abort limits  (same as x8_run_smc.py)
# ---------------------------------------------------------------------------
ABORT_ROLL_DEG    = 75.0
ABORT_PITCH_DEG   = 35.0
ABORT_AIRSPEED_LO =  9.0
ABORT_AIRSPEED_HI = 28.0

R_EARTH = 6_371_000.0   # m


# ---------------------------------------------------------------------------
# Logger  (29 columns — identical schema to x8_run_smc.py)
# ---------------------------------------------------------------------------

_FIELDS = [
    't',
    'x', 'y', 'z',
    'vx', 'vy', 'vz',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'airspeed',
    'e_n', 'e_t', 'e_z',
    'chi_err',
    'kappa', 'psi_r',
    's1', 's2', 's3',
    'phi_cmd', 'theta_cmd', 'T_cmd',
    'x_r', 'y_r',
    'gamma',
]


class L1Logger:
    def __init__(self, prefix: str = 'sitl_l1'):
        self._rows   = []
        self._prefix = prefix

    def record(self, state, elapsed, x_rel, y_rel,
               e_n, e_t, e_z, chi_err, ref, t_nearest):
        gamma = math.atan2(-state.vz, math.hypot(state.vx, state.vy))
        self._rows.append([
            elapsed,
            x_rel,      y_rel,      state.z,
            state.vx,   state.vy,   state.vz,
            state.phi,  state.theta, state.psi,
            state.p,    state.q,    state.r,
            state.airspeed,
            e_n, e_t, e_z,
            chi_err,
            ref.psi_dot_ref, ref.psi_ref,
            0.0, 0.0, 0.0,           # s1, s2, s3
            0.0, 0.0,                # phi_cmd, theta_cmd
            state.throttle / 100.0,  # T_cmd  (0–1)
            ref.x_ref, ref.y_ref,
            gamma,
        ])

    def save(self):
        if not self._rows:
            print("[LOG] No data to save.")
            return None
        arr  = np.array(self._rows, dtype=np.float64)
        path = Path(f"{self._prefix}_{int(time.time())}.npz")
        np.savez_compressed(path, data=arr, fields=np.array(_FIELDS))
        print(f"[LOG] {len(self._rows)} rows → {path}")
        return path


# ---------------------------------------------------------------------------
# Abort checker
# ---------------------------------------------------------------------------

def _check_abort(state) -> str:
    if abs(math.degrees(state.phi)) > ABORT_ROLL_DEG:
        return f"roll {math.degrees(state.phi):+.1f}° > ±{ABORT_ROLL_DEG}°"
    if abs(math.degrees(state.theta)) > ABORT_PITCH_DEG:
        return f"pitch {math.degrees(state.theta):+.1f}° > ±{ABORT_PITCH_DEG}°"
    if not (ABORT_AIRSPEED_LO <= state.airspeed <= ABORT_AIRSPEED_HI):
        return f"airspeed {state.airspeed:.1f} m/s out of [{ABORT_AIRSPEED_LO}, {ABORT_AIRSPEED_HI}]"
    return ""


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ned_to_global(x_ned: float, y_ned: float, home_lat: float, home_lon: float):
    """Convert local NED offset (m) to global lat/lon (degrees)."""
    lat = home_lat + math.degrees(x_ned / R_EARTH)
    lon = home_lon + math.degrees(y_ned / (R_EARTH * math.cos(math.radians(home_lat))))
    return lat, lon


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Anchor position
# ---------------------------------------------------------------------------

def get_anchor_position(conn, timeout: float = 15.0):
    """
    Read current aircraft global position from GLOBAL_POSITION_INT.
    Returns (lat_deg, lon_deg).

    Must be called BEFORE MAVReceiver is started — the receiver thread would
    otherwise consume this message first.  GLOBAL_POSITION_INT is broadcast
    continuously so no explicit request is needed.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg and msg.lat != 0:
            return msg.lat / 1e7, msg.lon / 1e7
    sys.exit("[ERROR] GLOBAL_POSITION_INT not received — "
             "is the vehicle flying with a valid GPS fix?")


# ---------------------------------------------------------------------------
# Mission upload
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mission helpers — straight legs with loiter-turn transitions
# ---------------------------------------------------------------------------

def _build_mission_items(traj: LawnmowerTrajectory):
    """
    Return a list of mission item descriptors for a clean lawnmower pattern:
      - Each straight leg ends with a NAV_WAYPOINT at the leg exit point.
      - Each turn arc becomes a NAV_LOITER_TURNS centered at the arc center,
        with the correct radius and direction (+1 CW / -1 CCW).

    Each item is a dict:
        {'type': 'waypoint', 'x': float, 'y': float}
        {'type': 'loiter',   'cx': float, 'cy': float,
                             'radius': float, 'turns': float, 'dir': int}
    """
    items = []
    for seg in traj._segments:
        if seg['type'] == 'straight':
            # Endpoint of the straight leg
            pt = traj.query(seg['t_end'])
            items.append({'type': 'waypoint', 'x': pt.x_ref, 'y': pt.y_ref})

        elif seg['type'] == 'turn':
            # Arc center, radius, and direction are stored on the segment.
            # Fallback: derive center from geometry if not present.
            cx = seg.get('cx')
            cy = seg.get('cy')
            radius = seg.get('radius', traj.turn_radius)
            direction = seg.get('direction', 1)   # +1 CW, -1 CCW

            if cx is None or cy is None:
                # Reconstruct center from start/end points and known radius
                pt_s = traj.query(seg['t_start'])
                pt_e = traj.query(seg['t_end'])
                # Midpoint perpendicular — simple approximation for 180° turns
                cx = (pt_s.x_ref + pt_e.x_ref) / 2.0
                cy = (pt_s.y_ref + pt_e.y_ref) / 2.0

            # Arc angle → fractional turns (lawnmower turns are nominally π rad)
            dt   = seg['t_end'] - seg['t_start']
            arc  = traj.airspeed * dt          # arc length (m)
            turns = arc / (2 * math.pi * radius)  # fractional turns

            items.append({
                'type':   'loiter',
                'cx':     cx,
                'cy':     cy,
                'radius': radius,
                'turns':  max(turns, 0.5),     # ArduPlane needs ≥ 0.5 turns to register
                'dir':    direction,
            })

    return items


def set_loiter_radius(conn, radius_m: float, verbose: bool = False):
    """Push WP_LOITER_RAD to ArduPlane so NAV_LOITER_TURNS uses the right radius."""
    conn.mav.param_set_send(
        conn.target_system,
        conn.target_component,
        b'WP_LOITER_RAD',
        radius_m,           # positive = CW, negative = CCW
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    ack = conn.recv_match(type='PARAM_VALUE', blocking=True, timeout=5.0)
    if verbose or ack is None:
        print(f"[PARAM] WP_LOITER_RAD set to {radius_m} m  (ack={ack})")

def upload_mission(conn, traj: LawnmowerTrajectory, home_lat: float, home_lon: float,
                   alt_agl: float, airspeed: float,
                   wp_divs: int = 1, verbose: bool = False):
    """
    Upload lawnmower mission: DO_CHANGE_SPEED → alternating NAV_WAYPOINT (leg
    exits) and NAV_LOITER_TURNS (turn centers).

    wp_divs is accepted for API compatibility but ignored — loiter turns
    replace the old arc-discretisation approach.
    """
    mission_items = _build_mission_items(traj)
    total_items   = 1 + len(mission_items)   # +1 for speed command at seq 0

    print(f"[MISS] Uploading {total_items} mission items ...")
    if verbose:
        for i, it in enumerate(mission_items):
            print(f"  [{i+1}] {it}")

    # 1. Clear existing mission
    conn.mav.mission_clear_all_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
    ack = conn.recv_match(type='MISSION_ACK', blocking=True, timeout=5.0)
    if verbose:
        print(f"[MISS] CLEAR_ALL ack: {ack}")

    # 2. Announce count
    conn.mav.mission_count_send(
        conn.target_system, conn.target_component,
        total_items, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

    # 3. Service MISSION_REQUEST_INT messages
    sent = 0
    t0   = time.monotonic()
    while sent < total_items:
        if time.monotonic() - t0 > 30.0:
            sys.exit("[ERROR] Mission upload timed out.")

        msg = conn.recv_match(
            type=['MISSION_REQUEST_INT', 'MISSION_REQUEST', 'MISSION_ACK'],
            blocking=True, timeout=5.0)
        if msg is None:
            continue

        mt = msg.get_type()
        if mt == 'MISSION_ACK':
            if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("[MISS] Mission accepted.")
            else:
                print(f"[MISS] WARNING: Mission ACK type {msg.type}")
            break

        seq = msg.seq
        if verbose:
            print(f"[MISS] Request seq={seq}")

        if seq == 0:
            # ---- Item 0: set cruise airspeed --------------------------------
            conn.mav.mission_item_int_send(
                conn.target_system, conn.target_component,
                0,
                mavutil.mavlink.MAV_FRAME_MISSION,
                mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                0, 1,
                0, airspeed, -1, 0,
                0, 0, 0,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        else:
            # ---- Items 1…N: waypoints and loiter turns ----------------------
            it = mission_items[seq - 1]

            if it['type'] == 'waypoint':
                lat, lon = ned_to_global(it['x'], it['y'], home_lat, home_lon)
                if verbose:
                    print(f"[MISS]   WP  seq={seq}: N={it['x']:.1f} E={it['y']:.1f}")
                conn.mav.mission_item_int_send(
                    conn.target_system, conn.target_component,
                    seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 1,
                    0,           # hold time
                    5.0,         # acceptance radius (m)
                    0, 0,        # pass-through, yaw
                    int(lat * 1e7),
                    int(lon * 1e7),
                    alt_agl,
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )

            else:  # loiter
                lat, lon = ned_to_global(it['cx'], it['cy'], home_lat, home_lon)
                # ArduPlane param3 for LOITER_TURNS: positive = CW, negative = CCW
                radius_signed = it['radius'] * it['dir']
                if verbose:
                    print(f"[MISS]   LTR seq={seq}: "
                          f"ctr=({it['cx']:.1f},{it['cy']:.1f}) "
                          f"R={radius_signed:.1f} m  "
                          f"turns={it['turns']:.2f}")
                conn.mav.mission_item_int_send(
                    conn.target_system, conn.target_component,
                    seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
                    0, 1,
                    it['turns'],     # param1: number of turns
                    0,               # param2: heading required (0 = no)
                    radius_signed,   # param3: radius (+CW / -CCW)
                    0,               # param4: xtrack exit (0 = use WP_LOITER_RAD)
                    int(lat * 1e7),
                    int(lon * 1e7),
                    alt_agl,
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )

        sent += 1

    conn.mav.mission_set_current_send(
        conn.target_system, conn.target_component, 0)
    print("[MISS] Mission set current → item 0")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='X8 ArduPlane L1 baseline runner (AUTO mode)')
    p.add_argument('--connect',   default='tcp:127.0.0.1:5762')
    p.add_argument('--hz',        type=float, default=50.0)
    # Trajectory (same as x8_run_smc.py)
    p.add_argument('--alt',       type=float, default=100.0,
                   help='Cruise altitude AGL (m)')
    p.add_argument('--leg',       type=float, default=200.0,
                   help='Straight leg length (m)')
    p.add_argument('--radius',    type=float, default=40.0,
                   help='Turn radius (m) — used for strip spacing only; '
                        'L1 turn tightness set by WP_LOITER_RAD param in ArduPilot')
    p.add_argument('--legs',      type=int,   default=4,
                   help='Number of legs')
    p.add_argument('--airspeed',  type=float, default=17.0,
                   help='Desired cruise airspeed (m/s) — sent via DO_CHANGE_SPEED')
    p.add_argument('--runway',    type=float, default=200.0,
                   help='North lead-in runway before lawnmower grid (m); 0 to disable')
    # Runner
    p.add_argument('--max-time',  type=float, default=600.0,
                   help='Observation time limit (s); 0 = no limit')
    p.add_argument('--wp-divs',   type=int,   default=1,
                   help='Waypoints per trajectory segment (1=leg endpoints only; '
                        '>1 adds evenly-spaced WPs through turns for finer L1 fidelity)')
    p.add_argument('--log',       default='sitl_l1')
    p.add_argument('--verbose',   action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dt   = 1.0 / args.hz

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------
    conn = connect(args.connect, stream_hz=int(args.hz))

    # Capture anchor position BEFORE starting receiver thread.
    # The receiver would otherwise consume GLOBAL_POSITION_INT first.
    # Anchoring to current position mirrors SMC's x0/y0 capture —
    # the trajectory starts from wherever the aircraft is right now.
    print("[ANCHOR] Reading current aircraft position ...")
    anchor_lat, anchor_lon = get_anchor_position(conn)
    print(f"[ANCHOR] lat={anchor_lat:.6f}  lon={anchor_lon:.6f}")

    # ------------------------------------------------------------------
    # Start receiver
    # ------------------------------------------------------------------
    buf      = StateBuffer()
    receiver = MAVReceiver(conn, buf, verbose=args.verbose)
    receiver.start()

    print("[WAIT] Waiting for attitude telemetry ...")
    if not wait_for_state(buf, timeout=10.0):
        sys.exit("[ERROR] No attitude data after 10 s.")

    # ------------------------------------------------------------------
    # Build trajectory + upload mission
    # ------------------------------------------------------------------
    traj = LawnmowerTrajectory(
        altitude      = args.alt,
        airspeed      = args.airspeed,
        leg_length    = args.leg,
        turn_radius   = args.radius,
        num_legs      = args.legs,
        runway_length = args.runway,
    )

    print(f"[TRAJ] Total time: {traj.total_time:.1f} s  "
          f"({args.legs} legs × {args.leg} m, R={args.radius} m, alt={args.alt} m AGL)")

    set_loiter_radius(conn, traj.turn_radius, verbose=args.verbose)
    upload_mission(conn, traj, anchor_lat, anchor_lon,
                   alt_agl=args.alt, airspeed=args.airspeed,
                   wp_divs=args.wp_divs, verbose=args.verbose)

    # ------------------------------------------------------------------
    # Wait for position telemetry, then switch to AUTO
    # ------------------------------------------------------------------
    print("[WAIT] Waiting for LOCAL_POSITION_NED ...")
    t_wait = time.monotonic()
    while True:
        if buf.read().pos_valid:
            break
        if time.monotonic() - t_wait > 15.0:
            sys.exit("[ERROR] No LOCAL_POSITION_NED after 15 s.")
        time.sleep(0.1)

    print("[MODE] Switching to AUTO ...")
    set_mode(conn, AUTO_MODE)
    time.sleep(1.0)
    print("[RUN]  Monitoring L1 flight.  Ctrl-C to stop.\n")

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------
    logger  = L1Logger(prefix=args.log)
    t_start = None
    x0 = y0 = 0.0
    tick    = 0
    t_next  = time.monotonic()

    try:
        while True:
            t_next += dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

            state = buf.read()
            if not state.pos_valid:
                continue

            if t_start is None:
                t_start = time.monotonic()
                x0, y0  = state.x, state.y
                print(f"[GO]  Clock started.  Origin: ({x0:.1f}, {y0:.1f}) NED")

            elapsed = time.monotonic() - t_start

            if args.max_time > 0 and elapsed >= args.max_time:
                print(f"\n[STOP] Max time {args.max_time:.0f} s reached.")
                break

            x_rel = state.x - x0
            y_rel = state.y - y0

            # Nearest point on trajectory
            t_nearest = traj.nearest_t(x_rel, y_rel)
            ref = traj.query(t_nearest)
            if ref.segment == 'done':
                print("\n[DONE] Trajectory complete.")
                break

            # Tracking errors
            e_n, chi_err = traj.cross_track_error(x_rel, y_rel, state.psi, t_nearest)
            e_t = (t_nearest - elapsed) * traj.airspeed
            e_z = state.z - ref.z_ref

            # Abort check
            reason = _check_abort(state)
            if reason:
                print(f"\n[ABORT] {reason}")
                set_mode(conn, FBWA_MODE)
                break

            logger.record(state, elapsed, x_rel, y_rel,
                          e_n, e_t, e_z, chi_err, ref, t_nearest)

            if tick % int(args.hz) == 0:
                print(
                    f"t={elapsed:6.1f}s  "
                    f"e_n={e_n:+6.2f}m  "
                    f"e_z={e_z:+6.2f}m  "
                    f"airspeed={state.airspeed:.1f}m/s  "
                    f"T={state.throttle/100:.2f}  [{ref.segment}]"
                )

            tick += 1

    except KeyboardInterrupt:
        print("\n[STOP] User interrupt.")

    finally:
        receiver.stop()
        logger.save()
        print("[DONE]")


if __name__ == '__main__':
    main()
