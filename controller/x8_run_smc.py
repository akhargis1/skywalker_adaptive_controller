"""
x8_run_smc.py — Path-following SMC runner for Skywalker X8 SITL.

Runs the three-surface sliding-mode controller (x8_path_smc.py) in GUIDED mode,
flying a lawnmower coverage trajectory defined by the same parameters as
generate_waypoints.py.

Usage
-----
    # Defaults: 4-leg, 200 m legs, 40 m turns, 100 m AGL, 17 m/s
    python3 x8_run_smc.py --alt 100 --leg 200 --radius 40 --legs 4

    # Custom connect / gains / log prefix:
    python3 x8_run_smc.py --alt 100 --leg 200 --radius 40 --legs 4 \\
        --connect tcp:127.0.0.1:5762 --log my_run

Workflow
--------
    Terminal 1:  sim_vehicle.py -v ArduPlane --model JSON ...  (SITL)
    Terminal 2:  gz sim -v4 -r skywalker_x8_runway.sdf         (Gazebo)
    Terminal 3:  (fly manually to cruise alt, then:)
                 python3 x8_run_smc.py --alt 100 --leg 200 --radius 40

The script switches to GUIDED mode, waits for LOCAL_POSITION_NED, then
starts the mission clock and runs the SMC loop at --hz until the trajectory
is complete, --max-time elapses, or an abort limit is hit.

Log format
----------
Saves <prefix>_<unix_timestamp>.npz with keys:
    'data'   — float64 array, shape (N, 29)
    'fields' — array of column-name strings (29 entries)

Post-flight quick-plot:
    import numpy as np, matplotlib.pyplot as plt
    d = np.load('sitl_smc_<ts>.npz')
    data = d['data']; cols = list(d['fields'])
    plt.plot(data[:, cols.index('t')], data[:, cols.index('e_n')])
    plt.xlabel('t (s)'); plt.ylabel('e_n (m)'); plt.show()
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from trajectory_generation import LawnmowerTrajectory
from x8_path_smc import PathSMC, PathSMCGains, SMCOutput
from x8_mavlink import (
    connect, wait_for_state, set_mode,
    StateBuffer, MAVReceiver,
    send_attitude_target,
    GUIDED_MODE, FBWA_MODE,
)


# ---------------------------------------------------------------------------
# Safety abort limits  (matching x8_run.py conventions)
# ---------------------------------------------------------------------------
ABORT_ROLL_DEG    = 55.0
ABORT_PITCH_DEG   = 35.0
ABORT_AIRSPEED_LO =  9.0
ABORT_AIRSPEED_HI = 28.0


# ---------------------------------------------------------------------------
# Logger
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


class SMCLogger:
    def __init__(self, prefix: str = 'sitl_smc'):
        self._rows   = []
        self._prefix = prefix
        self._t0     = time.monotonic()

    def record(self, state, out: SMCOutput, elapsed: float, x: float, y: float):
        self._rows.append([
            elapsed,
            x,        y,        state.z,
            state.vx, state.vy, state.vz,
            state.phi, state.theta, state.psi,
            state.p,   state.q,    state.r,
            state.airspeed,
            out.e_n, out.e_t, out.e_z,
            out.chi_err,
            out.kappa, out.psi_r,
            out.s1, out.s2, out.s3,
            out.phi_cmd, out.theta_cmd, out.T_cmd,
            out.x_r, out.y_r,
            out.gamma,
        ])

    def save(self):
        if not self._rows:
            print("[LOG] No data to save.")
            return
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='X8 path-following SMC runner (GUIDED mode)')
    p.add_argument('--connect',   default='tcp:127.0.0.1:5762')
    p.add_argument('--hz',        type=float, default=50.0)
    # Trajectory
    p.add_argument('--alt',       type=float, default=100.0,
                   help='Cruise altitude AGL (m)')
    p.add_argument('--leg',       type=float, default=200.0,
                   help='Straight leg length (m)')
    p.add_argument('--radius',    type=float, default=40.0,
                   help='Turn radius (m)')
    p.add_argument('--legs',      type=int,   default=4,
                   help='Number of legs')
    p.add_argument('--airspeed',  type=float, default=17.0,
                   help='Desired cruise airspeed (m/s)')
    p.add_argument('--runway',    type=float, default=200.0,
                   help='North lead-in runway before lawnmower grid (m); 0 to disable')
    # Runner
    p.add_argument('--max-time',  type=float, default=600.0,
                   help='Mission time limit (s); 0 = no limit')
    p.add_argument('--log',       default='sitl_smc')
    p.add_argument('--verbose',   action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dt   = 1.0 / args.hz

    # ------------------------------------------------------------------
    # Connect + start receiver
    # ------------------------------------------------------------------
    conn     = connect(args.connect, stream_hz=int(args.hz))
    buf      = StateBuffer()
    receiver = MAVReceiver(conn, buf, verbose=args.verbose)
    receiver.start()

    print("[WAIT] Waiting for attitude telemetry ...")
    if not wait_for_state(buf, timeout=10.0):
        sys.exit("[ERROR] No attitude data after 10 s.")

    # ------------------------------------------------------------------
    # Switch to GUIDED
    # ------------------------------------------------------------------
    print("[MODE] Requesting GUIDED ...")
    set_mode(conn, GUIDED_MODE)
    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Wait for position telemetry
    # ------------------------------------------------------------------
    print("[WAIT] Waiting for LOCAL_POSITION_NED (pos_valid) ...")
    t_wait = time.monotonic()
    while True:
        if buf.read().pos_valid:
            break
        if time.monotonic() - t_wait > 15.0:
            sys.exit("[ERROR] No LOCAL_POSITION_NED after 15 s. "
                     "Check MAV_DATA_STREAM or EKF status.")
        time.sleep(0.1)
    print("[WAIT] Position valid.\n")

    # ------------------------------------------------------------------
    # Initialise trajectory + SMC
    # ------------------------------------------------------------------
    traj = LawnmowerTrajectory(
        altitude      = args.alt,
        airspeed      = args.airspeed,
        leg_length    = args.leg,
        turn_radius   = args.radius,
        num_legs      = args.legs,
        runway_length = args.runway,
    )
    gains = PathSMCGains()
    smc   = PathSMC(traj, gains)

    print(f"[TRAJ] Total trajectory time: {traj.total_time:.1f} s  "
          f"({args.legs} legs × {args.leg} m, R={args.radius} m, "
          f"alt={args.alt} m AGL)")
    print(f"[SMC]  λ=({gains.lambda_n:.2f}, {gains.lambda_t:.2f}, {gains.lambda_z:.2f})  "
          f"η=({gains.eta_n:.2f}, {gains.eta_t:.2f}, {gains.eta_z:.2f})  "
          f"Φ=({gains.phi_n:.2f}, {gains.phi_t:.2f}, {gains.phi_z:.2f})")
    print(f"[RUN]  {args.hz:.0f} Hz loop.  Ctrl-C to stop.\n")

    logger  = SMCLogger(prefix=args.log)
    t_start = None
    x0 = y0 = 0.0
    tick    = 0
    t_next  = time.monotonic()

    try:
        while True:
            # --- Pace ---
            t_next += dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

            state = buf.read()

            # Skip until position is available
            if not state.pos_valid:
                continue

            # Start mission clock on first valid position tick
            if t_start is None:
                t_start = time.monotonic()
                x0, y0 = state.x, state.y
                print(f"[GO]  Mission clock started. Origin: ({x0:.1f}, {y0:.1f}) NED")

            elapsed = time.monotonic() - t_start

            # --- Max-time guard ---
            if args.max_time > 0 and elapsed >= args.max_time:
                print(f"\n[STOP] Max time {args.max_time:.0f} s reached.")
                break

            # --- Trajectory complete? ---
            ref = traj.query(elapsed)
            if ref.segment == 'done':
                print("\n[DONE] Trajectory complete.")
                break

            # --- SMC update ---
            x_rel = state.x - x0
            y_rel = state.y - y0
            out = smc.update(
                x=x_rel, y=y_rel, z=state.z,
                vx=state.vx, vy=state.vy, vz=state.vz,
                phi=state.phi,
                t=elapsed,
            )

            # --- Abort check ---
            reason = _check_abort(state)
            if reason:
                print(f"\n[ABORT] {reason}")
                set_mode(conn, FBWA_MODE)
                break

            # --- Send commands ---
            send_attitude_target(conn,
                                 roll_d=out.phi_cmd,
                                 pitch_d=out.theta_cmd,
                                 yaw_d=state.psi,
                                 thrust=out.T_cmd,
                                 type_mask=0b00000111)

            # --- Log ---
            logger.record(state, out, elapsed, x=x_rel, y=y_rel)

            # --- Console (1 Hz) ---
            if tick % int(args.hz) == 0:
                print(
                    f"t={elapsed:6.1f}s  "
                    f"e_n={out.e_n:+6.2f}m  "
                    f"e_t={out.e_t:+6.2f}m  "
                    f"e_z={out.e_z:+6.2f}m  "
                    f"s1={out.s1:+5.2f}  "
                    f"φ={math.degrees(out.phi_cmd):+6.1f}°  "
                    f"θ={math.degrees(out.theta_cmd):+5.1f}°  "
                    f"T={out.T_cmd:.2f}  [{ref.segment}]"
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
