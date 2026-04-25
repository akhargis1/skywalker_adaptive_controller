"""
x8_run.py — SITL runner.  Entry point.

Wires together all other modules and runs the 50 Hz control loop.

Usage:
    python3 x8_run.py
    python3 x8_run.py --connect tcp:127.0.0.1:5762 --test chirp --hz 50
    python3 x8_run.py --rc       # use RC override instead of SET_ATTITUDE_TARGET

Prerequisites:
    Terminal 1:  sim_vehicle.py -v ArduPlane -f gazebo-zephyr --map --console
    Terminal 2:  gz sim -r skywalker_x8.sdf          (if not launched by sim_vehicle)
    Terminal 3:  python3 x8_run.py

All five source files must be in the same directory:
    x8_params.py      airframe parameters + gains
    x8_controller.py  control law (regressor, adaptive law, mixer)
    x8_mavlink.py     MAVLink connection + rx thread + senders
    x8_sequencer.py   test sequences + abort checker
    x8_logger.py      flight data logger
    x8_run.py         ← this file
    x8_plot.py        (run separately after a flight to view logs)
"""

import argparse
import sys
import time

import numpy as np

from x8_params     import X8Params, CASMCGains
from x8_controller import X8Controller
from x8_mavlink    import (connect, wait_for_state, set_mode,
                           StateBuffer, MAVReceiver,
                           send_attitude_target, send_elevon_direct, 
                           toggle_research_mode, 
                           GUIDED_MODE, FBWA_MODE)
from x8_sequencer  import TestSequencer, AbortChecker
from x8_logger     import FlightLogger


def parse_args():
    p = argparse.ArgumentParser(description="X8 CASMC SITL runner")
    p.add_argument('--connect', default='tcp:127.0.0.1:5762')
    p.add_argument('--hz',     type=float, default=50.0)
    p.add_argument('--test',   default='doublet',
                   choices=TestSequencer.SEQUENCES)
    p.add_argument('--log',    default='sitl_run')
    p.add_argument('--rc',     action='store_true',
                   help='Send RC override instead of SET_ATTITUDE_TARGET')
    p.add_argument('--arm',    action='store_true',
                   help='Auto-arm and switch to GUIDED')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

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

    print("[WAIT] Waiting for valid state ...")
    if not wait_for_state(buf, timeout=10.0):
        sys.exit("[ERROR] No attitude data after 10 s.")
    print("[WAIT] State valid.\n")

    # ------------------------------------------------------------------
    # Optional arm + GUIDED
    # ------------------------------------------------------------------
    if args.arm:
        print("[ARM] Arming ...")
        conn.arducopter_arm()
        time.sleep(2)
        set_mode(conn, GUIDED_MODE)
        time.sleep(1)
        print("[MODE] GUIDED set.")

    # ------------------------------------------------------------------
    # Initialise modules
    # ------------------------------------------------------------------
    params   = X8Params()
    gains    = CASMCGains()
    ctrl     = X8Controller(params, gains)
    ctrl.dt  = dt

    state0   = buf.read()
    ctrl.reset(np.array([state0.phi, state0.theta, state0.psi]))

    seq      = TestSequencer(kind=args.test)
    seq.start()
    abort    = AbortChecker()
    logger   = FlightLogger(prefix=args.log)

    if args.rc:   
        print("[MODE] Switching to MANUAL ...")
        set_mode(conn, 0)   # 0 = MANUAL on ArduPlane
        toggle_research_mode(conn, active=True)
        time.sleep(1)

    print(f"[RUN] {args.test} sequence at {args.hz:.0f} Hz.  Ctrl-C to stop.\n")

    tick   = 0
    t_next = time.monotonic()

    try:
        while True:
            # --- Pace the loop ---
            t_next += dt
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

            # --- Read state ---
            state = buf.read()
            if not state.valid:
                continue

            q_att = np.array([state.phi, state.theta, state.psi])
            dq    = np.array([state.p,   state.q,     state.r])
            q_cmd = seq.get_command(state)

            # --- Control ---
            out = ctrl.update(
                q_att    = q_att,
                dq       = dq,
                q_cmd    = q_cmd,
                alpha    = state.alpha,
                beta     = state.beta,
                airspeed = state.airspeed,
            )

            # --- Abort check ---
            reason = abort.check(state, out['s'],
                                 out['theta_hat'], params.theta_nominal)
            if reason:
                print(f"\n[ABORT] {reason}")
                set_mode(conn, FBWA_MODE)
                toggle_research_mode(conn, active=False)
                break

            # --- Send ---
            if args.rc:
                send_elevon_direct(conn, out['delta_L'], out['delta_R'],
                                 elevon_limit_deg=params.elevon_limit_deg)
            else:
                q_d  = out['q_d']
                dq_d = out['dq_d']
                send_attitude_target(conn,
                                     roll_d=q_d[0],  pitch_d=q_d[1],  yaw_d=q_d[2],
                                     roll_rate_d=dq_d[0], pitch_rate_d=dq_d[1],
                                     yaw_rate_d=dq_d[2])

            # --- Log ---
            logger.record(state, out)

            # --- Console (1 Hz) ---
            if tick % int(args.hz) == 0:
                s_n = float(np.linalg.norm(np.degrees(out['s'])))
                print(
                    f"t={tick*dt:6.1f}s  "
                    f"φ={float(np.degrees(state.phi)):+6.1f}°  "
                    f"θ={float(np.degrees(state.theta)):+6.1f}°  "
                    f"|s|={s_n:5.1f}°/s  "
                    f"δL={float(np.degrees(out['delta_L'])):+5.1f}°  "
                    f"δR={float(np.degrees(out['delta_R'])):+5.1f}°  "
                    f"V={out['V']:.4f}"
                )

            tick += 1

    except KeyboardInterrupt:
        print("\n[STOP] User interrupt.")

    finally:
        receiver.stop()
        logger.save()
        print("[DONE]")


if __name__ == "__main__":
    main()
