"""
x8_startup.py — Arm, take off, and fly north to cruise altitude/airspeed.

Used by run_sitl.sh to bring the aircraft to cruise conditions before handing
off to x8_run_smc.py.

Run via run_sitl.sh (which sets PYTHONPATH=.../controller); can also be run
directly if controller/ is on PYTHONPATH:

    PYTHONPATH=../controller python3 x8_startup.py --alt 100 --airspeed 17

Exit codes
    0  cruise conditions reached, safe to start mission
    1  timeout or fatal error
"""

import argparse
import math
import sys
import time

from pymavlink import mavutil

from x8_mavlink import (
    connect, StateBuffer, MAVReceiver,
    set_mode, send_airspeed_command,
    GUIDED_MODE,
)

TAKEOFF_MODE = 13   # PLANE_MODE_TAKEOFF

TAKEOFF_TIMEOUT  = 120.0   # s  — abort if not at altitude by this time
AIRSPEED_TIMEOUT =  60.0   # s  — warn and proceed if not at airspeed by this time
ALT_THRESHOLD    =  10.0   # m  — within this of target to declare altitude reached
AIRSPEED_TOL     =   2.0   # m/s — within this of target to declare cruise reached
NORTH_VX_SCALE   =   1.0   # fraction of airspeed to command as northward velocity


def _wait_port(host: str = '127.0.0.1', port: int = 5762, timeout: float = 60.0):
    """Block until TCP port is accepting connections (already done by bash; kept as safety)."""
    import socket
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    sys.exit(f'[ERROR] Port {port} not reachable after {timeout:.0f} s')


def _cmd_fly_north(conn, airspeed: float):
    """
    Command ArduPlane GUIDED to fly north at cruise airspeed.
    Uses SET_POSITION_TARGET_LOCAL_NED with velocity-only type_mask
    (ignore position, acceleration, yaw — only vx/vy/vz).
    """
    conn.mav.set_position_target_local_ned_send(
        0,                                        # time_boot_ms (ignored)
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b110111000111,                           # velocity only (vx, vy, vz)
        0, 0, 0,                                  # position (ignored)
        float(airspeed), 0.0, 0.0,               # vx=north, vy=0, vz=0
        0, 0, 0,                                  # acceleration (ignored)
        0, 0,                                     # yaw, yaw_rate (ignored)
    )


def main():
    ap = argparse.ArgumentParser(
        description='Arm, take off, and cruise to target altitude heading north')
    ap.add_argument('--alt',      type=float, default=100.0,
                    help='Target cruise altitude AGL (m)')
    ap.add_argument('--airspeed', type=float, default=17.0,
                    help='Target cruise airspeed (m/s)')
    ap.add_argument('--connect',  default='tcp:127.0.0.1:5762',
                    help='MAVLink address')
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------
    print(f'[STARTUP] Connecting to {args.connect} ...')
    conn = connect(args.connect, stream_hz=10)
    buf  = StateBuffer()
    rx   = MAVReceiver(conn, buf, verbose=True)
    rx.start()

    # ------------------------------------------------------------------
    # Wait for valid telemetry
    # ------------------------------------------------------------------
    print('[STARTUP] Waiting for telemetry ...')
    t0 = time.monotonic()
    while not buf.read().valid:
        if time.monotonic() - t0 > 30.0:
            rx.stop()
            sys.exit('[ERROR] No valid telemetry after 30 s')
        time.sleep(0.5)
    print('[STARTUP] Telemetry OK.')

    # ------------------------------------------------------------------
    # Wait for GPS lock (LOCAL_POSITION_NED) — confirms JSON bridge live
    # ------------------------------------------------------------------
    print('[STARTUP] Waiting for GPS lock (LOCAL_POSITION_NED) ...')
    t0 = time.monotonic()
    while not buf.read().pos_valid:
        if time.monotonic() - t0 > 30.0:
            rx.stop()
            sys.exit('[ERROR] No GPS lock after 30 s — is Gazebo running and connected?')
        time.sleep(0.5)
    print('[STARTUP] GPS lock OK.')

    # ------------------------------------------------------------------
    # Arm — retry with normal arm until pre-arm checks clear
    # (STATUSTEXT with verbose=True shows any failure reason)
    # ------------------------------------------------------------------
    print('[STARTUP] Waiting for pre-arm checks to clear ...')
    t0 = time.monotonic()
    while not buf.read().armed:
        if time.monotonic() - t0 > 60.0:
            rx.stop()
            sys.exit('[ERROR] Could not arm after 60 s — '
                     'check STATUSTEXT above for pre-arm failure reason')
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,   # param2=0: normal arm, no force
        )
        time.sleep(2.0)
    print('[STARTUP] Armed.')

    # ------------------------------------------------------------------
    # Set TKOFF_ALT to our target altitude so TAKEOFF mode climbs to it
    # ------------------------------------------------------------------
    conn.mav.param_set_send(
        conn.target_system, conn.target_component,
        b'TKOFF_ALT', float(args.alt),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    time.sleep(0.3)

    # ------------------------------------------------------------------
    # Switch to TAKEOFF mode — poll until confirmed
    # ------------------------------------------------------------------
    print('[STARTUP] Switching to TAKEOFF mode ...')
    set_mode(conn, TAKEOFF_MODE)
    t0 = time.monotonic()
    while buf.read().mode != TAKEOFF_MODE:
        if time.monotonic() - t0 > 10.0:
            rx.stop()
            sys.exit('[ERROR] TAKEOFF mode not confirmed after 10 s')
        set_mode(conn, TAKEOFF_MODE)
        time.sleep(0.5)
    print('[STARTUP] TAKEOFF mode — climbing ...')

    # ------------------------------------------------------------------
    # Wait for altitude
    # ------------------------------------------------------------------
    print(f'[STARTUP] Climbing ... target {args.alt:.0f} m AGL')
    t0 = time.monotonic()
    while True:
        state = buf.read()
        h = -state.z   # NED: z-down → h = altitude AGL
        elapsed = time.monotonic() - t0
        print(f'\r  h={h:6.1f}/{args.alt:.0f} m   V={state.airspeed:5.1f} m/s   '
              f'ψ={math.degrees(state.psi):+6.1f}°   t={elapsed:.0f} s   ',
              end='', flush=True)
        if h >= args.alt - ALT_THRESHOLD:
            print()
            break
        if elapsed > TAKEOFF_TIMEOUT:
            print()
            rx.stop()
            sys.exit(f'[ERROR] Takeoff timeout after {TAKEOFF_TIMEOUT:.0f} s '
                     f'(reached {h:.1f} m of {args.alt:.0f} m)')
        time.sleep(0.5)
    print(f'[STARTUP] Altitude reached ({-buf.read().z:.1f} m AGL).')

    # ------------------------------------------------------------------
    # Switch to GUIDED for airspeed / velocity commands
    # ------------------------------------------------------------------
    print('[STARTUP] Switching to GUIDED mode ...')
    set_mode(conn, GUIDED_MODE)
    t0 = time.monotonic()
    while buf.read().mode != GUIDED_MODE:
        if time.monotonic() - t0 > 10.0:
            rx.stop()
            sys.exit('[ERROR] GUIDED mode not confirmed after altitude reached')
        set_mode(conn, GUIDED_MODE)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Command cruise airspeed and fly north
    # ------------------------------------------------------------------
    print(f'[STARTUP] Commanding airspeed {args.airspeed:.1f} m/s and flying north ...')
    send_airspeed_command(conn, args.airspeed)
    _cmd_fly_north(conn, args.airspeed)

    # ------------------------------------------------------------------
    # Wait for airspeed to stabilise
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    while True:
        state = buf.read()
        V       = state.airspeed
        elapsed = time.monotonic() - t0
        print(f'\r  V={V:5.1f}/{args.airspeed:.1f} m/s   '
              f'ψ={math.degrees(state.psi):+6.1f}°   t={elapsed:.0f} s   ',
              end='', flush=True)
        if abs(V - args.airspeed) < AIRSPEED_TOL:
            print()
            break
        if elapsed > AIRSPEED_TIMEOUT:
            print()
            print(f'[WARN] Airspeed not stable after {AIRSPEED_TIMEOUT:.0f} s '
                  f'(V={V:.1f} m/s) — proceeding anyway')
            break
        time.sleep(0.5)
        # Re-send velocity command every 5 s (ArduPlane may time out)
        if int(elapsed) % 5 == 0 and elapsed > 0:
            _cmd_fly_north(conn, args.airspeed)

    print(f'[STARTUP] Cruise conditions reached.  '
          f'h={-buf.read().z:.1f} m   V={buf.read().airspeed:.1f} m/s')
    print('[STARTUP] Ready — handing off to mission.')
    rx.stop()
    sys.exit(0)


if __name__ == '__main__':
    main()
