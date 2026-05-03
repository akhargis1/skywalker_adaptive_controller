"""
inject_wind.py — Inject wind disturbances into ArduPlane SITL via MAVLink PARAM_SET.

Sets SIM_WIND_SPD, SIM_WIND_DIR, and SIM_WIND_TURB on a running SITL instance.
Runs independently of x8_run.py (separate terminal, same MAVLink connection address).

Wind direction convention (ArduPilot meteorological):
    0   = wind blowing FROM the North  (southward flow)
    90  = wind blowing FROM the East   (westward flow)
    180 = wind blowing FROM the South
    270 = wind blowing FROM the West   (eastward flow)

Usage
-----
    # Reset to calm
    python3 sim/inject_wind.py --mode step --speed 0

    # 5 m/s from the west for 60 s
    python3 sim/inject_wind.py --mode step --speed 5 --dir 270 --duration 60

    # Sinusoidal ±4 m/s @ 0.1 Hz from west, 90 s
    python3 sim/inject_wind.py --mode sine --amp 4 --freq 0.1 --dir 270 --duration 90

    # Ramp to 8 m/s over 10 s, hold for 50 s
    python3 sim/inject_wind.py --mode ramp --speed 8 --ramp-time 10 --duration 60

    # 4-phase test sequence (calm straight → calm turn → step turn → sine turn)
    python3 sim/inject_wind.py --mode sequence \\
        --gust-speed 5 --gust-dir 270 --leg-time 30 --turn-time 20

Verification (with SITL running):
    MAVProxy> param show SIM_WIND_SPD   # should change within 1 s of script start
    MAVProxy> param show SIM_WIND_DIR
"""

import argparse
import math
import time

from pymavlink import mavutil


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def connect(address: str):
    print(f"[wind] Connecting to {address} ...")
    conn = mavutil.mavlink_connection(address, autoreconnect=False)
    conn.wait_heartbeat(timeout=15)
    print(f"[wind] Heartbeat received (sys={conn.target_system} comp={conn.target_component})")
    return conn


def _set_param(conn, name: str, value: float):
    conn.mav.param_set_send(
        conn.target_system,
        conn.target_component,
        name.encode(),
        value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    time.sleep(0.05)


def set_wind(conn, speed: float, direction: float, turb: float = 0.0):
    """Apply constant wind (speed m/s, direction deg meteorological, turb m/s σ)."""
    _set_param(conn, 'SIM_WIND_SPD',  speed)
    _set_param(conn, 'SIM_WIND_DIR',  direction)
    _set_param(conn, 'SIM_WIND_TURB', turb)


def calm(conn):
    set_wind(conn, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Gust profiles
# ---------------------------------------------------------------------------

def run_step(conn, args):
    """Constant wind for --duration seconds, then calm."""
    print(f"[wind] STEP  speed={args.speed} m/s  dir={args.dir}°  duration={args.duration} s")
    set_wind(conn, args.speed, args.dir, getattr(args, 'turb', 0.0))
    time.sleep(args.duration)
    print("[wind] Duration elapsed — resetting to calm")
    calm(conn)


def run_sine(conn, args):
    """
    Sinusoidal speed modulation: speed(t) = amp * sin(2π·freq·t).
    Negative half-cycles set speed=0 (wind can't blow backwards without changing direction).
    Direction is fixed at --dir.
    """
    print(f"[wind] SINE  amp={args.amp} m/s  freq={args.freq} Hz  "
          f"dir={args.dir}°  duration={args.duration} s")
    dt = 0.1          # update interval (s)
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= args.duration:
            break
        spd = args.amp * math.sin(2 * math.pi * args.freq * elapsed)
        spd = max(0.0, spd)   # clamp negative half to calm
        set_wind(conn, spd, args.dir, 0.0)
        time.sleep(dt)
    print("[wind] Duration elapsed — resetting to calm")
    calm(conn)


def run_ramp(conn, args):
    """Linear ramp from 0 → --speed over --ramp-time, then hold for remainder of --duration."""
    print(f"[wind] RAMP  speed={args.speed} m/s  ramp-time={args.ramp_time} s  "
          f"dir={args.dir}°  total={args.duration} s")
    dt = 0.1
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= args.duration:
            break
        if elapsed < args.ramp_time:
            spd = args.speed * elapsed / args.ramp_time
        else:
            spd = args.speed
        set_wind(conn, spd, args.dir, 0.0)
        time.sleep(dt)
    print("[wind] Duration elapsed — resetting to calm")
    calm(conn)


def run_sequence(conn, args):
    """
    4-phase reproducible test sequence for SMC vs L1 comparison:
      Phase 1 — calm, straight leg      (leg-time s)
      Phase 2 — calm, turn segment      (turn-time s)
      Phase 3 — step gust, turn         (turn-time s)
      Phase 4 — sinusoidal gust, turn   (turn-time s)
    """
    L = args.leg_time
    T = args.turn_time
    spd = args.gust_speed
    d   = args.gust_dir

    def phase(n, label, wind_speed, direction):
        duration = L if n == 1 else T
        print(f"[wind] Phase {n}: {label}  ({duration} s)  "
              f"wind={wind_speed:.1f} m/s @ {direction}°")
        set_wind(conn, wind_speed, direction)
        time.sleep(duration)

    print(f"[wind] SEQUENCE  gust={spd} m/s  dir={d}°  "
          f"leg={L} s  turn={T} s")

    phase(1, "calm straight", 0.0, 0.0)
    phase(2, "calm turn",     0.0, 0.0)
    phase(3, "step-gust turn", spd, d)

    # Phase 4: sinusoidal gust on a turn
    print(f"[wind] Phase 4: sine-gust turn  ({T} s)  amp={spd} m/s @ {d}°")
    dt = 0.1
    t0 = time.monotonic()
    while time.monotonic() - t0 < T:
        elapsed = time.monotonic() - t0
        s = spd * math.sin(2 * math.pi * 0.1 * elapsed)
        set_wind(conn, max(0.0, s), d)
        time.sleep(dt)

    print("[wind] Sequence complete — resetting to calm")
    calm(conn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Inject SITL wind disturbances via MAVLink PARAM_SET')
    parser.add_argument('--connect', default='tcp:127.0.0.1:5762',
                        help='MAVLink address (default tcp:127.0.0.1:5762)')
    parser.add_argument('--mode', required=True,
                        choices=['step', 'sine', 'ramp', 'sequence'],
                        help='Gust profile')

    # shared
    parser.add_argument('--dir',      type=float, default=270.0,
                        help='Wind direction deg meteorological (default 270 = from West)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duration to hold wind before resetting to calm (s)')
    parser.add_argument('--turb',     type=float, default=0.0,
                        help='Turbulence intensity σ (m/s, default 0 = laminar)')

    # step / ramp
    parser.add_argument('--speed',    type=float, default=5.0,
                        help='Wind speed (m/s) for step/ramp')

    # sine
    parser.add_argument('--amp',  type=float, default=4.0,
                        help='Sine amplitude (m/s)')
    parser.add_argument('--freq', type=float, default=0.1,
                        help='Sine frequency (Hz, default 0.1)')

    # ramp
    parser.add_argument('--ramp-time', type=float, default=10.0,
                        help='Time (s) to ramp from 0 to --speed')

    # sequence
    parser.add_argument('--gust-speed', type=float, default=5.0,
                        help='Gust speed for sequence phases (m/s)')
    parser.add_argument('--gust-dir',   type=float, default=270.0,
                        help='Gust direction for sequence phases (deg)')
    parser.add_argument('--leg-time',   type=float, default=30.0,
                        help='Duration of straight-leg phases in sequence (s)')
    parser.add_argument('--turn-time',  type=float, default=20.0,
                        help='Duration of turn phases in sequence (s)')

    args = parser.parse_args()
    conn = connect(args.connect)

    if args.mode == 'step':
        run_step(conn, args)
    elif args.mode == 'sine':
        run_sine(conn, args)
    elif args.mode == 'ramp':
        run_ramp(conn, args)
    elif args.mode == 'sequence':
        run_sequence(conn, args)

    conn.close()
    print("[wind] Done.")


if __name__ == '__main__':
    main()
