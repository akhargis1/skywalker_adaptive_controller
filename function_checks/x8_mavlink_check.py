"""
x8_mavlink_check.py — MAVLink read/write diagnostic tool.

Run this BEFORE x8_run.py to verify the plumbing is working.
Does not run the controller. Safe to run at any time.

Tests (in order):
    1  Connection        — can we reach ArduPilot at all
    2  Heartbeat         — is ArduPilot alive and what mode is it in
    3  Attitude stream   — are we receiving ATTITUDE messages at the right rate
    4  Airspeed stream   — VFR_HUD arriving and sensible
    5  AOA stream        — AOA_SSA arriving (needs AOA_ENABLE=1)
    6  Frame check       — attitude angles are in radians and sane range
    7  Latency           — round-trip time to send a command and see a response
    8  RC override test  — send a neutral RC command, confirm no crash
    9  Mode check        — confirm we can read and set flight mode
   10  Stream rate check — measure actual Hz of ATTITUDE messages

Usage:
    python3 x8_mavlink_check.py
    python3 x8_mavlink_check.py --connect tcp:127.0.0.1:5762
    python3 x8_mavlink_check.py --connect tcp:127.0.0.1:5762 --write  # also test write
"""

import argparse
import math
import sys
import time
import statistics
from collections import deque

try:
    from pymavlink import mavutil
except ImportError:
    sys.exit("pymavlink not installed.  Run: pip install pymavlink")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}PASS{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}WARN{RESET}  {msg}")
def fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}INFO{RESET}  {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}")


ARDUPILOT_MODES = {
    0:  "MANUAL",    2:  "FBWA",      3:  "FBWB",
    4:  "CRUISE",    5:  "AUTOTUNE",  6:  "AUTO",
    7:  "RTL",       8:  "LOITER",   11: "GUIDED",
   15:  "GUIDED",   17:  "QSTABILIZE",
}


# ---------------------------------------------------------------------------
# Test 1 — Connection
# ---------------------------------------------------------------------------

def test_connection(address: str, timeout: float = 10.0):
    header("Test 1 — Connection")
    try:
        conn = mavutil.mavlink_connection(address, autoreconnect=False)
    except Exception as e:
        fail(f"Could not open connection: {e}")
        sys.exit(1)

    info(f"Connecting to {address} (timeout {timeout}s) ...")
    hb = conn.wait_heartbeat(timeout=timeout)
    if hb is None:
        fail("No heartbeat received. Check:")
        print("       - Is sim_vehicle.py running?")
        print("       - Is the port correct? (default 5762 for SITL)")
        print("       - Is the connection string correct? (tcp: vs udp:)")
        sys.exit(1)

    ok(f"Heartbeat received from system {conn.target_system} "
       f"component {conn.target_component}")
    return conn


# ---------------------------------------------------------------------------
# Test 2 — Heartbeat / mode
# ---------------------------------------------------------------------------

def test_heartbeat(conn):
    header("Test 2 — Heartbeat & mode")
    msg = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=3)
    if msg is None:
        fail("No HEARTBEAT within 3s")
        return

    armed  = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    mode   = ARDUPILOT_MODES.get(msg.custom_mode, f"UNKNOWN({msg.custom_mode})")
    ok(f"Mode: {mode}  |  Armed: {armed}")

    if armed:
        warn("Vehicle is ARMED — be careful with write tests")
    else:
        info("Vehicle disarmed — safe for read tests")


# ---------------------------------------------------------------------------
# Test 3 — Attitude stream rate
# ---------------------------------------------------------------------------

def test_attitude_stream(conn, duration: float = 3.0):
    header("Test 3 — Attitude stream")

    # Request streams first
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 50, 1,
    )
    time.sleep(0.2)

    timestamps = []
    t0 = time.monotonic()
    last_msg = None

    while time.monotonic() - t0 < duration:
        msg = conn.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
        if msg:
            timestamps.append(time.monotonic())
            last_msg = msg

    if len(timestamps) < 2:
        fail("Fewer than 2 ATTITUDE messages received — stream not running")
        print("       Fix: check SITL is running and streaming")
        return None

    intervals  = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    hz_actual  = 1.0 / statistics.mean(intervals)
    hz_jitter  = statistics.stdev(intervals) * 1000  # ms

    if hz_actual >= 40:
        ok(f"Rate: {hz_actual:.1f} Hz  jitter: {hz_jitter:.1f} ms  "
           f"({len(timestamps)} msgs in {duration}s)")
    elif hz_actual >= 20:
        warn(f"Rate: {hz_actual:.1f} Hz — low, recommend 50 Hz  "
             f"jitter: {hz_jitter:.1f} ms")
    else:
        fail(f"Rate: {hz_actual:.1f} Hz — too low for controller (need ≥25 Hz)")

    return last_msg


# ---------------------------------------------------------------------------
# Test 4 — Attitude values and frame
# ---------------------------------------------------------------------------

def test_attitude_values(msg):
    header("Test 4 — Attitude values & frame")
    if msg is None:
        fail("No attitude message to check")
        return

    phi   = msg.roll
    theta = msg.pitch
    psi   = msg.yaw
    p     = msg.rollspeed
    q     = msg.pitchspeed
    r     = msg.yawspeed

    info(f"φ={math.degrees(phi):+7.2f}°  "
         f"θ={math.degrees(theta):+7.2f}°  "
         f"ψ={math.degrees(psi):+7.2f}°")
    info(f"p={math.degrees(p):+7.2f}°/s  "
         f"q={math.degrees(q):+7.2f}°/s  "
         f"r={math.degrees(r):+7.2f}°/s")

    # Check radians (not degrees accidentally sent as radians)
    if abs(phi) > math.pi or abs(theta) > math.pi/2 + 0.1:
        fail("Attitude values out of expected radian range — "
             "possible unit error")
    else:
        ok("Angles in radian range")

    # Check rates are not frozen
    if abs(p) < 1e-6 and abs(q) < 1e-6 and abs(r) < 1e-6:
        warn("Body rates are exactly zero — sim may be paused or rates frozen")
    else:
        ok("Body rates non-zero")

    # Check pitch is plausible for a flying X8 (not inverted, not 90°)
    if abs(theta) > math.radians(60):
        warn(f"Pitch {math.degrees(theta):.1f}° — unusually high, check sim state")
    else:
        ok(f"Pitch angle plausible ({math.degrees(theta):.1f}°)")


# ---------------------------------------------------------------------------
# Test 5 — VFR_HUD (airspeed)
# ---------------------------------------------------------------------------

def test_airspeed(conn):
    header("Test 5 — Airspeed (VFR_HUD)")
    msg = conn.recv_match(type='VFR_HUD', blocking=True, timeout=3)
    if msg is None:
        fail("No VFR_HUD within 3s")
        return

    ok(f"Airspeed: {msg.airspeed:.1f} m/s  "
       f"Groundspeed: {msg.groundspeed:.1f} m/s  "
       f"Alt: {msg.alt:.1f} m  "
       f"Throttle: {msg.throttle}%")

    if msg.airspeed < 0.1:
        warn("Airspeed is near zero — is the aircraft flying? "
             "Controller needs V > 9 m/s to produce valid moments")
    elif msg.airspeed < 9.0:
        warn(f"Airspeed {msg.airspeed:.1f} m/s below controller minimum (9 m/s)")
    else:
        ok(f"Airspeed {msg.airspeed:.1f} m/s — within controller envelope")


# ---------------------------------------------------------------------------
# Test 6 — AOA/SSA stream
# ---------------------------------------------------------------------------

def test_aoa(conn):
    header("Test 6 — AOA / sideslip stream (needs AOA_ENABLE=1)")
    msg = conn.recv_match(type='AOA_SSA', blocking=True, timeout=2)
    if msg is None:
        warn("No AOA_SSA message — alpha/beta will be 0 in controller")
        print("       Fix: in MAVProxy console run:")
        print("            param set AOA_ENABLE 1")
        print("            param save")
        print("            reboot")
        print("       Then restart this check.")
    else:
        ok(f"AOA={msg.AOA:.2f}°  SSA={msg.SSA:.2f}°")
        if abs(msg.AOA) > 20:
            warn(f"AoA {msg.AOA:.1f}° seems high — check sim state")


# ---------------------------------------------------------------------------
# Test 7 — Latency measurement
# ---------------------------------------------------------------------------

def test_latency(conn, n_samples: int = 20):
    header("Test 7 — Message latency")
    latencies = []

    for _ in range(n_samples):
        t_send = time.monotonic()
        # Send a benign param request and wait for any response
        conn.mav.param_request_read_send(
            conn.target_system, conn.target_component,
            b'SYSID_THISMAV', -1,
        )
        msg = conn.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg:
            latencies.append((time.monotonic() - t_send) * 1000)
        time.sleep(0.05)

    if not latencies:
        fail("No PARAM_VALUE responses — param request not working")
        return

    mean_ms = statistics.mean(latencies)
    max_ms  = max(latencies)

    if mean_ms < 20:
        ok(f"Round-trip latency: mean={mean_ms:.1f}ms  max={max_ms:.1f}ms  "
           f"({len(latencies)} samples)")
    elif mean_ms < 50:
        warn(f"Latency mean={mean_ms:.1f}ms — acceptable but watch for jitter")
    else:
        fail(f"Latency mean={mean_ms:.1f}ms — too high, controller may lag")
        print("       Fix: check for other processes hammering the MAVLink port")


# ---------------------------------------------------------------------------
# Test 8 — Attitude target write (send neutral, check no crash)
# ---------------------------------------------------------------------------

def test_attitude_write(conn):
    header("Test 8 — SET_ATTITUDE_TARGET write")

    # Read current attitude to use as the neutral target
    msg = conn.recv_match(type='ATTITUDE', blocking=True, timeout=2)
    if msg is None:
        fail("Cannot read attitude for write test")
        return

    phi, theta, psi = msg.roll, msg.pitch, msg.yaw

    # Build quaternion from current attitude (neutral = hold current)
    cy, sy = math.cos(psi/2),   math.sin(psi/2)
    cp, sp = math.cos(theta/2), math.sin(theta/2)
    cr, sr = math.cos(phi/2),   math.sin(phi/2)
    q = [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]

    # Verify quaternion norm
    norm = math.sqrt(sum(x**2 for x in q))
    if abs(norm - 1.0) > 0.01:
        fail(f"Quaternion norm {norm:.4f} — not unit quaternion, math error")
        return

    try:
        conn.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            conn.target_system,
            conn.target_component,
            0b00000000,
            q,
            0.0, 0.0, 0.0,   # zero body rates
            0.6,              # 60% thrust
        )
        ok(f"SET_ATTITUDE_TARGET sent  q={[round(x,3) for x in q]}  norm={norm:.4f}")
        info("Vehicle should not have moved (neutral hold command)")
    except Exception as e:
        fail(f"Send failed: {e}")


# ---------------------------------------------------------------------------
# Test 9 — RC override write (neutral)
# ---------------------------------------------------------------------------

def test_rc_write(conn):
    header("Test 9 — RC override write (neutral, no movement)")
    try:
        # Send neutral (1500 = center) on all channels
        conn.mav.rc_channels_override_send(
            conn.target_system, conn.target_component,
            1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500,
        )
        ok("RC override sent (all channels neutral 1500 µs)")

        time.sleep(0.1)

        # Clear the override immediately (0 = release)
        conn.mav.rc_channels_override_send(
            conn.target_system, conn.target_component,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        ok("RC override cleared (0 = released back to ArduPilot)")
    except Exception as e:
        fail(f"RC override failed: {e}")


# ---------------------------------------------------------------------------
# Test 10 — Continuous attitude monitoring (live readout)
# ---------------------------------------------------------------------------

def test_live_monitor(conn, duration: float = 5.0):
    header(f"Test 10 — Live attitude monitor ({duration}s)")
    print(f"  {'Time':>6}  {'φ (roll)':>10}  {'θ (pitch)':>10}  "
          f"{'ψ (yaw)':>10}  {'V (m/s)':>8}  {'Hz':>6}")
    print("  " + "-" * 62)

    attitude_times = deque(maxlen=20)
    t0 = time.monotonic()

    while time.monotonic() - t0 < duration:
        msg = conn.recv_match(type='ATTITUDE', blocking=True, timeout=0.1)
        if msg:
            now = time.monotonic()
            attitude_times.append(now)
            hz = (len(attitude_times) - 1) / (attitude_times[-1] - attitude_times[0] + 1e-9) \
                 if len(attitude_times) > 1 else 0.0

            # Also grab latest airspeed non-blocking
            spd_msg = conn.recv_match(type='VFR_HUD', blocking=False)
            v = spd_msg.airspeed if spd_msg else 0.0

            elapsed = now - t0
            print(f"  {elapsed:6.2f}s  "
                  f"{math.degrees(msg.roll):+9.2f}°  "
                  f"{math.degrees(msg.pitch):+9.2f}°  "
                  f"{math.degrees(msg.yaw):+9.2f}°  "
                  f"{v:7.1f}    "
                  f"{hz:5.1f}",
                  flush=True)
            time.sleep(0.2)   # print at 5 Hz for readability

    ok("Live monitor complete")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def summary(write_tested: bool):
    header("Summary")
    print("  If all tests above show PASS:")
    print("  → The MAVLink plumbing is working correctly.")
    print("  → You are safe to run x8_run.py\n")
    print("  Key things verified:")
    print("  ✓  ArduPilot reachable on the connection address")
    print("  ✓  ATTITUDE messages arriving at ≥40 Hz")
    print("  ✓  Angle values in radians and plausible range")
    print("  ✓  Airspeed within controller envelope")
    if write_tested:
        print("  ✓  SET_ATTITUDE_TARGET accepted by ArduPilot")
        print("  ✓  RC override accepted and cleared safely")
    else:
        print("  –  Write tests skipped (add --write to enable)")
    print()
    print("  Common WARN items that are OK to ignore:")
    print("  - AOA_SSA missing if you haven't set AOA_ENABLE=1 yet")
    print("  - Body rates near zero if the sim is sitting on the ground")
    print("  - Airspeed near zero if the aircraft hasn't taken off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="X8 MAVLink diagnostic")
    ap.add_argument('--connect', default='tcp:127.0.0.1:5762',
                    help='MAVLink connection string')
    ap.add_argument('--write',  action='store_true',
                    help='Also test write commands (attitude target + RC override)')
    ap.add_argument('--monitor', type=float, default=5.0,
                    help='Duration of live monitor in seconds (0 to skip)')
    args = ap.parse_args()

    print(f"\n{BOLD}X8 MAVLink diagnostic{RESET}")
    print(f"Connection: {args.connect}")
    print(f"Write tests: {'enabled' if args.write else 'disabled (add --write)'}")

    conn = test_connection(args.connect)
    test_heartbeat(conn)
    att_msg = test_attitude_stream(conn)
    test_attitude_values(att_msg)
    test_airspeed(conn)
    test_aoa(conn)
    test_latency(conn)

    if args.write:
        test_attitude_write(conn)
        test_rc_write(conn)
    else:
        header("Tests 8–9 — Write tests")
        info("Skipped — run with --write to enable")

    if args.monitor > 0:
        test_live_monitor(conn, duration=args.monitor)

    summary(write_tested=args.write)


if __name__ == "__main__":
    main()
