"""
x8_mavlink.py — MAVLink connection management.

Handles everything that touches the wire:
  - connecting to ArduPilot SITL
  - background thread that reads incoming messages into a StateBuffer
  - functions that send attitude targets and RC overrides

Nothing in here knows about the control law.
"""

import copy
import math
import threading
import time
from dataclasses import dataclass

from pymavlink import mavutil


# ---------------------------------------------------------------------------
# Vehicle state snapshot
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    # Attitude  (radians)
    phi:      float = 0.0
    theta:    float = 0.0
    psi:      float = 0.0
    # Body rates  (rad/s)
    p:        float = 0.0
    q:        float = 0.0
    r:        float = 0.0
    # Air data
    airspeed: float = 17.0
    alpha:    float = 0.0    # rad — needs AOA_ENABLE=1 in ArduPilot params
    beta:     float = 0.0    # rad
    # Vehicle status
    armed:    bool  = False
    mode:     int   = -1
    # Freshness flag
    valid:    bool  = False


class StateBuffer:
    """Thread-safe latest-value store written by rx thread, read by main loop."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._state = VehicleState()

    def write(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)
            self._state.valid = True

    def read(self) -> VehicleState:
        with self._lock:
            return copy.copy(self._state)


# ---------------------------------------------------------------------------
# Receive thread
# ---------------------------------------------------------------------------

class MAVReceiver(threading.Thread):
    """
    Reads MAVLink messages in a background daemon thread.
    Writes parsed fields into StateBuffer.

    Messages consumed:
        ATTITUDE    → phi, theta, psi, p, q, r
        VFR_HUD     → airspeed
        AOA_SSA     → alpha, beta  (set AOA_ENABLE=1 in ArduPilot)
        HEARTBEAT   → armed, mode
    """

    def __init__(self, conn, buf: StateBuffer, verbose: bool = False):
        super().__init__(daemon=True, name="MAVReceiver")
        self.conn    = conn
        self.buf     = buf
        self.verbose = verbose
        self._stop   = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            t = msg.get_type()

            if t == 'ATTITUDE':
                self.buf.write(
                    phi=msg.roll, theta=msg.pitch, psi=msg.yaw,
                    p=msg.rollspeed, q=msg.pitchspeed, r=msg.yawspeed,
                )
            elif t == 'VFR_HUD':
                self.buf.write(airspeed=max(float(msg.airspeed), 1.0))
            elif t == 'AOA_SSA':
                self.buf.write(
                    alpha=math.radians(msg.AOA),
                    beta=math.radians(msg.SSA),
                )
            elif t == 'HEARTBEAT':
                armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.buf.write(armed=armed, mode=int(msg.custom_mode))
            elif t == 'STATUSTEXT' and self.verbose:
                print(f"[AP] {msg.text}")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

GUIDED_MODE = 15
FBWA_MODE   = 2


def connect(address: str, stream_hz: int = 50) -> mavutil.mavfile:
    """
    Open MAVLink connection, wait for heartbeat, request attitude stream.

    address examples:
        'tcp:127.0.0.1:5762'   (SITL default)
        'udp:127.0.0.1:14550'
        '/dev/ttyUSB0,57600'   (telemetry radio)
    """
    print(f"[MAV] Connecting to {address} ...")
    conn = mavutil.mavlink_connection(address, autoreconnect=True)
    conn.wait_heartbeat(timeout=15)
    print(f"[MAV] Heartbeat from system {conn.target_system} "
          f"component {conn.target_component}")

    # Request ATTITUDE at stream_hz, VFR_HUD at 20 Hz
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, stream_hz, 1,
    )
    return conn


def wait_for_state(buf: StateBuffer, timeout: float = 10.0) -> bool:
    """Block until StateBuffer contains at least one valid snapshot."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if buf.read().valid:
            return True
        time.sleep(0.05)
    return False
  
def get_trim_pwm(conn, samples: int = 25, interval: float = 0.02) -> tuple:
    """
    Sample SERVO_OUTPUT_RAW while ArduPlane is flying level.
    Returns (servo1_avg_pwm, servo2_avg_pwm) — the trim PWM for each elevon.
    Call this BEFORE switching to research mode.
    """
    s1_samples, s2_samples = [], []
    for _ in range(samples):
        msg = conn.recv_match(type='SERVO_OUTPUT_RAW', blocking=True, timeout=0.5)
        if msg:
            s1_samples.append(msg.servo1_raw)
            s2_samples.append(msg.servo2_raw)
        time.sleep(interval)

    if not s1_samples:
        print("[TRIM] WARNING: No SERVO_OUTPUT_RAW received — defaulting to 1500")
        return 1500.0, 1500.0

    s1 = sum(s1_samples) / len(s1_samples)
    s2 = sum(s2_samples) / len(s2_samples)
    print(f"[TRIM] Servo1={s1:.0f}µs  Servo2={s2:.0f}µs  ({len(s1_samples)} samples)")
    return s1, s2


def set_mode(conn, mode_num: int):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_num, 0, 0, 0, 0, 0,
    )


# ---------------------------------------------------------------------------
# Output senders
# ---------------------------------------------------------------------------

def send_attitude_target(conn,
                         roll_d:       float,
                         pitch_d:      float,
                         yaw_d:        float,
                         roll_rate_d:  float = 0.0,
                         pitch_rate_d: float = 0.0,
                         yaw_rate_d:   float = 0.0,
                         thrust:       float = 0.6):
    """
    Send SET_ATTITUDE_TARGET.
    ArduPlane uses the quaternion for attitude hold; rates are feedforward.
    """
    cy, sy = math.cos(yaw_d   / 2), math.sin(yaw_d   / 2)
    cp, sp = math.cos(pitch_d / 2), math.sin(pitch_d / 2)
    cr, sr = math.cos(roll_d  / 2), math.sin(roll_d  / 2)
    q = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    conn.mav.set_attitude_target_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,
        conn.target_system,
        conn.target_component,
        0b00000000,
        q,
        roll_rate_d, pitch_rate_d, yaw_rate_d,
        thrust,
    )


def send_rc_override(conn,
                     delta_L_rad: float,
                     delta_R_rad: float,
                     throttle_pct: float = 60.0,
                     elevon_limit_deg: float = 30.0):
    """
    Alternative output: override RC channels directly.
    X8 default ArduPilot mixer: Ch1 = right elevon, Ch2 = left elevon.
    ±elevon_limit_deg → PWM 1000–2000 µs.
    """
    def to_pwm(rad: float) -> int:
        pct = max(-1.0, min(1.0, math.degrees(rad) / elevon_limit_deg))
        return int(1500 + pct * 400)

    conn.mav.rc_channels_override_send(
        conn.target_system, conn.target_component,
        to_pwm(delta_R_rad),            # Ch1
        to_pwm(delta_L_rad),            # Ch2
        int(1000 + throttle_pct * 10),  # Ch3 throttle
        65535, 65535, 65535, 65535, 65535,
    )

def toggle_research_mode(connection, active=True):
        """Switches SERVO_FUNCTION between 1 (PassThru) and 77/78 (Elevon)"""
        val_l = 1 if active else 77
        val_r = 1 if active else 78
        
        params = [('SERVO1_FUNCTION', val_l), ('SERVO2_FUNCTION', val_r)]
        for p_name, p_val in params:
            connection.mav.param_set_send(
                connection.target_system, connection.target_component,
                p_name.encode('utf-8'), p_val, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
        print(f"Mode Switched: {'Direct Research Control' if active else 'ArduPilot Internal'}")
        

def send_elevon_direct(conn,
                       delta_L_rad:      float,
                       delta_R_rad:      float,
                       throttle_pct:     float = 60.0,
                       elevon_limit_deg: float = 30.0):
    """
    Send direct elevon commands via RC override.

    Verified channel mapping from hardware test:
        Ch1 = SERVO1 = left elevon   (SERVO1_FUNCTION=77)
        Ch2 = SERVO2 = right elevon  (SERVO2_FUNCTION=78)

    PWM range [1100, 1900] matching SDF servo_min/servo_max.
    Right elevon negated to correct for mirrored hinge axis.

    delta_L_rad, delta_R_rad:  signed deflection in radians
        positive = trailing edge up
        negative = trailing edge down
    """
    def to_pwm(rad: float) -> int:
        pct = max(-1.0, min(1.0, math.degrees(rad) / elevon_limit_deg))
        return int(1500 + pct * 400)   # 400 → maps to [1100, 1900]


    def set_servo_pwm(servo_n, pwm):
        """
        Directly sets a PWM value to a specific output pin.
        servo_n: 1 for SERVO1, 2 for SERVO2, etc.
        pwm: 1000 to 2000 (1500 is neutral)
        """
        conn.mav.command_long_send(
            conn.target_system,
            conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,            # Confirmation
            servo_n,      # Instance (Servo Number)
            pwm,          # PWM Value
            0, 0, 0, 0, 0 # Unused parameters
        )

    conn.mav.rc_channels_override_send(
        conn.target_system, conn.target_component,
        65535, 65535,
        int(1000 + throttle_pct * 10),  # Ch3 = throttle
        65535, 65535, 65535, 65535, 65535
    )
    
    set_servo_pwm(1, to_pwm(delta_L_rad))
    set_servo_pwm(2, to_pwm(delta_R_rad))

    
