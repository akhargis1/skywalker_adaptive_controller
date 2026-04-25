import math, time
from pymavlink import mavutil

conn = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
conn.wait_heartbeat()

def to_pwm(deg, limit=30.0):
    pct = max(-1.0, min(1.0, deg / limit))
    return int(1500 + pct * 400)

def send(left_deg, right_deg):
    ch1 = to_pwm(left_deg)   # SERVO1 = left elevon
    ch2 = to_pwm(right_deg)  # SERVO2 = right elevon
    print(f"Sending L={left_deg:+.0f}° ({ch1} µs)  R={right_deg:+.0f}° ({ch2} µs)")
    conn.mav.rc_channels_override_send(
        conn.target_system, conn.target_component,
        ch1, ch2, 65535, 65535, 65535, 65535, 65535, 65535,
    )

tests = [
    ("neutral",          0,    0),
    ("symmetric up",   +15,  +15),   # watch: nose up or down?
    ("neutral",          0,    0),
    ("differential",   +15,  -15),   # watch: roll left or right?
    ("neutral final",    0,    0),
]

for label, l, r in tests:
    print(f"\n--- {label} ---")
    send(l, r)
    time.sleep(3)   # watch the Gazebo joint topic for 3 seconds

# Release override
conn.mav.rc_channels_override_send(
    conn.target_system, conn.target_component,
    0, 0, 0, 0, 0, 0, 0, 0,
)
print("\nDone. Override released.")
