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

#for label, l, r in tests:
#    print(f"\n--- {label} ---")
#    send(l, r)
#    time.sleep(3)   # watch the Gazebo joint topic for 3 seconds

# Release override
#conn.mav.rc_channels_override_send(
#    conn.target_system, conn.target_component,
#    0, 0, 0, 0, 0, 0, 0, 0,
#)
#print("\nDone. Override released.")

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

def angle_to_pwm(degrees, reverse=False):
    """
    Maps -30 to +30 degrees to 1100-1900 PWM.
    Adjust 'limit' and 'range' based on your specific aircraft model.
    """
    limit = 30.0
    # Constrain input
    degrees = max(-limit, min(limit, degrees))
    
    if reverse:
        degrees = -degrees
        
    # Simple linear mapping: 1500 +/- 400
    return int(1500 + (degrees / limit) * 400)

try:
    print("\n--- Starting Direct Control Test ---")
    
    # 1. Symmetric Up (Pitch Up)
    # Both elevons move the same way
    print("Commanding: Symmetric Up (+20°)")
    set_servo_pwm(1, angle_to_pwm(20)) # Left
    set_servo_pwm(2, angle_to_pwm(20)) # Right
    time.sleep(3)

    # 2. Neutral
    print("Commanding: Neutral")
    set_servo_pwm(1, 1500)
    set_servo_pwm(2, 1500)
    time.sleep(2)

    # 3. Differential (Roll Right)
    # Left goes up, Right goes down
    print("Commanding: Roll Right")
    set_servo_pwm(1, angle_to_pwm(20))  # Left Up
    set_servo_pwm(2, angle_to_pwm(-20)) # Right Down
    time.sleep(3)

finally:
    # 4. Cleanup: Always return to neutral
    print("\nCleaning up... Resetting servos to 1500.")
    set_servo_pwm(1, 1500)
    set_servo_pwm(2, 1500)
    print("Done.")

