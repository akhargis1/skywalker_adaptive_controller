"""
generate_waypoints.py — Export lawnmower trajectory as a QGC WPL 110 mission file
                         for ArduPlane AUTO mode (L1 guidance baseline).

Usage
-----
    # Recommended — query home position from running SITL automatically:
    python3 generate_waypoints.py --alt 100 --leg 200 --radius 40 --legs 4 --auto-home

    # Or supply coordinates manually (run 'wp show 0' in MAVProxy to get them):
    python3 generate_waypoints.py --alt 100 --leg 200 --radius 40 --legs 4 \\
        --home-lat -35.363261 --home-lon 149.165230 --home-alt 584.0

    # Then in MAVProxy (after manual takeoff to cruise altitude):
    wp load lawnmower_mission.waypoints
    wp list
    mode AUTO

Home position note
------------------
The waypoints are offsets in NED from the SITL home position.  If the home used
here does not match the SITL home the waypoints will be in the wrong location.
Use --auto-home to detect it automatically, or 'wp show 0' in MAVProxy to read
the home coords and pass them via --home-lat/lon/alt.

Mission structure
-----------------
    WP 0  : HOME  — ground reference (absolute MSL alt, FRAME=0)
    WP 1  : start of lawnmower (NED origin, alt AGL, FRAME=3)
    WP 2..N-1 : leg endpoints in order
    WP N  : RTL  — return to launch

Recommended ArduPilot params for clean L1 tracking
----------------------------------------------------
    WP_RADIUS    30      # waypoint acceptance radius (m)
    NAVL1_PERIOD 20      # L1 period (s)
    NAVL1_DAMPING 0.75
"""

import argparse
import math
import sys
from pathlib import Path

from pymavlink import mavutil

# Allow running from the controller/ directory or repo root
sys.path.insert(0, str(Path(__file__).parent))
from trajectory_generation import LawnmowerTrajectory


# ---------------------------------------------------------------------------
# SITL defaults (Sonoma, CA — from SITL_Models/Gazebo launch script)
# ArduPilot's built-in SITL default is Canberra, AU: -35.363261, 149.165230, 584 m
# ---------------------------------------------------------------------------
SITL_HOME_LAT = 38.161479
SITL_HOME_LON = -122.454630
SITL_HOME_ALT = 488.0   # meters MSL

# QGC WPL 110 MAVLink command IDs
CMD_WAYPOINT = 16
CMD_RTL      = 20
FRAME_GLOBAL = 0   # absolute lat/lon/alt (MSL)
FRAME_GLOBAL_REL_ALT = 3   # altitude relative to home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_home_from_sitl(address: str = 'tcp:127.0.0.1:5762'):
    """
    Connect to a running SITL, request HOME_POSITION, return (lat, lon, alt_msl).
    Raises RuntimeError if no response within 10 s.
    """
    print(f"[auto-home] Connecting to {address} ...")
    conn = mavutil.mavlink_connection(address, autoreconnect=False)
    conn.wait_heartbeat(timeout=15)
    print(f"[auto-home] Heartbeat received.")

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_GET_HOME_POSITION,
        0, 0, 0, 0, 0, 0, 0, 0,
    )
    msg = conn.recv_match(type='HOME_POSITION', blocking=True, timeout=10)
    conn.close()

    if msg is None:
        raise RuntimeError(
            "HOME_POSITION not received within 10 s.\n"
            "Make sure SITL is running and the vehicle is armed or has set home.\n"
            "Alternative: run 'wp show 0' in MAVProxy and pass coords manually."
        )

    lat = msg.latitude  / 1e7   # int32 (1e-7 deg) → float deg
    lon = msg.longitude / 1e7
    alt = msg.altitude  / 1e3   # int32 (mm MSL)  → float m MSL
    print(f"[auto-home] Home detected: lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f} m MSL")
    return lat, lon, alt


def ned_to_gps(x_north: float, y_east: float,
               home_lat: float, home_lon: float):
    """Convert NED offset (m) to GPS decimal degrees."""
    lat = home_lat + x_north / 111111.0
    lon = home_lon + y_east  / (111111.0 * math.cos(math.radians(home_lat)))
    return lat, lon


def _wp_row(index, current, frame, cmd,
            p1, p2, p3, p4, lat, lon, alt, autocontinue=1):
    """Format one QGC WPL 110 tab-separated row."""
    return (f"{index}\t{current}\t{frame}\t{cmd}\t"
            f"{p1:.6f}\t{p2:.6f}\t{p3:.6f}\t{p4:.6f}\t"
            f"{lat:.8f}\t{lon:.8f}\t{alt:.6f}\t{autocontinue}")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_waypoints(altitude:      float,
                       leg_length:    float,
                       turn_radius:   float,
                       num_legs:      int,
                       home_lat:      float = SITL_HOME_LAT,
                       home_lon:      float = SITL_HOME_LON,
                       home_alt:      float = SITL_HOME_ALT,
                       airspeed:      float = 17.0,
                       north_offset:  float = 50.0,
                       wp_spacing:    float = 40.0) -> list[str]:
    """
    Build the QGC WPL 110 mission lines for a lawnmower pattern.

    north_offset : metres north of home where WP1 is placed (keeps pattern
                   away from the takeoff zone so ArduPilot does not skip WP1)
    wp_spacing   : target arc-length spacing between consecutive waypoints (m);
                   smaller values give tighter L1 tracking through turns

    Returns a list of strings (one per line, including the header).
    """
    traj = LawnmowerTrajectory(altitude, airspeed, leg_length, turn_radius, num_legs)

    # Uniform time-sampling → approximately uniform arc-length spacing.
    # Airspeed is constant, so Δt ∝ Δs.  This naturally places more points
    # on long legs and samples the turn arcs (rather than cutting corners).
    total_arc  = (num_legs * leg_length
                  + (num_legs - 1) * math.pi * turn_radius)
    n_samples  = max(2 * num_legs, round(total_arc / wp_spacing) + 1)
    T          = traj.total_time
    ts         = [i * T / (n_samples - 1) for i in range(n_samples)]
    ts[-1]     = T - 1e-6   # stay just inside the last segment

    ned_points = []
    for t in ts:
        pt = traj.query(t)
        ned_points.append((pt.x_ref + north_offset, pt.y_ref))

    lines = ["QGC WPL 110"]

    # WP 0: HOME — absolute MSL alt, ground level
    lines.append(_wp_row(0, 1, FRAME_GLOBAL, CMD_WAYPOINT,
                         0, 0, 0, 0,
                         home_lat, home_lon, home_alt))

    # WP 1 .. N: lawnmower leg endpoints, relative alt (AGL)
    for i, (x_n, y_e) in enumerate(ned_points):
        lat, lon = ned_to_gps(x_n, y_e, home_lat, home_lon)
        lines.append(_wp_row(i + 1, 0, FRAME_GLOBAL_REL_ALT, CMD_WAYPOINT,
                             0, 0, 0, 0,
                             lat, lon, altitude))

    # Final WP: RTL
    rtl_idx = len(ned_points) + 1
    lines.append(_wp_row(rtl_idx, 0, FRAME_GLOBAL_REL_ALT, CMD_RTL,
                         0, 0, 0, 0, 0, 0, 0))

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate QGC WPL 110 waypoints for lawnmower L1 baseline')
    parser.add_argument('--alt',      type=float, required=True,
                        help='Cruise altitude AGL (m)')
    parser.add_argument('--leg',      type=float, required=True,
                        help='Straight leg length (m)')
    parser.add_argument('--radius',   type=float, required=True,
                        help='Turn radius (m)')
    parser.add_argument('--legs',     type=int,   default=4,
                        help='Number of legs (default 4)')
    parser.add_argument('--airspeed', type=float, default=17.0,
                        help='Cruise airspeed m/s (default 17)')
    parser.add_argument('--home-lat',   type=float, default=None,
                        help='Home latitude  (decimal deg); overridden by --auto-home')
    parser.add_argument('--home-lon',   type=float, default=None,
                        help='Home longitude (decimal deg); overridden by --auto-home')
    parser.add_argument('--home-alt',   type=float, default=None,
                        help='Home altitude MSL (m);        overridden by --auto-home')
    parser.add_argument('--north-offset', type=float, default=50.0,
                        help='Shift lawnmower N metres north of home (default 50)')
    parser.add_argument('--wp-spacing',   type=float, default=40.0,
                        help='Target arc-length spacing between waypoints in metres (default 40)')
    parser.add_argument('--auto-home',  action='store_true',
                        help='Query SITL for actual home position (recommended)')
    parser.add_argument('--connect',    type=str,   default='tcp:127.0.0.1:5762',
                        help='MAVLink address used by --auto-home (default tcp:127.0.0.1:5762)')
    parser.add_argument('--output',     type=str,   default='lawnmower_mission.waypoints',
                        help='Output filename (default lawnmower_mission.waypoints)')
    args = parser.parse_args()

    # Resolve home position: --auto-home > explicit flags > built-in Sonoma default
    if args.auto_home:
        home_lat, home_lon, home_alt = fetch_home_from_sitl(args.connect)
    else:
        home_lat = args.home_lat if args.home_lat is not None else SITL_HOME_LAT
        home_lon = args.home_lon if args.home_lon is not None else SITL_HOME_LON
        home_alt = args.home_alt if args.home_alt is not None else SITL_HOME_ALT
        if args.home_lat is None:
            print(f"[warn] No home position specified. Using Sonoma default "
                  f"({SITL_HOME_LAT}, {SITL_HOME_LON}).")
            print("[warn] Use --auto-home (SITL running) or --home-lat/lon/alt to fix.")

    lines = generate_waypoints(
        altitude=args.alt,
        leg_length=args.leg,
        turn_radius=args.radius,
        num_legs=args.legs,
        home_lat=home_lat,
        home_lon=home_lon,
        home_alt=home_alt,
        airspeed=args.airspeed,
        north_offset=args.north_offset,
        wp_spacing=args.wp_spacing,
    )

    output = Path(args.output)
    output.write_text('\n'.join(lines) + '\n')

    print(f"Wrote {len(lines) - 1} waypoints to {output}")
    print(f"  Home : {home_lat:.6f}, {home_lon:.6f}  alt={home_alt} m MSL")
    print(f"  Legs : {args.legs} × {args.leg} m  R={args.radius} m  alt={args.alt} m AGL")
    print(f"  Tip  : if waypoints look far away, run 'wp show 0' in MAVProxy to verify home coords")
    print(f"  Total mission waypoints (excl. home/RTL): {len(lines) - 3}")
    print()
    print("MAVProxy upload:")
    print(f"  wp load {output}")
    print( "  wp list")
    print( "  mode AUTO")
    print()
    print("Recommended params:")
    print("  param set WP_RADIUS 30")
    print("  param set NAVL1_PERIOD 20")
    print("  param set NAVL1_DAMPING 0.75")


if __name__ == '__main__':
    main()
