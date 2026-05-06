#!/usr/bin/env bash
# run_sitl.sh — Headless SITL + Gazebo automation for Skywalker X8 SMC data collection.
#
# Launches Gazebo (server-only) and ArduPlane SITL, arms and takes off the
# aircraft to cruise altitude/airspeed heading north, optionally injects wind,
# then runs the lawnmower SMC mission and saves a .npz log.
#
# Usage
# -----
#   bash sim/run_sitl.sh [options]
#
# Key options
#   --speedup N         Sim speedup (SITL -S + Gazebo RTF); default 1
#   --alt M             Cruise altitude AGL, m; default 100
#   --leg M             Lawnmower leg length, m; default 200
#   --radius M          Turn radius, m; default 40
#   --legs N            Number of legs; default 4
#   --airspeed M        Cruise airspeed, m/s; default 17
#   --runway M          North lead-in before grid, m; default 200
#   --max-time S        Mission time limit, s; default 600
#   --log PREFIX        NPZ log prefix; default sitl_smc
#   --wind-mode MODE    step|sine|ramp|sequence  (omit = no wind)
#   --wind-speed M      Step/ramp wind speed, m/s; default 5
#   --wind-dir D        Wind direction, deg meteorological; default 270 (from west)
#   --wind-turb T       Turbulence σ, m/s; default 0
#   --wind-amp M        Sine amplitude, m/s; default 4
#   --wind-freq F       Sine frequency, Hz; default 0.1
#   --wind-ramp T       Ramp-up time, s; default 10
#   --wind-duration S   Wind duration, s; default max-time + 60
#   --gust-speed M      Sequence gust speed, m/s; default 5
#   --gust-dir D        Sequence gust direction, deg; default 270
#   --leg-time S        Sequence straight-leg duration, s; default 30
#   --turn-time S       Sequence turn duration, s; default 20
#
# Wind modes
#   step      Constant wind for --wind-duration s (set before takeoff)
#   sine      Sinusoidal gust profile (started after reaching cruise)
#   ramp      Linear ramp to target speed (started after reaching cruise)
#   sequence  4-phase calm/gust test sequence (started after reaching cruise)
#
# Speedup notes
#   --speedup N passes -S N to SITL AND injects <real_time_factor>N into a
#   temporary copy of the Gazebo world SDF.  Without matching Gazebo RTF, the
#   SITL clock will outrun physics and produce garbage data.
#   Recommended range: 1–5.  Start with --speedup 3 and verify physics stability.
#
# Examples
#   bash sim/run_sitl.sh --alt 100 --leg 200 --radius 40 --legs 4
#   bash sim/run_sitl.sh --speedup 3 --wind-mode step --wind-speed 5 --wind-dir 270
#   bash sim/run_sitl.sh --speedup 3 --wind-mode sine --wind-amp 4 --wind-freq 0.1
#   bash sim/run_sitl.sh --wind-mode sequence --gust-speed 5 --leg-time 30 --turn-time 20

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SPEEDUP=1
ALT=100
LEG=200
RADIUS=40
LEGS=4
AIRSPEED=17.0
RUNWAY=200.0
MAX_TIME=600
LOG_PREFIX=sitl_smc
CONNECT=tcp:127.0.0.1:5762
SITL_PORT=5762

WIND_MODE=""
WIND_SPEED=5.0
WIND_DIR=270.0
WIND_TURB=0.0
WIND_AMP=4.0
WIND_FREQ=0.1
WIND_RAMP=10.0
WIND_DURATION=""   # empty = MAX_TIME + 60
GUST_SPEED=5.0
GUST_DIR=270.0
LEG_TIME=30.0
TURN_TIME=20.0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --speedup)       SPEEDUP=$2;       shift 2 ;;
    --alt)           ALT=$2;           shift 2 ;;
    --leg)           LEG=$2;           shift 2 ;;
    --radius)        RADIUS=$2;        shift 2 ;;
    --legs)          LEGS=$2;          shift 2 ;;
    --airspeed)      AIRSPEED=$2;      shift 2 ;;
    --runway)        RUNWAY=$2;        shift 2 ;;
    --max-time)      MAX_TIME=$2;      shift 2 ;;
    --log)           LOG_PREFIX=$2;    shift 2 ;;
    --connect)       CONNECT=$2;       shift 2 ;;
    --wind-mode)     WIND_MODE=$2;     shift 2 ;;
    --wind-speed)    WIND_SPEED=$2;    shift 2 ;;
    --wind-dir)      WIND_DIR=$2;      shift 2 ;;
    --wind-turb)     WIND_TURB=$2;     shift 2 ;;
    --wind-amp)      WIND_AMP=$2;      shift 2 ;;
    --wind-freq)     WIND_FREQ=$2;     shift 2 ;;
    --wind-ramp)     WIND_RAMP=$2;     shift 2 ;;
    --wind-duration) WIND_DURATION=$2; shift 2 ;;
    --gust-speed)    GUST_SPEED=$2;    shift 2 ;;
    --gust-dir)      GUST_DIR=$2;      shift 2 ;;
    --leg-time)      LEG_TIME=$2;      shift 2 ;;
    --turn-time)     TURN_TIME=$2;     shift 2 ;;
    -h|--help)
      sed -n '/^# Usage/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \?//'
      exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

# Default wind duration covers the full mission plus buffer
if [[ -z "$WIND_DURATION" ]]; then
  WIND_DURATION=$(( MAX_TIME + 60 ))
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTROLLER_DIR="$REPO_ROOT/controller"
SIM_DIR="$SCRIPT_DIR"

ARDUPILOT_AUTOTEST="$HOME/ardupilot/Tools/autotest"
PARAM_FILE="$HOME/SITL_Models/Gazebo/config/skywalker_x8.param"
WORLD_ORIG="$HOME/SITL_Models/Gazebo/worlds/skywalker_x8_runway.sdf"

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
for req in gz python3; do
  if ! command -v "$req" &>/dev/null; then
    echo "[ERROR] '$req' not found in PATH"
    exit 1
  fi
done
if [[ ! -d "$ARDUPILOT_AUTOTEST" ]]; then
  echo "[ERROR] ArduPilot autotest directory not found: $ARDUPILOT_AUTOTEST"
  exit 1
fi
if [[ ! -f "$PARAM_FILE" ]]; then
  echo "[ERROR] Param file not found: $PARAM_FILE"
  exit 1
fi
if [[ ! -f "$WORLD_ORIG" ]]; then
  echo "[ERROR] World SDF not found: $WORLD_ORIG"
  exit 1
fi

# ---------------------------------------------------------------------------
# Process tracking + cleanup
# ---------------------------------------------------------------------------
PIDS=()

cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background processes ..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  # Remove temp SDF if we made one
  [[ -n "${TMP_SDF:-}" ]] && rm -f "$TMP_SDF"
  echo "[CLEANUP] Done."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Print run configuration
# ---------------------------------------------------------------------------
echo "======================================================================"
echo " X8 SITL Automation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
echo " Speedup:    ${SPEEDUP}×"
echo " Trajectory: alt=${ALT}m  leg=${LEG}m  radius=${RADIUS}m  legs=${LEGS}"
echo "             airspeed=${AIRSPEED}m/s  runway=${RUNWAY}m"
echo " Mission:    max=${MAX_TIME}s  log='${LOG_PREFIX}'"
if [[ -n "$WIND_MODE" ]]; then
  echo " Wind:       mode=${WIND_MODE}  speed=${WIND_SPEED}  dir=${WIND_DIR}°  turb=${WIND_TURB}"
  [[ "$WIND_MODE" == "sine" ]]     && echo "             amp=${WIND_AMP}  freq=${WIND_FREQ}Hz"
  [[ "$WIND_MODE" == "ramp" ]]     && echo "             ramp-time=${WIND_RAMP}s"
  [[ "$WIND_MODE" == "sequence" ]] && echo "             gust=${GUST_SPEED}m/s@${GUST_DIR}°  leg=${LEG_TIME}s  turn=${TURN_TIME}s"
  echo "             duration=${WIND_DURATION}s"
else
  echo " Wind:       none"
fi
echo "======================================================================"

# ---------------------------------------------------------------------------
# Prepare world SDF (inject real_time_factor when speedup > 1)
# ---------------------------------------------------------------------------
TMP_SDF=""
WORLD_SDF="$WORLD_ORIG"

if (( $(echo "$SPEEDUP > 1" | bc -l) )); then
  TMP_SDF="$(mktemp /tmp/x8_world_XXXX.sdf)"
  echo "[SIM] Speedup=${SPEEDUP}: injecting <real_time_factor> into temp SDF ..."
  sed "s|</world>|  <physics name=\"default\" type=\"dart\">\n    <max_step_size>0.001</max_step_size>\n    <real_time_factor>${SPEEDUP}</real_time_factor>\n  </physics>\n</world>|" \
    "$WORLD_ORIG" > "$TMP_SDF"
  WORLD_SDF="$TMP_SDF"
fi

# ---------------------------------------------------------------------------
# 1. Launch Gazebo (headless, server-only)
# ---------------------------------------------------------------------------
echo "[SIM] Starting Gazebo (headless) ..."
gz sim -s -r "$WORLD_SDF" > /tmp/gazebo.log 2>&1 &
PIDS+=($!)
echo "[SIM] Gazebo PID=${PIDS[-1]}"
sleep 6   # wait for physics to initialise

# ---------------------------------------------------------------------------
# 2. Launch ArduPlane SITL (arduplane + mavproxy directly — avoids
#    run_in_terminal_window.sh which requires a display)
# ---------------------------------------------------------------------------
ARDUPLANE="$HOME/ardupilot/build/sitl/bin/arduplane"
if [[ ! -f "$ARDUPLANE" ]]; then
  echo "[ERROR] arduplane binary not found: $ARDUPLANE"
  echo "        Build with:  cd ~/ardupilot && ./waf plane"
  exit 1
fi

echo "[SIM] Starting arduplane (speedup=${SPEEDUP}×) ..."
(cd "$ARDUPILOT_AUTOTEST" && \
  "$ARDUPLANE" \
    --model JSON \
    --speedup "$SPEEDUP" \
    --slave 0 \
    --defaults "models/plane.parm,$PARAM_FILE" \
    --sim-address=127.0.0.1 \
    -I0 > /tmp/arduplane.log 2>&1) &
PIDS+=($!)
echo "[SIM] arduplane PID=${PIDS[-1]}  (log: /tmp/arduplane.log)"

sleep 3   # wait for arduplane to open TCP 5760

echo "[SIM] Starting MAVProxy ..."
mavproxy.py \
  --non-interactive \
  --master  tcp:127.0.0.1:5760 \
  --out     udp:127.0.0.1:14550 \
  > /tmp/mavproxy.log 2>&1 &
PIDS+=($!)
echo "[SIM] MAVProxy PID=${PIDS[-1]}  (log: /tmp/mavproxy.log)"

# ---------------------------------------------------------------------------
# 3. Wait for MAVLink port to open
# ---------------------------------------------------------------------------
echo "[SIM] Waiting for MAVLink on port ${SITL_PORT} ..."
python3 - <<PYEOF
import socket, sys, time
for i in range(60):
    try:
        s = socket.create_connection(('127.0.0.1', ${SITL_PORT}), timeout=1.0)
        s.close()
        print(f'[SIM] MAVLink port open after {i*2} s')
        sys.exit(0)
    except OSError:
        time.sleep(2.0)
print('[ERROR] MAVLink port not available after 120 s')
sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# 4. Constant wind (step) — set before takeoff so it's active from ground up
# ---------------------------------------------------------------------------
if [[ "$WIND_MODE" == "step" ]]; then
  echo "[WIND] Setting constant wind: ${WIND_SPEED} m/s from ${WIND_DIR}° (turb=${WIND_TURB}) ..."
  python3 "$SIM_DIR/inject_wind.py" \
    --mode step \
    --speed "$WIND_SPEED" \
    --dir   "$WIND_DIR" \
    --turb  "$WIND_TURB" \
    --duration "$WIND_DURATION" \
    --connect "$CONNECT" &
  PIDS+=($!)
  sleep 2   # give inject_wind time to set params before arming
fi

# ---------------------------------------------------------------------------
# 5. Arm + takeoff + cruise to target altitude/airspeed heading north
# ---------------------------------------------------------------------------
echo "[STARTUP] Arming and climbing to ${ALT} m AGL at ${AIRSPEED} m/s ..."
PYTHONPATH="$CONTROLLER_DIR" python3 "$SIM_DIR/x8_startup.py" \
  --alt      "$ALT" \
  --airspeed "$AIRSPEED" \
  --connect  "$CONNECT"

# ---------------------------------------------------------------------------
# 6. Dynamic wind profiles — start after cruise reached
# ---------------------------------------------------------------------------
if [[ "$WIND_MODE" == "sine" ]]; then
  echo "[WIND] Starting sinusoidal wind (amp=${WIND_AMP} m/s, freq=${WIND_FREQ} Hz, dir=${WIND_DIR}°) ..."
  python3 "$SIM_DIR/inject_wind.py" \
    --mode     sine \
    --amp      "$WIND_AMP" \
    --freq     "$WIND_FREQ" \
    --dir      "$WIND_DIR" \
    --duration "$WIND_DURATION" \
    --connect  "$CONNECT" &
  PIDS+=($!)

elif [[ "$WIND_MODE" == "ramp" ]]; then
  echo "[WIND] Starting ramp wind (target=${WIND_SPEED} m/s, ramp=${WIND_RAMP}s, dir=${WIND_DIR}°) ..."
  python3 "$SIM_DIR/inject_wind.py" \
    --mode      ramp \
    --speed     "$WIND_SPEED" \
    --ramp-time "$WIND_RAMP" \
    --dir       "$WIND_DIR" \
    --duration  "$WIND_DURATION" \
    --connect   "$CONNECT" &
  PIDS+=($!)

elif [[ "$WIND_MODE" == "sequence" ]]; then
  echo "[WIND] Starting wind sequence (gust=${GUST_SPEED} m/s@${GUST_DIR}°, leg=${LEG_TIME}s, turn=${TURN_TIME}s) ..."
  python3 "$SIM_DIR/inject_wind.py" \
    --mode       sequence \
    --gust-speed "$GUST_SPEED" \
    --gust-dir   "$GUST_DIR" \
    --leg-time   "$LEG_TIME" \
    --turn-time  "$TURN_TIME" \
    --connect    "$CONNECT" &
  PIDS+=($!)
fi

# ---------------------------------------------------------------------------
# 7. Run lawnmower SMC mission
# ---------------------------------------------------------------------------
echo ""
echo "[MISSION] Starting lawnmower SMC mission ..."
echo "[MISSION] Log prefix: ${LOG_PREFIX}"
echo ""

PYTHONPATH="$CONTROLLER_DIR" python3 "$CONTROLLER_DIR/x8_run_smc.py" \
  --alt      "$ALT" \
  --leg      "$LEG" \
  --radius   "$RADIUS" \
  --legs     "$LEGS" \
  --airspeed "$AIRSPEED" \
  --runway   "$RUNWAY" \
  --max-time "$MAX_TIME" \
  --log      "$LOG_PREFIX" \
  --connect  "$CONNECT"

echo ""
echo "[DONE] Mission complete.  NPZ log written to: ${LOG_PREFIX}_*.npz"
echo "[DONE] Plot with:  python3 controller/x8_plot_smc.py ${LOG_PREFIX}_*.npz"
