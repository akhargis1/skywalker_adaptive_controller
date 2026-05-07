"""
x8_plot_l1.py — Post-flight visualiser for x8_run_l1.py logs.

Four panels:
    Panel 1  XY ground track   actual path, smooth reference lawnmower, uploaded WPs
    Panel 2  Tracking errors   e_n, e_z (+ e_t informational) vs time
    Panel 3  Speed & altitude  airspeed (m/s) and altitude AGL vs time
    Panel 4  Bank & throttle   roll angle and T_cmd vs time

The reference lawnmower is reconstructed from the trajectory parameters saved in
the NPZ (traj_alt, traj_airspeed, traj_leg, traj_radius, traj_legs, traj_runway).
Older logs that lack these keys fall back to plotting the logged x_r/y_r column
with a warning annotation.

Usage:
    python3 x8_plot_l1.py sitl_l1_<ts>.npz
    python3 x8_plot_l1.py sitl_l1_<ts>.npz --save
    python3 x8_plot_l1.py sitl_l1_<ts>.npz --panel 1
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    sys.exit("Install matplotlib:  pip install matplotlib")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(path: str):
    npz    = np.load(path, allow_pickle=True)
    data   = np.array(npz['data'], dtype=float)
    fields = list(npz['fields'])
    return npz, data, fields


def col(data, fields, name):
    return data[:, fields.index(name)]


# ---------------------------------------------------------------------------
# Reference reconstruction
# ---------------------------------------------------------------------------

def _build_smooth_ref(npz):
    """
    Reconstruct full lawnmower path from saved traj_* params.
    Returns (y_east, x_north) arrays for plotting, or None if params absent.
    """
    required = ('traj_alt', 'traj_airspeed', 'traj_leg', 'traj_radius',
                 'traj_legs', 'traj_runway')
    if not all(k in npz.files for k in required):
        return None

    try:
        from trajectory_generation import LawnmowerTrajectory
    except ImportError:
        return None

    traj = LawnmowerTrajectory(
        altitude      = float(npz['traj_alt']),
        airspeed      = float(npz['traj_airspeed']),
        leg_length    = float(npz['traj_leg']),
        turn_radius   = float(npz['traj_radius']),
        num_legs      = int(npz['traj_legs']),
        runway_length = float(npz['traj_runway']),
    )
    ts  = np.linspace(0, traj.total_time, max(500, int(traj.total_time * 5)))
    pts = [traj.query(t) for t in ts]
    return (np.array([p.y_ref for p in pts]),   # East
            np.array([p.x_ref for p in pts]))   # North


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(data, fields):
    t     = col(data, fields, 't')
    e_n   = col(data, fields, 'e_n')
    e_z   = col(data, fields, 'e_z')
    e_t   = col(data, fields, 'e_t')
    T_cmd = col(data, fields, 'T_cmd')

    print("\n── L1 flight summary ─────────────────────────────────")
    print(f"  Duration          {t[-1]:.1f} s   ({len(t)} ticks)")
    print(f"  Cross-track RMS   {np.sqrt(np.mean(e_n**2)):.2f} m")
    print(f"  Cross-track peak  {np.max(np.abs(e_n)):.2f} m")
    print(f"  Altitude RMS      {np.sqrt(np.mean(e_z**2)):.2f} m")
    print(f"  Altitude peak     {np.max(np.abs(e_z)):.2f} m")
    print(f"  Along-track RMS   {np.sqrt(np.mean(e_t**2)):.2f} m")
    print(f"  Along-track peak  {np.max(np.abs(e_t)):.2f} m")
    print(f"  Throttle mean     {np.mean(T_cmd):.3f}  "
          f"(min {np.min(T_cmd):.2f}  max {np.max(T_cmd):.2f})")
    print("──────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _panel1_ground_track(ax, npz, data, fields):
    x   = col(data, fields, 'x')
    y   = col(data, fields, 'y')

    ax.plot(y, x, color='#378ADD', lw=1.5, label='actual', zorder=3)
    ax.plot(y[0], x[0], 'go', ms=7, label='start', zorder=5)

    smooth = _build_smooth_ref(npz)
    if smooth is not None:
        y_ref, x_ref = smooth
        ax.plot(y_ref, x_ref, color='#D85A30', lw=1.0, ls='--',
                alpha=0.8, label='reference (smooth)', zorder=2)
        if 'waypoints' in npz.files:
            wps = np.array(npz['waypoints'])   # shape (N, 2): [x_north, y_east]
            ax.plot(wps[:, 1], wps[:, 0], 'x', color='#D85A30',
                    ms=7, mew=1.5, label='waypoints', zorder=4)
    else:
        # Old log — fall back to logged nearest-point column
        x_r = col(data, fields, 'x_r')
        y_r = col(data, fields, 'y_r')
        ax.plot(y_r, x_r, color='#D85A30', lw=0.8, ls='--',
                alpha=0.6, label='ref (nearest point — no traj params)', zorder=2)
        ax.text(0.02, 0.98,
                'No traj_* params in log.\nShowing nearest-point reference.',
                transform=ax.transAxes, fontsize=8, va='top',
                color='#888780', style='italic')

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Ground track (NED)', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3)


def _panel2_errors(ax, data, fields):
    t   = col(data, fields, 't')
    e_n = col(data, fields, 'e_n')
    e_z = col(data, fields, 'e_z')
    e_t = col(data, fields, 'e_t')

    ax.plot(t, e_n, color='#378ADD', lw=1.2, label='e_n cross-track (m)')
    ax.plot(t, e_z, color='#D85A30', lw=1.2, label='e_z altitude (m)')
    ax.plot(t, e_t, color='#1D9E75', lw=1.0, ls='--', alpha=0.7,
            label='e_t along-track (m)')
    ax.axhline(0, color='#888780', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Tracking errors', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3)


def _panel3_speed_alt(ax, data, fields):
    t        = col(data, fields, 't')
    airspeed = col(data, fields, 'airspeed')
    z        = col(data, fields, 'z')   # NED z (negative = up)

    ax.plot(t, airspeed, color='#378ADD', lw=1.5, label='airspeed (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Airspeed (m/s)', color='#378ADD')
    ax.tick_params(axis='y', labelcolor='#378ADD')

    ax2 = ax.twinx()
    ax2.plot(t, -z, color='#D85A30', lw=1.2, label='altitude AGL (m)', alpha=0.85)
    ax2.set_ylabel('Altitude AGL (m)', color='#D85A30')
    ax2.tick_params(axis='y', labelcolor='#D85A30')

    ax.set_title('Speed & altitude', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, lw=0.3)


def _panel4_bank_throttle(ax, data, fields):
    t     = col(data, fields, 't')
    phi   = np.degrees(col(data, fields, 'phi'))
    T_cmd = col(data, fields, 'T_cmd')

    ax.plot(t, phi, color='#378ADD', lw=1.5, label='φ bank (deg)')
    ax.axhline(0, color='#888780', lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bank angle (deg)', color='#378ADD')
    ax.tick_params(axis='y', labelcolor='#378ADD')

    ax2 = ax.twinx()
    ax2.plot(t, T_cmd, color='#7F77DD', lw=1.0, alpha=0.8, label='throttle T_cmd')
    ax2.set_ylabel('Throttle (0–1)', color='#7F77DD')
    ax2.tick_params(axis='y', labelcolor='#7F77DD')
    ax2.set_ylim(0, 1.1)

    ax.set_title('Bank angle & throttle', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, lw=0.3)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def plot_all(npz, data, fields, title='', save=False, out_path=''):
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    fig.suptitle(title or 'X8 L1 baseline', fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    _panel1_ground_track(ax1, npz, data, fields)

    ax2 = fig.add_subplot(gs[0, 1])
    _panel2_errors(ax2, data, fields)

    ax3 = fig.add_subplot(gs[1, 0])
    _panel3_speed_alt(ax3, data, fields)

    ax4 = fig.add_subplot(gs[1, 1])
    _panel4_bank_throttle(ax4, data, fields)

    if save:
        p = Path(out_path or 'x8_l1_plot.png')
        fig.savefig(p, dpi=150)
        print(f"[PLOT] Saved → {p}")
    else:
        plt.show()


def plot_single(npz, data, fields, panel_num, title='', save=False, out_path=''):
    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    fig.suptitle(title or f'Panel {panel_num}', fontsize=12)
    ax = fig.add_subplot(1, 1, 1)

    if panel_num == 1:
        _panel1_ground_track(ax, npz, data, fields)
    elif panel_num == 2:
        _panel2_errors(ax, data, fields)
    elif panel_num == 3:
        _panel3_speed_alt(ax, data, fields)
    elif panel_num == 4:
        _panel4_bank_throttle(ax, data, fields)
    else:
        sys.exit(f"[ERROR] Unknown panel {panel_num}. Valid: 1–4.")

    if save:
        p = Path(out_path or f'x8_l1_panel{panel_num}.png')
        fig.savefig(p, dpi=150)
        print(f"[PLOT] Saved → {p}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='X8 L1 baseline log plotter')
    ap.add_argument('log',           help='.npz log from x8_run_l1.py')
    ap.add_argument('--save',        action='store_true', help='Save PNG instead of showing')
    ap.add_argument('--out',         default='', help='Output PNG path')
    ap.add_argument('--panel',       type=int, default=0,
                    help='Show single panel (1=track, 2=errors, 3=speed+alt, 4=bank+throttle)')
    ap.add_argument('--no-summary',  action='store_true')
    args = ap.parse_args()

    npz, data, fields = load(args.log)
    print(f"[PLOT] Loaded {len(data)} rows from {args.log}")
    print(f"[PLOT] NPZ keys: {', '.join(npz.files)}")

    if not args.no_summary:
        print_summary(data, fields)

    title = Path(args.log).stem
    if args.panel:
        plot_single(npz, data, fields, args.panel,
                    title=title, save=args.save, out_path=args.out)
    else:
        plot_all(npz, data, fields,
                 title=title, save=args.save, out_path=args.out)


if __name__ == '__main__':
    main()
