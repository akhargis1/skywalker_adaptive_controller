"""
x8_plot_smc.py — Post-flight visualiser for x8_run_smc.py logs.

Five panels:
    Panel 1  XY ground track   actual vs reference path (NED, East–North axes)
    Panel 2  Tracking errors   e_n, e_z vs time
    Panel 3  Sliding surfaces  s1 (cross-track), s2 (along-track), s3 (altitude)
    Panel 4  Commands          phi_cmd vs phi, theta_cmd vs theta, T_cmd
    Panel 5  Adaptation        T_trim_hat and K_s_hat convergence vs time

Usage:
    python3 x8_plot_smc.py sitl_smc_<ts>.npz
    python3 x8_plot_smc.py sitl_smc_<ts>.npz --save
    python3 x8_plot_smc.py sitl_smc_<ts>.npz --panel 1
    python3 x8_plot_smc.py sitl_smc_<ts>.npz --panel 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    sys.exit("Install matplotlib:  pip install matplotlib")


def load(path: str) -> tuple:
    npz    = np.load(path, allow_pickle=True)
    data   = np.array(npz['data'], dtype=float)
    fields = list(npz['fields'])
    return data, fields


def col(data, fields, name):
    return data[:, fields.index(name)]


def col_safe(data, fields, name, default=np.nan):
    """Return column by name, or an array of `default` if absent (old logs)."""
    if name in fields:
        return data[:, fields.index(name)]
    return np.full(len(data), default)


def print_summary(data, fields):
    t     = col(data, fields, 't')
    e_n   = col(data, fields, 'e_n')
    e_t   = col(data, fields, 'e_t')
    e_z   = col(data, fields, 'e_z')
    s1    = col(data, fields, 's1')
    s2    = col(data, fields, 's2')
    s3    = col(data, fields, 's3')
    T_cmd = col(data, fields, 'T_cmd')

    print("\n── SMC flight summary ──────────────────────────────")
    print(f"  Duration          {t[-1]:.1f} s   ({len(t)} ticks)")
    print(f"  Cross-track RMS   {np.sqrt(np.mean(e_n**2)):.2f} m")
    print(f"  Cross-track peak  {np.max(np.abs(e_n)):.2f} m")
    print(f"  Altitude RMS      {np.sqrt(np.mean(e_z**2)):.2f} m")
    print(f"  Altitude peak     {np.max(np.abs(e_z)):.2f} m")
    print(f"  |s1| peak         {np.max(np.abs(s1)):.3f} m/s")
    print(f"  |s2| peak         {np.max(np.abs(s2)):.3f} m/s")
    print(f"  |s3| peak         {np.max(np.abs(s3)):.3f} m/s")
    print(f"  Throttle mean     {np.mean(T_cmd):.3f}  (min {np.min(T_cmd):.2f}  max {np.max(T_cmd):.2f})")
    if 'T_trim_hat' in fields:
        T_trim_hat = col(data, fields, 'T_trim_hat')
        K_s_hat    = col(data, fields, 'K_s_hat')
        print(f"  T_trim_hat final  {T_trim_hat[-1]:.4f}  (init {T_trim_hat[0]:.4f})")
        print(f"  K_s_hat final     {K_s_hat[-1]:.4f}   (init {K_s_hat[0]:.4f})")
    print("────────────────────────────────────────────────────\n")


def _plot_adaptation(ax, data, fields, t):
    """Render T_trim_hat / K_s_hat convergence onto an existing axes."""
    T_trim_hat = col_safe(data, fields, 'T_trim_hat')
    K_s_hat    = col_safe(data, fields, 'K_s_hat')

    if np.all(np.isnan(T_trim_hat)):
        ax.text(0.5, 0.5, 'No adaptation data in this log\n(pre-adaptive log file)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11,
                color='#888780')
        ax.set_title('Adaptive throttle estimates', fontsize=11)
        ax.grid(True, lw=0.3)
        return

    ax.plot(t, T_trim_hat, color='#378ADD', lw=1.5, label='T̂_trim')
    ax.axhline(T_trim_hat[0], color='#378ADD', lw=0.8, ls='--', alpha=0.5,
               label=f'T̂_trim init ({T_trim_hat[0]:.3f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('T̂_trim (throttle fraction)', color='#378ADD')
    ax.tick_params(axis='y', labelcolor='#378ADD')

    ax2 = ax.twinx()
    ax2.plot(t, K_s_hat, color='#D85A30', lw=1.5, label='K̂_s')
    ax2.axhline(K_s_hat[0], color='#D85A30', lw=0.8, ls='--', alpha=0.5,
                label=f'K̂_s init ({K_s_hat[0]:.3f})')
    ax2.set_ylabel('K̂_s (throttle sensitivity)', color='#D85A30')
    ax2.tick_params(axis='y', labelcolor='#D85A30')

    ax.set_title('Adaptive throttle estimates (T̂_trim, K̂_s)', fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2)
    ax.grid(True, lw=0.3)


def plot_panel(fig, gs, row, col_idx, data, fields, panel_num):
    t       = col(data, fields, 't')

    if panel_num == 1:
        ax = fig.add_subplot(gs[row, col_idx])
        x     = col(data, fields, 'x')
        y     = col(data, fields, 'y')
        x_r   = col(data, fields, 'x_r')
        y_r   = col(data, fields, 'y_r')
        ax.plot(y,   x,   color='#378ADD', lw=1.5, label='actual')
        ax.plot(y_r, x_r, color='#D85A30', lw=1.0, ls='--', alpha=0.7, label='reference')
        ax.plot(y[0], x[0], 'go', ms=6, label='start')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('Ground track (NED)', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3)

    elif panel_num == 2:
        ax = fig.add_subplot(gs[row, col_idx])
        e_n = col(data, fields, 'e_n')
        e_t = col(data, fields, 'e_t')
        e_z = col(data, fields, 'e_z')
        ax.plot(t, e_n, color='#378ADD', lw=1.2, label='e_n cross-track (m)')
        ax.plot(t, e_t, color='#398DCE', lw=1.2, label='e_t along-track (m)')
        ax.plot(t, e_z, color='#D85A30', lw=1.2, label='e_z altitude (m)')
        ax.axhline(0, color='#888780', lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Tracking errors', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3)

    elif panel_num == 3:
        ax = fig.add_subplot(gs[row, col_idx])
        s1 = col(data, fields, 's1')
        s2 = col(data, fields, 's2')
        s3 = col(data, fields, 's3')
        ax.plot(t, s1, color='#378ADD', lw=1.2, label='s1 cross-track (m/s)')
        ax.plot(t, s2, color='#1D9E75', lw=1.2, label='s2 along-track (m/s)')
        ax.plot(t, s3, color='#D85A30', lw=1.2, label='s3 altitude (m/s)')
        ax.axhline(0, color='#888780', lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Surface value (m/s)')
        ax.set_title('Sliding surfaces', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3)

    elif panel_num == 4:
        ax = fig.add_subplot(gs[row, col_idx])
        phi       = np.degrees(col(data, fields, 'phi'))
        theta     = np.degrees(col(data, fields, 'theta'))
        phi_cmd   = np.degrees(col(data, fields, 'phi_cmd'))
        theta_cmd = np.degrees(col(data, fields, 'theta_cmd'))
        T_cmd     = col(data, fields, 'T_cmd')

        ax.plot(t, phi,       color='#378ADD', lw=1.5,  label='φ actual')
        ax.plot(t, phi_cmd,   color='#378ADD', lw=1.0, ls='--', label='φ_cmd')
        ax.plot(t, theta,     color='#D85A30', lw=1.5,  label='θ actual')
        ax.plot(t, theta_cmd, color='#D85A30', lw=1.0, ls='--', label='θ_cmd')
        ax2 = ax.twinx()
        ax2.plot(t, T_cmd, color='#7F77DD', lw=1.0, alpha=0.7, label='T_cmd')
        ax2.set_ylabel('Throttle (0–1)', color='#7F77DD')
        ax2.tick_params(axis='y', labelcolor='#7F77DD')
        ax2.set_ylim(0, 1.1)
        ax.axhline(0, color='#888780', lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title('Commands vs actual', fontsize=11)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2)
        ax.grid(True, lw=0.3)

    elif panel_num == 5:
        ax = fig.add_subplot(gs[row, col_idx])
        _plot_adaptation(ax, data, fields, t)


def plot_all(data, fields, title='', save=False, out_path=''):
    fig = plt.figure(figsize=(14, 13), constrained_layout=True)
    fig.suptitle(title or 'X8 Path-following SMC', fontsize=13)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    positions = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]
    for r, c, p in positions:
        plot_panel(fig, gs, r, c, data, fields, p)

    # Panel 5 spans both columns in the bottom row
    ax5 = fig.add_subplot(gs[2, :])
    _plot_adaptation(ax5, data, fields, col(data, fields, 't'))


    if save:
        p = Path(out_path or 'x8_smc_plot.png')
        fig.savefig(p, dpi=150)
        print(f"[PLOT] Saved → {p}")
    else:
        plt.show()


def plot_single(data, fields, panel_num, title='', save=False, out_path=''):
    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    fig.suptitle(title or f'Panel {panel_num}', fontsize=12)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    plot_panel(fig, gs, 0, 0, data, fields, panel_num)
    if save:
        p = Path(out_path or f'x8_smc_panel{panel_num}.png')
        fig.savefig(p, dpi=150)
        print(f"[PLOT] Saved → {p}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description='X8 path-following SMC log plotter')
    ap.add_argument('log',           help='.npz log from x8_run_smc.py')
    ap.add_argument('--save',        action='store_true', help='Save PNG instead of showing')
    ap.add_argument('--out',         default='', help='Output PNG path')
    ap.add_argument('--panel',       type=int, default=0,
                    help='Show single panel (1=track, 2=errors, 3=surfaces, 4=commands, 5=adaptation)')
    ap.add_argument('--no-summary',  action='store_true')
    args = ap.parse_args()

    data, fields = load(args.log)
    print(f"[PLOT] Loaded {len(data)} rows from {args.log}")

    if not args.no_summary:
        print_summary(data, fields)

    title = Path(args.log).stem
    if args.panel:
        plot_single(data, fields, args.panel, title=title, save=args.save, out_path=args.out)
    else:
        plot_all(data, fields, title=title, save=args.save, out_path=args.out)


if __name__ == '__main__':
    main()
