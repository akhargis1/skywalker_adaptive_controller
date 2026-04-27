"""
x8_plot.py — Post-flight log visualiser.

Reads a .npz file written by x8_logger.py and produces a four-panel figure:

    Panel 1  Attitude tracking     φ, θ measured vs reference
    Panel 2  Sliding surface       |s| norm over time
    Panel 3  Parameter adaptation  all 10 theta_hat traces
    Panel 4  Elevon deflections    delta_L, delta_R in degrees

Usage:
    python3 x8_plot.py sitl_run_1234567890.npz
    python3 x8_plot.py sitl_run_1234567890.npz --save        # saves PNG
    python3 x8_plot.py sitl_run_1234567890.npz --panel 1     # single panel
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


PARAM_LABELS = [
    'Ixx', 'Iyy', 'Izz', 'Ixz',
    'Cl_α·qSb', 'Cm_α·qSc',
    'Cl_δ·qSb', 'Cm_δ·qSc',
    'Cn_β·qSb', 'K_drag·qSb',
]


def load(path: str) -> tuple:
    """Returns (data array, field-name list)."""
    npz    = np.load(path, allow_pickle=True)
    data   = npz['data']
    fields = list(npz['fields'])
    return data, fields


def col(data, fields, name):
    return data[:, fields.index(name)]


def plot_all(data, fields, title: str = '', save: bool = False, out_path: str = ''):
    t         = col(data, fields, 't')
    phi       = np.degrees(col(data, fields, 'phi'))
    theta     = np.degrees(col(data, fields, 'theta'))
    phi_d     = np.degrees(col(data, fields, 'phi_d'))
    theta_d   = np.degrees(col(data, fields, 'theta_d'))
    s0        = np.degrees(col(data, fields, 's0'))
    s1        = np.degrees(col(data, fields, 's1'))
    s2        = np.degrees(col(data, fields, 's2'))
    s_norm    = np.sqrt(s0**2 + s1**2 + s2**2)
    delta_L   = np.degrees(col(data, fields, 'delta_L'))
    delta_R   = np.degrees(col(data, fields, 'delta_R'))
    th        = np.column_stack([col(data, fields, f'th{i}') for i in range(10)])

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.suptitle(title or 'X8 CASMC flight log', fontsize=13)
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # --- Panel 1: Attitude tracking ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, phi,   color='#378ADD', lw=1.5,  label='φ measured')
    ax1.plot(t, phi_d, color='#378ADD', lw=1.0, ls='--', label='φ reference')
    ax1.plot(t, theta,   color='#D85A30', lw=1.5,  label='θ measured')
    ax1.plot(t, theta_d, color='#D85A30', lw=1.0, ls='--', label='θ reference')
    ax1.axhline(0, color='#888780', lw=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (deg)')
    ax1.set_title('Attitude tracking', fontsize=11)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, lw=0.3)

    # --- Panel 2: Sliding surface norm ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, s_norm, color='#7F77DD', lw=1.5, label='|s| norm')
    ax2.plot(t, np.abs(s0), color='#378ADD', lw=0.8, alpha=0.6, label='|s_φ|')
    ax2.plot(t, np.abs(s1), color='#D85A30', lw=0.8, alpha=0.6, label='|s_θ|')
    ax2.plot(t, np.abs(s2), color='#1D9E75', lw=0.8, alpha=0.6, label='|s_ψ|')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('deg/s')
    ax2.set_title('Sliding surface', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, lw=0.3)

    # --- Panel 3: Parameter adaptation ---
    ax3 = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.get_cmap('tab10', 10)
    for i in range(10):
        th_norm = th[:, i] / (abs(th[0, i]) + 1e-9)   # normalise to t=0 value
        ax3.plot(t, th_norm, color=cmap(i), lw=1.2, label=PARAM_LABELS[i])
    ax3.axhline(1.0, color='#888780', lw=0.8, ls='--', label='nominal')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('θ̂ / θ̂₀')
    ax3.set_title('Parameter adaptation', fontsize=11)
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.grid(True, lw=0.3)

    # --- Panel 4: Elevon deflections ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, delta_L, color='#378ADD', lw=1.2, label='δ_L')
    ax4.plot(t, delta_R, color='#D85A30', lw=1.2, label='δ_R')
    ax4.axhline( 25, color='#E24B4A', lw=0.7, ls='--', alpha=0.6, label='±limit')
    ax4.axhline(-25, color='#E24B4A', lw=0.7, ls='--', alpha=0.6)
    ax4.axhline(0, color='#888780', lw=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Deflection (deg)')
    ax4.set_title('Elevon deflections', fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, lw=0.3)

    if save:
        p = Path(out_path or 'x8_casmc_plot.png')
        fig.savefig(p, dpi=150)
        print(f"[PLOT] Saved → {p}")
    else:
        plt.show()


def print_summary(data, fields):
    t       = col(data, fields, 't')
    s0      = np.degrees(col(data, fields, 's0'))
    s1      = np.degrees(col(data, fields, 's1'))
    e0      = np.degrees(col(data, fields, 'e0'))
    e1      = np.degrees(col(data, fields, 'e1'))
    delta_L = np.degrees(col(data, fields, 'delta_L'))
    delta_R = np.degrees(col(data, fields, 'delta_R'))

    print("\n── Flight summary ─────────────────────────────")
    print(f"  Duration        {t[-1]:.1f} s   ({len(t)} ticks)")
    print(f"  Roll error RMS  {np.sqrt(np.mean(e0**2)):.2f}°")
    print(f"  Pitch error RMS {np.sqrt(np.mean(e1**2)):.2f}°")
    print(f"  |s_φ| peak      {np.max(np.abs(s0)):.1f} °/s")
    print(f"  |s_θ| peak      {np.max(np.abs(s1)):.1f} °/s")
    print(f"  δ_L saturated   {100*np.mean(np.abs(delta_L) > 24):.1f}% of ticks")
    print(f"  δ_R saturated   {100*np.mean(np.abs(delta_R) > 24):.1f}% of ticks")
    print("───────────────────────────────────────────────\n")


def main():
    ap = argparse.ArgumentParser(description="X8 CASMC log plotter")
    ap.add_argument('log',          help='.npz log file from x8_logger')
    ap.add_argument('--save',       action='store_true', help='Save PNG instead of showing')
    ap.add_argument('--out',        default='', help='Output PNG path')
    ap.add_argument('--no-summary', action='store_true')
    args = ap.parse_args()

    data, fields = load(args.log)
    print(f"[PLOT] Loaded {len(data)} rows from {args.log}")

    if not args.no_summary:
        print_summary(data, fields)

    plot_all(data, fields,
             title=Path(args.log).stem,
             save=args.save,
             out_path=args.out)


if __name__ == "__main__":
    main()
