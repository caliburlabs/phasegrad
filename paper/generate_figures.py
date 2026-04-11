#!/usr/bin/env python3
"""Generate all paper figures for 'The Gradient Was Already There'.

Produces:
  figures/fig1_schematic.pdf   — Hero: architecture + two-phase protocol
  figures/fig2_verification.pdf — Gradient scatter + residual across scales
  figures/fig3_ablation.pdf     — omega vs K ablation + param-matched bar
  figures/fig5_convergence.pdf  — Convergence diagnosis
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Style — clean, serif, publication-ready
# ---------------------------------------------------------------------------

C_INPUT  = '#4A90D9'   # blue
C_HIDDEN = '#8E8E93'   # gray
C_OUTPUT = '#E8A838'   # amber/gold
C_OMEGA  = '#5B8C5A'   # sage green
C_K      = '#C75C5C'   # muted red
C_BOTH   = '#7B68AE'   # purple
C_CLAMP  = '#D64045'   # red for clamping force
C_GRAD   = '#2D6A4F'   # dark green for gradient arrows

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
})

FIGDIR = Path(__file__).parent / 'figures'
FIGDIR.mkdir(exist_ok=True)
EXPDIR = Path(__file__).parent.parent / 'experiments'


# ===================================================================
# Fig 1  —  Architecture + Two-Phase Protocol (hero, full-width)
# ===================================================================

def _draw_osc(ax, x, y, phase, color, r=0.22, lw=1.6):
    """Draw oscillator: circle with phase needle."""
    c = plt.Circle((x, y), r, fc='white', ec=color, lw=lw, zorder=5)
    ax.add_patch(c)
    dx = r * 0.72 * np.cos(phase)
    dy = r * 0.72 * np.sin(phase)
    ax.plot([x, x + dx], [y, y + dy], color='#333333', lw=2.0,
            solid_capstyle='round', zorder=6)

def _draw_edge(ax, x1, y1, x2, y2, alpha=0.35):
    ax.plot([x1, x2], [y1, y2], color='#AAAAAA', lw=0.9, alpha=alpha,
            zorder=2, solid_capstyle='round')


def fig1_schematic():
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(1, 5, width_ratios=[3, 0.5, 3, 0.5, 3],
                           wspace=0.02)

    # --- Node positions (2-input, 5-hidden, 2-output) ---
    inp  = [(0.0, 0.8), (0.0, -0.8)]
    hid  = [(1.3, 1.4), (1.3, 0.55), (1.3, -0.55), (1.3, -1.4), (2.0, 0.0)]
    out  = [(3.2, 0.55), (3.2, -0.55)]
    pos  = inp + hid + out  # 9 nodes

    # Edges
    edges = []
    for i in range(2):
        for h in range(2, 7):
            edges.append((i, h))
    for h in range(2, 7):
        for o in range(7, 9):
            edges.append((h, o))
    for h in range(2, 6):
        edges.append((h, h + 1))

    # Phases
    rng = np.random.default_rng(7)
    ph_free = rng.uniform(-0.6, 0.6, 9)
    ph_free[0] = 0.9; ph_free[1] = -0.5  # inputs fixed

    ph_clamp = ph_free.copy()
    # Output shift (clamped toward target)
    ph_clamp[7] += 0.40; ph_clamp[8] -= 0.30
    # Hidden shift (gradient diffuses back)
    ph_clamp[2] += 0.15; ph_clamp[3] += 0.10
    ph_clamp[4] -= 0.08; ph_clamp[5] -= 0.12; ph_clamp[6] += 0.18

    disp = ph_clamp - ph_free   # displacement = gradient proxy

    def _panel(ax, phases, label, subtitle, clamp=False, grad=False):
        ax.set_xlim(-0.7, 4.0)
        ax.set_ylim(-2.1, 2.3)
        ax.set_aspect('equal'); ax.axis('off')

        for (i, j) in edges:
            _draw_edge(ax, *pos[i], *pos[j])

        for k, (x, y) in enumerate(pos):
            c = C_INPUT if k < 2 else (C_OUTPUT if k >= 7 else C_HIDDEN)
            _draw_osc(ax, x, y, phases[k], c)

        if clamp:
            R = 0.22  # oscillator radius
            for oi in (7, 8):
                ox, oy = pos[oi]
                # dashed target line
                tx = ox + 0.58
                ax.plot([tx, tx], [oy - 0.22, oy + 0.22], color=C_CLAMP,
                        lw=1.6, ls='--', alpha=0.8)
                # Arrow tip touches circle edge, tail at target line
                ax.annotate('', xy=(ox + R, oy), xytext=(ox + 0.55, oy),
                           arrowprops=dict(arrowstyle='->', color=C_CLAMP,
                                           lw=1.8, shrinkA=0, shrinkB=0))
            ax.text(3.88, 0.0, r'$\beta$', color=C_CLAMP, fontsize=13,
                    ha='left', va='center', fontstyle='italic')

        if grad:
            R = 0.22  # oscillator radius
            for k in range(2, 9):
                x, y = pos[k]
                g = disp[k]
                if abs(g) < 0.02:
                    continue
                sign = 1 if g > 0 else -1
                length = abs(g) * 2.5
                # Arrow starts at circle edge, extends outward
                y_start = y + sign * R
                y_end = y_start + sign * length
                ax.annotate('',
                    xy=(x, y_end),
                    xytext=(x, y_start),
                    arrowprops=dict(arrowstyle='->', color=C_GRAD,
                                    lw=1.6, shrinkA=0, shrinkB=0))

        ax.text(-0.55, 2.15, label, fontsize=13, fontweight='bold', va='top')
        ax.text(1.6, -1.95, subtitle, fontsize=8.5, ha='center', va='top',
                color='#555555')

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    _panel(ax_a, ph_free, '(a)', r'Free equilibrium $\boldsymbol{\theta}^*$')
    # Layer labels
    ax_a.text(0.0, 2.15, 'Input', fontsize=7.5, ha='center', color=C_INPUT,
              fontweight='bold')
    ax_a.text(1.65, 2.15, 'Hidden', fontsize=7.5, ha='center', color=C_HIDDEN,
              fontweight='bold')
    ax_a.text(3.2, 2.15, 'Output', fontsize=7.5, ha='center', color=C_OUTPUT,
              fontweight='bold')

    # Arrow a->b
    ax1 = fig.add_subplot(gs[0, 1]); ax1.axis('off')
    ax1.set_xlim(0, 1); ax1.set_ylim(-1, 1)
    ax1.annotate('', xy=(0.85, 0), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.5))
    ax1.text(0.5, 0.28, 'nudge\noutput', fontsize=6.5, ha='center',
            color='#999999', linespacing=1.3)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 2])
    _panel(ax_b, ph_clamp, '(b)',
           r'Nudged equilibrium $\boldsymbol{\theta}^\beta$', clamp=True)

    # Arrow b->c
    ax2 = fig.add_subplot(gs[0, 3]); ax2.axis('off')
    ax2.set_xlim(0, 1); ax2.set_ylim(-1, 1)
    ax2.annotate('', xy=(0.85, 0), xytext=(0.15, 0),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.5))
    ax2.text(0.5, 0.28, 'read\nphases', fontsize=6.5, ha='center',
            color='#999999', linespacing=1.3)

    # Panel (c)
    ax_c = fig.add_subplot(gs[0, 4])
    _panel(ax_c, ph_free, '(c)',
           r'$\Delta\theta/\beta\;\approx\;-\partial L/\partial\omega$',
           grad=True)

    fig.savefig(FIGDIR / 'fig1_schematic.pdf', bbox_inches='tight',
                pad_inches=0.08)
    plt.close(fig)
    print('  fig1_schematic.pdf')


# ===================================================================
# Fig 2  —  Gradient verification scatter + scale residuals
# ===================================================================

def fig2_verification():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from phasegrad.kuramoto import make_random_network
    from phasegrad.gradient import two_phase_gradient, analytical_gradient
    from phasegrad.losses import mse_target

    # (a) Per-node scatter at N=15
    ax = axes[0]
    net = make_random_network(N=15, K_mean=5.0, omega_spread=0.3,
                              connectivity=0.6, seed=42)
    th, _ = net.equilibrium()
    tgt = mse_target(net.N, net.output_ids, 0, margin=0.2)
    g_tp, _, _ = two_phase_gradient(net, th, tgt, beta=1e-3)
    g_an = analytical_gradient(net, th, tgt)
    idx = list(range(1, net.N))
    tp, an = g_tp[idx], g_an[idx]

    ax.scatter(an, tp, s=40, c=C_INPUT, edgecolors='white', linewidth=0.5,
              zorder=5, alpha=0.9)
    lim = max(abs(an).max(), abs(tp).max()) * 1.35
    ax.plot([-lim, lim], [-lim, lim], '--', color='#CCCCCC', lw=0.8, zorder=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'Analytical $-\tilde{J}^{-1}\mathbf{e}$')
    ax.set_ylabel(r'Two-phase $-\Delta\theta/\beta$')
    ax.set_aspect('equal')
    cs = np.dot(tp, an) / (np.linalg.norm(tp) * np.linalg.norm(an))
    ax.text(0.95, 0.08, f'cosine = {cs:.6f}', transform=ax.transAxes,
           fontsize=8, ha='right', va='bottom', color='#666666',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DDD', alpha=0.9))
    ax.text(0.05, 0.93, '(a)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')

    # (b) Residual + cosine across sizes
    ax = axes[1]
    sizes = [6, 10, 15, 20, 30, 50, 100, 200]
    residuals, cosines = [], []
    for N in sizes:
        net = make_random_network(N=N, K_mean=5.0, omega_spread=0.3,
                                  connectivity=0.6, seed=42)
        th, res = net.equilibrium()
        residuals.append(res)
        tgt = mse_target(net.N, net.output_ids, 0, margin=0.2)
        g1, _, _ = two_phase_gradient(net, th, tgt, beta=1e-4)
        g2 = analytical_gradient(net, th, tgt)
        ii = list(range(1, net.N))
        cosines.append(np.dot(g1[ii], g2[ii]) /
                       (np.linalg.norm(g1[ii]) * np.linalg.norm(g2[ii])))

    ax.semilogy(sizes, residuals, 'o-', color=C_INPUT, ms=5, label='Eq. residual')
    ax.axhline(np.finfo(float).eps, color='#CCC', lw=0.8, ls=':',
              label=r'Machine $\epsilon$')
    ax.set_xlabel('Network size $N$')
    ax.set_ylabel('Max $|F(\\theta^*)|$')
    ax.set_ylim(1e-17, 1e-10)
    ax.legend(frameon=True, fancybox=False, edgecolor='#DDD', loc='upper right')

    ax2 = ax.twinx()
    ax2.plot(sizes, cosines, 's--', color=C_OMEGA, ms=4, alpha=0.8,
            label='Cosine sim.')
    ax2.set_ylabel('Cosine similarity', color=C_OMEGA)
    ax2.set_ylim(0.9999990, 1.0000005)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.7f}'))
    ax2.tick_params(axis='y', colors=C_OMEGA)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_OMEGA)

    ax.text(0.05, 0.93, '(b)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGDIR / 'fig2_verification.pdf', bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print('  fig2_verification.pdf')


# ===================================================================
# Fig 3  —  Ablation: violin + parameter-matched bars
# ===================================================================

def fig3_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    with open(EXPDIR / 'ablation_100seeds_results.json') as f:
        abl = json.load(f)
    pm_path = EXPDIR / 'param_matched_results.json'
    if pm_path.exists():
        with open(pm_path) as f:
            pm = json.load(f)
    else:
        pm = {}

    # Results from 100-seed param-matched run (Table 3 in paper)
    PM_FALLBACK = {
        'omega_only_conv_mean': 0.960, 'omega_only_conv_std': 0.070,
        'omega_only_conv_n': 47,
        'K_matched_7_conv_mean': 0.833, 'K_matched_7_conv_std': 0.078,
        'K_matched_7_conv_n': 46,
        'K_full_24_conv_mean': 0.830, 'K_full_24_conv_std': 0.076,
        'K_full_24_conv_n': 47,
    }

    # Extract accuracies from result dicts
    def _extract(results_list, key='test_acc'):
        """Extract a field from a list of result dicts, or return as-is if floats."""
        if isinstance(results_list[0], dict):
            return np.array([r[key] for r in results_list])
        return np.array(results_list)

    # Test accuracy for violin plot values
    w = _extract(abl['results']['omega_only'], 'test_acc')
    k = _extract(abl['results']['K_only'], 'test_acc')
    b = _extract(abl['results']['both'], 'test_acc')

    # Train accuracy for convergence criterion (paper: final-epoch train acc > 60%)
    w_train = _extract(abl['results']['omega_only'], 'train_acc')
    k_train = _extract(abl['results']['K_only'], 'train_acc')
    b_train = _extract(abl['results']['both'], 'train_acc')

    # (a) Violin — all 100 seeds
    ax = axes[0]
    data = [w, k, b]
    colors = [C_OMEGA, C_K, C_BOTH]
    xlabels = [r'$\omega$', '$K$', 'Both']

    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=False,
                          showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.25); pc.set_edgecolor(colors[i])

    # Convergence masks based on TRAIN accuracy (paper criterion)
    conv_masks = [w_train > 0.6, k_train > 0.6, b_train > 0.6]

    jrng = np.random.default_rng(42)
    for i, (d, xp) in enumerate(zip(data, [1, 2, 3])):
        jit = jrng.uniform(-0.13, 0.13, len(d))
        conv = conv_masks[i]
        ax.scatter(xp + jit[conv], d[conv], s=8, c=colors[i], alpha=0.55,
                  edgecolors='none', zorder=4)
        ax.scatter(xp + jit[~conv], d[~conv], s=6, c='#CCCCCC', alpha=0.35,
                  edgecolors='none', zorder=3)
        # Converged median line (test acc of training-converged seeds)
        cm = d[conv]
        if len(cm):
            ax.plot([xp - 0.22, xp + 0.22], [np.median(cm)] * 2,
                   color=colors[i], lw=2.2, zorder=7)

    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(xlabels)
    ax.set_ylabel('Test accuracy'); ax.set_ylim(-0.05, 1.12)
    ax.axhline(0.6, color='#DDD', lw=0.8, ls=':')
    ax.text(3.35, 0.6, 'conv.', fontsize=6, color='#AAA', va='center')
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('100-seed ablation', fontsize=9, pad=6)

    # Annotations: convergence rates (by train accuracy)
    for i, xp in enumerate([1, 2, 3]):
        n_c = np.sum(conv_masks[i])
        ax.text(xp, 1.07, f'{n_c}/100', fontsize=6.5, ha='center',
               color=colors[i])

    # (b) Parameter-matched bars
    ax = axes[1]
    bar_labels = [r'$\omega$ (7)', '$K$ (7)', '$K$ (24)']
    bar_colors = [C_OMEGA, C_K, '#D4888B']
    edge_colors = ['#3D6B3C', '#8B3030', '#A06060']

    pm_d = pm.get('param_matched', {})
    bar_keys = ['omega_only', 'K_matched_7', 'K_full_24']
    means, sems, ns = [], [], []

    if bar_keys[0] in pm_d:
        for key in bar_keys:
            test_v = _extract(pm_d[key], 'test_acc')
            train_v = _extract(pm_d[key], 'train_acc')
            conv = train_v > 0.6  # convergence by train accuracy
            cv = test_v[conv]     # report test accuracy of converged seeds
            means.append(cv.mean() if len(cv) else 0)
            sems.append(cv.std() / np.sqrt(len(cv)) if len(cv) > 1 else 0)
            ns.append(len(cv))
    else:
        import warnings
        warnings.warn(
            "param_matched_results.json not found or missing keys — "
            "using hardcoded fallback values. Re-run experiments/"
            "param_matched_ablation.py to regenerate.",
            stacklevel=2,
        )
        fb = PM_FALLBACK
        for prefix in ['omega_only', 'K_matched_7', 'K_full_24']:
            means.append(fb[f'{prefix}_conv_mean'])
            n = fb[f'{prefix}_conv_n']
            sems.append(fb[f'{prefix}_conv_std'] / np.sqrt(n))
            ns.append(n)

    ax.bar(range(3), means, yerr=sems, color=bar_colors, alpha=0.75,
          edgecolor=edge_colors, lw=0.8, capsize=3, error_kw={'lw': 0.8})
    ax.set_xticks(range(3)); ax.set_xticklabels(bar_labels)
    ax.set_ylabel('Converged accuracy'); ax.set_ylim(0.68, 1.04)

    # Significance bracket
    yt = max(means) + max(sems) + 0.02
    ax.plot([0, 0, 1, 1], [yt, yt + 0.012, yt + 0.012, yt],
           color='#555', lw=0.8)
    ax.text(0.5, yt + 0.018, r'$p = 1.8 \times 10^{-12}$', fontsize=7, ha='center',
           color='#555')

    for i in range(3):
        ax.text(i, 0.70, f'n={ns[i]}', fontsize=6.5, ha='center', color='#888')

    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('Parameter-matched (converged)', fontsize=9, pad=6)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGDIR / 'fig3_ablation.pdf', bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print('  fig3_ablation.pdf')


# ===================================================================
# Fig 5  —  Convergence diagnosis
# ===================================================================

def fig5_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    with open(EXPDIR / 'convergence_diagnosis.json') as f:
        diag = json.load(f)

    conv_seeds = [s for s in diag if s['converged']]
    fail_seeds = [s for s in diag if not s['converged']]

    # (a) Training accuracy trajectories — the bifurcation
    ax = axes[0]

    # All seeds as thin background lines
    for s in conv_seeds:
        eps = [r['epoch'] for r in s['records']]
        acc = [r['test_acc'] for r in s['records']]
        ax.plot(eps, acc, color=C_OMEGA, alpha=0.12, lw=0.7, zorder=2)
    for s in fail_seeds:
        eps = [r['epoch'] for r in s['records']]
        acc = [r['test_acc'] for r in s['records']]
        ax.plot(eps, acc, color=C_K, alpha=0.12, lw=0.7, zorder=2)

    # Highlight 3 representative seeds from each group
    for s in conv_seeds[:3]:
        eps = [r['epoch'] for r in s['records']]
        acc = [r['test_acc'] for r in s['records']]
        ax.plot(eps, acc, color=C_OMEGA, alpha=0.8, lw=1.5, zorder=4)
    for s in fail_seeds[:3]:
        eps = [r['epoch'] for r in s['records']]
        acc = [r['test_acc'] for r in s['records']]
        ax.plot(eps, acc, color=C_K, alpha=0.8, lw=1.5, zorder=4)

    ax.axhline(0.6, color='#DDD', lw=0.8, ls=':')
    ax.axhline(0.5, color='#EEE', lw=0.6, ls=':')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0.0, 1.05)

    # Legend
    ax.plot([], [], color=C_OMEGA, lw=1.5,
            label=f'Converged ({len(conv_seeds)} seeds)')
    ax.plot([], [], color=C_K, lw=1.5,
            label=f'Failed ({len(fail_seeds)} seeds)')
    ax.legend(fontsize=7, loc='center right', frameon=True, fancybox=False,
             edgecolor='#DDD')

    ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('Accuracy trajectories bifurcate early', fontsize=8, pad=6)

    # (b) Condition number over training — identical for both groups
    ax = axes[1]

    # All seeds as thin lines
    for s in conv_seeds:
        eps = [r['epoch'] for r in s['records']]
        conds = [r['cond'] for r in s['records']]
        ax.plot(eps, conds, color=C_OMEGA, alpha=0.12, lw=0.7, zorder=2)
    for s in fail_seeds:
        eps = [r['epoch'] for r in s['records']]
        conds = [r['cond'] for r in s['records']]
        ax.plot(eps, conds, color=C_K, alpha=0.12, lw=0.7, zorder=2)

    # Mean lines for each group
    for group, color, label in [(conv_seeds, C_OMEGA, 'Converged'),
                                 (fail_seeds, C_K, 'Failed')]:
        epoch_conds = {}
        for s in group:
            for r in s['records']:
                epoch_conds.setdefault(r['epoch'], []).append(r['cond'])
        eps = sorted(epoch_conds.keys())
        means = [np.mean(epoch_conds[e]) for e in eps]
        ax.plot(eps, means, color=color, lw=2.2, zorder=5, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Jacobian cond$(\tilde{J})$')
    ax.legend(fontsize=7, loc='upper right', frameon=True, fancybox=False,
             edgecolor='#DDD')

    ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('Gradient quality identical for both groups', fontsize=8, pad=6)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGDIR / 'fig5_convergence.pdf', bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print('  fig5_convergence.pdf')


# ===================================================================
# Fig 6  —  Spectral seeding: random vs spectral init
# ===================================================================

def fig6_spectral_seeding():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    ss_path = EXPDIR / 'spectral_seeding_100seed_results.json'
    if not ss_path.exists():
        print('  SKIP fig6_spectral_seeding.pdf (no results file)')
        return

    with open(ss_path) as f:
        ss = json.load(f)

    # Extract accuracies from result dicts
    def _get_accs(strategy_results):
        return np.array([r['final_acc'] for r in strategy_results])

    rand_acc = _get_accs(ss['random'])
    spec_acc = _get_accs(ss['multi_eigen'])

    # (a) Histogram / violin of final accuracy distributions
    ax = axes[0]
    bins = np.linspace(-0.05, 1.05, 25)

    ax.hist(rand_acc, bins=bins, alpha=0.5, color=C_K, label='Random',
            edgecolor='white', linewidth=0.5)
    ax.hist(spec_acc, bins=bins, alpha=0.5, color=C_OMEGA, label='Spectral',
            edgecolor='white', linewidth=0.5)
    ax.axvline(0.6, color='#DDD', lw=0.8, ls=':')
    ax.text(0.62, ax.get_ylim()[1] * 0.9, 'conv.', fontsize=6, color='#AAA')

    rand_conv = np.sum(rand_acc > 0.6)
    spec_conv = np.sum(spec_acc > 0.6)
    ax.set_xlabel('Final test accuracy')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='#DDD',
             loc='upper left')

    ax.text(0.95, 0.95, f'Random: {rand_conv}/100\nSpectral: {spec_conv}/100',
           transform=ax.transAxes, fontsize=7, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DDD', alpha=0.9))

    ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('Accuracy distribution (100 seeds)', fontsize=8, pad=6)

    # (b) Per-seed change: spectral - random
    ax = axes[1]
    delta = spec_acc - rand_acc

    # Color by whether the seed was rescued, unchanged, or lost
    rescued = (rand_acc <= 0.6) & (spec_acc > 0.6)
    lost = (rand_acc > 0.6) & (spec_acc <= 0.6)
    kept = ~rescued & ~lost

    seeds = np.arange(len(delta))

    ax.bar(seeds[kept], delta[kept], color='#CCCCCC', width=1.0, alpha=0.4,
          label=f'Unchanged ({kept.sum()})')
    ax.bar(seeds[rescued], delta[rescued], color=C_OMEGA, width=1.0, alpha=0.7,
          label=f'Rescued ({rescued.sum()})')
    if lost.sum() > 0:
        ax.bar(seeds[lost], delta[lost], color=C_K, width=1.0, alpha=0.7,
              label=f'Lost ({lost.sum()})')

    ax.axhline(0, color='#DDD', lw=0.6)
    ax.set_xlabel('Seed index')
    ax.set_ylabel(r'$\Delta$ accuracy (spectral $-$ random)')
    ax.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor='#DDD',
             loc='upper left')

    ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontsize=12,
           fontweight='bold', va='top')
    ax.set_title('Per-seed accuracy change', fontsize=8, pad=6)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGDIR / 'fig6_spectral_seeding.pdf', bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print('  fig6_spectral_seeding.pdf')


# ===================================================================
if __name__ == '__main__':
    print('Generating figures...')
    fig1_schematic()
    fig2_verification()
    fig3_ablation()
    fig5_convergence()
    fig6_spectral_seeding()
    print('All done.')
