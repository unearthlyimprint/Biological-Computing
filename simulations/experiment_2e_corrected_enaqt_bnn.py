#!/usr/bin/env python3
"""
experiment_2e_corrected_enaqt_bnn.py — Corrected ENAQT-Gated BNN
=================================================================

Phase 2, Experiment 2e of the Biological Computing research programme.

Re-runs the ENAQT-gated BNN reservoir (Experiment 2b) using corrected
transport physics from the GPU-accelerated Lindblad solver:

  - Donor-bridge-acceptor Hamiltonian: ε = [0, 15J, 15J, 0]
  - Nearest-neighbour hopping: J = 1 (natural units)
  - External trap state (6th state, irreversible sink κ = 0.05J)
  - ENAQT peak: γ_opt ≈ 11J, enhancement = 17.4× over coherent

Previous 2b used the original 4-site symmetric Hamiltonian. This version
uses physically correct energy landscape where coherent tunnelling through
the 15J bridge barrier is slow — only dephasing enables efficient transport.

Three stages:
  1. Four-condition comparison  (Coherent / Low / ENAQT peak / High dephasing)
  2. 20-point dephasing sweep   (γ → P₄ → MC curve)
  3. Co-peak analysis           (does MC peak with P₄?)

Outputs:
    experiment_2e_results.png    (6-panel dashboard)
    experiment_2e_metrics.json   (full numerical results)
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import Ridge

# CL SDK
import cl
from cl import ChannelSet, StimDesign


# ══════════════════════════════════════════════════
# CORRECTED ENAQT CURVE (precomputed from GPU solver)
# ══════════════════════════════════════════════════

# GPU Lindblad solver results: 4-site, ε=[0,15J,15J,0], J=1, κ=0.05J, t=20J
# Precomputed on RTX 5070 Ti using experiment_1c_gpu.py with external trap state
_PRECOMPUTED_GAMMA = np.array([
    0.01, 0.01767, 0.03123, 0.05520, 0.09756, 0.17241, 0.30471,
    0.53852, 0.95173, 1.0, 1.68201, 2.97264, 5.25359, 9.28476,
    11.0, 16.40910, 29.00006, 51.25228, 90.57897, 160.08166,
    200.0, 282.91488, 500.0,
])
_PRECOMPUTED_P4 = np.array([
    0.001827, 0.001755, 0.001640, 0.001472, 0.001259, 0.001043, 0.000929,
    0.001129, 0.002104, 0.002253, 0.004883, 0.011220, 0.021822, 0.031013,
    0.031774, 0.028514, 0.016311, 0.006142, 0.001710, 0.000395,
    0.000216, 0.000082, 0.000016,
])

from scipy.interpolate import interp1d
_ENAQT_INTERP = interp1d(
    np.log10(_PRECOMPUTED_GAMMA), _PRECOMPUTED_P4,
    kind='cubic', fill_value='extrapolate'
)


def compute_enaqt_curve_gpu(gamma_values):
    """
    Look up precomputed ENAQT transport efficiency via interpolation.
    
    Values were computed by the GPU Lindblad solver with:
      4-site, ε=[0, 15J, 15J, 0], J=1, κ=0.05J, t_max=20J
    """
    gamma_arr = np.asarray(gamma_values)
    log_gamma = np.log10(np.clip(gamma_arr, 1e-3, 1e4))
    return np.clip(_ENAQT_INTERP(log_gamma), 0.0, 1.0)



# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

TICKS_PER_SECOND = 1000
DURATION_SEC = 60
WARMUP_SEC = 5
N_TICKS = TICKS_PER_SECOND * DURATION_SEC
N_WARMUP = TICKS_PER_SECOND * WARMUP_SEC

N_INPUT_CHANNELS = 8
INPUT_CHANNELS = list(range(N_INPUT_CHANNELS))
STIM_DESIGN = StimDesign(160, -1.0, 160, 1.0)

N_CHANNELS = 64
MAX_DELAY = 30
RIDGE_ALPHA = 0.01
INPUT_PROB = 0.5
SEED = 42

# Dephasing sweep (in J units)
N_SWEEP = 20
GAMMA_SWEEP = np.logspace(np.log10(0.01), np.log10(500.0), N_SWEEP)

# Key conditions (in J units)
GAMMA_COHERENT = 0.01
GAMMA_LOW = 1.0
GAMMA_PEAK = 11.0    # GPU-derived peak
GAMMA_HIGH = 200.0


# ══════════════════════════════════════════════════
# MEMORY CAPACITY
# ══════════════════════════════════════════════════

def compute_memory_capacity(states, input_signal, max_delay=MAX_DELAY,
                            alpha=RIDGE_ALPHA):
    T = len(input_signal)
    mc_per_delay = np.zeros(max_delay)
    split = int(0.8 * T)

    for k in range(1, max_delay + 1):
        target = input_signal[max_delay - k: T - k]
        X = states[max_delay:]

        X_train, X_test = X[:split - max_delay], X[split - max_delay:]
        y_train, y_test = target[:split - max_delay], target[split - max_delay:]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        corr = np.corrcoef(y_test, y_pred)[0, 1]
        mc_per_delay[k - 1] = corr ** 2 if not np.isnan(corr) else 0.0

    return float(np.sum(mc_per_delay)), mc_per_delay


# ══════════════════════════════════════════════════
# BNN RESERVOIR RUN
# ══════════════════════════════════════════════════

def run_bnn_reservoir(p_rel=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    input_signal = rng.random(N_TICKS)
    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)
    total_spikes = 0
    stim_count = 0

    channel_set = ChannelSet(*INPUT_CHANNELS)

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            if input_signal[tick_idx] > (1.0 - INPUT_PROB):
                if rng.random() < p_rel:
                    neurons.stim(channel_set, STIM_DESIGN)
                    stim_count += 1

            for spike in tick.analysis.spikes:
                ch = spike.channel
                if 0 <= ch < N_CHANNELS:
                    spike_matrix[tick_idx, ch] += 1.0
                    total_spikes += 1

            tick_idx += 1

    states = spike_matrix[N_WARMUP:]
    signal = input_signal[N_WARMUP:]

    mc_total, mc_per_delay = compute_memory_capacity(states, signal)
    duration = (N_TICKS - N_WARMUP) / TICKS_PER_SECOND
    spike_rates = spike_matrix[N_WARMUP:].sum(axis=0) / duration

    return mc_total, mc_per_delay, spike_rates, total_spikes, stim_count


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_2e_results.png")
    output_json = os.path.join(script_dir, "experiment_2e_metrics.json")

    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    os.environ["CL_SDK_RANDOM_SEED"] = str(SEED)

    print("=" * 65)
    print("  EXPERIMENT 2e — CORRECTED ENAQT-GATED BNN RESERVOIR")
    print("  Phase 2 · Biological Computing · Donor-Bridge-Acceptor Model")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE CORRECTED ENAQT CURVE
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing corrected ENAQT curve (GPU Lindblad solver)")
    print(f"{'─' * 65}")
    print("  Model: 4-site, ε=[0, 15J, 15J, 0], κ=0.05J")

    # Compute p_rel for key conditions
    key_gammas = np.array([GAMMA_COHERENT, GAMMA_LOW, GAMMA_PEAK, GAMMA_HIGH])
    key_p4 = compute_enaqt_curve_gpu(key_gammas)
    p_coherent, p_low, p_peak, p_high = key_p4

    print(f"\n  Key P₄ values (corrected model):")
    print(f"    Coherent  (γ={GAMMA_COHERENT:.2f}J): P₄ = {p_coherent:.6f}")
    print(f"    Low       (γ={GAMMA_LOW:.2f}J):  P₄ = {p_low:.6f}")
    print(f"    ENAQT peak (γ={GAMMA_PEAK:.2f}J): P₄ = {p_peak:.6f}")
    print(f"    High      (γ={GAMMA_HIGH:.2f}J): P₄ = {p_high:.6f}")

    # Compute sweep curve
    sweep_p4 = compute_enaqt_curve_gpu(GAMMA_SWEEP)
    print(f"\n  Sweep: {N_SWEEP} points, γ ∈ [{GAMMA_SWEEP[0]:.2f}, {GAMMA_SWEEP[-1]:.0f}]J")
    print(f"  Peak P₄ in sweep: {np.max(sweep_p4):.6f} at γ={GAMMA_SWEEP[np.argmax(sweep_p4)]:.1f}J")

    # Normalize p_rel to [0.01, 1.0] range for BNN stimulation
    max_p4 = max(np.max(sweep_p4), np.max(key_p4))
    if max_p4 > 0:
        norm_factor = 1.0 / max_p4
    else:
        norm_factor = 1.0

    def p4_to_prel(p4):
        return max(0.01, min(1.0, p4 * norm_factor))

    # ═══════════════════════════════════════════════
    # STEP 2: FOUR-CONDITION COMPARISON
    # ═══════════════════════════════════════════════
    conditions = [
        ("Coherent",     GAMMA_COHERENT, p_coherent),
        ("Low dephasing", GAMMA_LOW,     p_low),
        ("ENAQT peak",   GAMMA_PEAK,     p_peak),
        ("High deph",    GAMMA_HIGH,     p_high),
    ]

    results = []
    for label, gamma, p4 in conditions:
        p_rel = p4_to_prel(p4)
        print(f"\n{'─' * 65}")
        print(f"  Condition: {label} (γ={gamma:.2f}J, P₄={p4:.6f}, p_rel={p_rel:.4f})")
        print(f"{'─' * 65}")

        mc, mc_delays, rates, n_spikes, n_stims = run_bnn_reservoir(
            p_rel=p_rel)

        print(f"  ◎ Total spikes: {n_spikes}")
        print(f"  ◎ Stims delivered: {n_stims}")
        print(f"  ◎ Mean spike rate: {rates.mean():.2f} Hz")
        print(f"  ◎ Memory capacity: {mc:.4f}")

        results.append({
            'label': label,
            'gamma_J': gamma,
            'p4': round(float(p4), 6),
            'p_release': round(p_rel, 6),
            'mc_total': round(mc, 4),
            'mc_per_delay': mc_delays.tolist(),
            'total_spikes': n_spikes,
            'stim_count': n_stims,
            'mean_spike_rate_Hz': round(float(rates.mean()), 2),
        })

    # ═══════════════════════════════════════════════
    # STEP 3: DEPHASING SWEEP — γ → P₄ → MC
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: Dephasing sweep — MC(γ) via corrected P₄(γ)")
    print(f"{'─' * 65}")

    mc_sweep = []
    prel_sweep = []

    for i, (gamma, p4) in enumerate(zip(GAMMA_SWEEP, sweep_p4)):
        p_rel = p4_to_prel(p4)
        prel_sweep.append(p_rel)

        mc, _, _, n_spikes, n_stims = run_bnn_reservoir(
            p_rel=p_rel, seed=SEED + i)
        mc_sweep.append(mc)

        if i % 4 == 0:
            print(f"    [{i+1:2d}/{N_SWEEP}] γ = {gamma:8.2f}J  "
                  f"→  P₄ = {p4:.6f}  →  p_rel = {p_rel:.4f}  →  MC = {mc:.4f}")

    mc_sweep = np.array(mc_sweep)
    prel_sweep = np.array(prel_sweep)

    # Co-peak analysis
    mc_peak_idx = np.argmax(mc_sweep)
    mc_peak_gamma = float(GAMMA_SWEEP[mc_peak_idx])
    mc_peak_val = float(mc_sweep[mc_peak_idx])
    p4_peak_idx = np.argmax(sweep_p4)
    p4_peak_gamma = float(GAMMA_SWEEP[p4_peak_idx])

    # Pearson correlation between P₄ and MC curves
    corr_p4_mc = np.corrcoef(sweep_p4, mc_sweep)[0, 1]

    print(f"\n  ◎ MC peak:     γ = {mc_peak_gamma:.1f}J (MC = {mc_peak_val:.4f})")
    print(f"  ◎ P₄ peak:    γ = {p4_peak_gamma:.1f}J (P₄ = {sweep_p4[p4_peak_idx]:.6f})")
    print(f"  ◎ Pearson r:   {corr_p4_mc:.3f}")

    copeak = bool(abs(np.log10(mc_peak_gamma + 1e-10) - np.log10(p4_peak_gamma + 1e-10)) < 0.5)
    print(f"  ◎ Co-peak:     {'YES — within half decade' if copeak else 'NO'}")

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY — CORRECTED ENAQT BNN")
    print(f"{'═' * 65}")
    for r in results:
        print(f"  ◎ {r['label']:15s}  p_rel = {r['p_release']:.4f}  "
              f"→ MC = {r['mc_total']:.4f}")
    print(f"  ◎ MC peak at γ = {mc_peak_gamma:.1f}J")
    print(f"  ◎ Pearson r(P₄, MC) = {corr_p4_mc:.3f}")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # Save metrics
    metrics = {
        'experiment': '2e_corrected_enaqt_bnn',
        'platform': 'CL SDK Simulator + GPU Lindblad',
        'model': {
            'n_sites': 4,
            'site_energies_J': [0.0, 15.0, 15.0, 0.0],
            'coupling_J': 1.0,
            'trap_rate_J': 0.05,
            'description': 'Donor-bridge-acceptor, corrected from experiment_1c_gpu.py',
        },
        'parameters': {
            'ticks_per_second': TICKS_PER_SECOND,
            'duration_sec': DURATION_SEC,
            'warmup_sec': WARMUP_SEC,
            'n_channels': N_CHANNELS,
            'n_input_channels': N_INPUT_CHANNELS,
            'max_delay': MAX_DELAY,
            'ridge_alpha': RIDGE_ALPHA,
            'n_sweep_points': N_SWEEP,
        },
        'four_condition_results': results,
        'dephasing_sweep': {
            'gamma_values_J': GAMMA_SWEEP.tolist(),
            'p4_values': sweep_p4.tolist(),
            'prel_values': prel_sweep.tolist(),
            'mc_values': mc_sweep.tolist(),
            'mc_peak_gamma_J': mc_peak_gamma,
            'mc_peak_value': mc_peak_val,
            'p4_peak_gamma_J': p4_peak_gamma,
            'pearson_r_p4_mc': round(corr_p4_mc, 4),
            'co_peak': copeak,
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # Dashboard
    make_dashboard(results, GAMMA_SWEEP, sweep_p4, mc_sweep,
                   mc_peak_gamma, mc_peak_val, corr_p4_mc, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 2e COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(results, gamma_sweep, p4_sweep, mc_sweep,
                   mc_peak_gamma, mc_peak_val, corr_p4_mc, output_path):
    """6-panel Scientific Data Observatory dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    BG_PRIMARY = "#080b18"
    BG_SECONDARY = "#0c1024"
    ACCENT_CYAN = "#00d2ff"
    ACCENT_TEAL = "#0affef"
    ACCENT_RED = "#f43f5e"
    ACCENT_GREEN = "#10b981"
    ACCENT_AMBER = "#f59e0b"
    TEXT_PRIMARY = (1, 1, 1, 0.95)
    TEXT_SECONDARY = (1, 1, 1, 0.55)
    BORDER = (0, 0.824, 1.0, 0.08)
    MONO = "monospace"

    colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 2, hspace=0.38, wspace=0.28)

    def style_ax(ax, title=""):
        ax.set_facecolor(BG_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title:
            ax.set_title(title, color=ACCENT_CYAN, fontsize=11,
                         fontfamily=MONO, fontweight="bold", loc="left", pad=10)

    # Panel A: MC per delay
    ax_mc = fig.add_subplot(gs[0, 0])
    style_ax(ax_mc, "◈  MEMORY CAPACITY PER DELAY")
    delays = np.arange(1, MAX_DELAY + 1)
    for idx, r in enumerate(results):
        ax_mc.plot(delays, r['mc_per_delay'], color=colors[idx],
                   linewidth=1.5, alpha=0.85,
                   label=f"{r['label']} (MC={r['mc_total']:.3f})")
    ax_mc.set_xlabel("Delay k", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_mc.set_ylabel("r²(k)", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_mc.legend(fontsize=7, facecolor=BG_SECONDARY, edgecolor=BORDER,
                 labelcolor=TEXT_PRIMARY)

    # Panel B: MC bar chart
    ax_bar = fig.add_subplot(gs[0, 1])
    style_ax(ax_bar, "◈  TOTAL MC vs ENAQT CONDITION")
    mc_vals = [r['mc_total'] for r in results]
    bars = ax_bar.bar(range(len(results)), mc_vals,
                      color=colors[:len(results)], alpha=0.85)
    ax_bar.set_xticks(range(len(results)))
    ax_bar.set_xticklabels([r['label'] for r in results],
                           fontfamily=MONO, fontsize=8, color=TEXT_SECONDARY, rotation=15)
    ax_bar.set_ylabel("Memory Capacity", color=TEXT_SECONDARY, fontfamily=MONO)
    for bar, mc in zip(bars, mc_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(mc_vals) * 0.03,
                    f"{mc:.3f}", ha='center', color=TEXT_PRIMARY,
                    fontfamily=MONO, fontsize=9)

    # Panel C: ENAQT P₄ curve (corrected)
    ax_p4 = fig.add_subplot(gs[1, 0])
    style_ax(ax_p4, "◈  CORRECTED ENAQT P₄(γ) — DONOR-BRIDGE-ACCEPTOR")
    ax_p4.semilogx(gamma_sweep, p4_sweep, color=ACCENT_TEAL,
                   linewidth=2, alpha=0.9)
    ax_p4.axvline(GAMMA_PEAK, color=ACCENT_RED, linestyle='--',
                  alpha=0.6, label=f'ENAQT peak (γ={GAMMA_PEAK}J)')
    ax_p4.set_xlabel("Dephasing γ (J units)", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_p4.set_ylabel("P₄ (trap population)", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_p4.legend(fontsize=7, facecolor=BG_SECONDARY, edgecolor=BORDER,
                 labelcolor=TEXT_PRIMARY)

    # Panel D: MC sweep
    ax_mcs = fig.add_subplot(gs[1, 1])
    style_ax(ax_mcs, "◈  MEMORY CAPACITY MC(γ) — CORRECTED MODEL")
    ax_mcs.semilogx(gamma_sweep, mc_sweep, 'o-', color=ACCENT_CYAN,
                    linewidth=1.5, markersize=4, alpha=0.85)
    ax_mcs.axvline(mc_peak_gamma, color=ACCENT_RED, linestyle=':',
                   alpha=0.7, label=f'MC peak γ={mc_peak_gamma:.0f}J')
    ax_mcs.set_xlabel("Dephasing γ (J units)", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_mcs.set_ylabel("Memory Capacity", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_mcs.legend(fontsize=7, facecolor=BG_SECONDARY, edgecolor=BORDER,
                  labelcolor=TEXT_PRIMARY)

    # Panel E: Co-peak overlay
    ax_ov = fig.add_subplot(gs[2, 0])
    style_ax(ax_ov, f"◈  CO-PEAK TEST: r = {corr_p4_mc:.3f}")
    ax_l, ax_r = ax_ov, ax_ov.twinx()
    ax_l.semilogx(gamma_sweep, p4_sweep, color=ACCENT_TEAL, linewidth=2,
                  alpha=0.85, label='P₄(γ)')
    ax_r.semilogx(gamma_sweep, mc_sweep, 'o-', color=ACCENT_CYAN,
                  linewidth=1.5, markersize=3, alpha=0.85, label='MC(γ)')
    ax_l.set_xlabel("Dephasing γ (J)", color=TEXT_SECONDARY, fontfamily=MONO)
    ax_l.set_ylabel("P₄", color=ACCENT_TEAL, fontfamily=MONO)
    ax_r.set_ylabel("MC", color=ACCENT_CYAN, fontfamily=MONO)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(BORDER)
    ax_r.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7,
                facecolor=BG_SECONDARY, edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # Panel F: Summary
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  CORRECTED ENAQT BNN — RESULTS")
    ax_sum.axis("off")
    lines = [
        "PLATFORM:        CL SDK Simulator + GPU Lindblad",
        "MODEL:           4-site, ε=[0, 15J, 15J, 0]",
        "                 J=1, κ=0.05J (external trap)",
        f"CHANNELS:        {N_CHANNELS} total, {N_INPUT_CHANNELS} input",
        f"SWEEP:           {N_SWEEP} points, γ ∈ [0.01, 500]J",
        "",
        "CONDITIONS:",
    ]
    for r in results:
        lines.append(f"  {r['label']:15s} p={r['p_release']:.4f}  "
                     f"MC={r['mc_total']:.4f}")
    lines.extend([
        "",
        f"MC PEAK:          γ = {mc_peak_gamma:.0f}J  (MC = {mc_peak_val:.4f})",
        f"PEARSON r:        {corr_p4_mc:.3f}",
    ])
    ax_sum.text(0.05, 0.95, "\n".join(lines), transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "EXPERIMENT 2e ── CORRECTED ENAQT-GATED BNN RESERVOIR",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99)
    fig.text(0.5, 0.97,
             "Donor-Bridge-Acceptor model · CL SDK Simulator · "
             f"{N_SWEEP}-point sweep",
             ha="center", color=TEXT_SECONDARY, fontsize=9, fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY, bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
