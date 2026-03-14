#!/usr/bin/env python3
"""
experiment_2d_closed_loop_enaqt.py — Closed-Loop Spike-Triggered ENAQT (CL SDK)
================================================================================

Phase 2, Experiment 2d of the Biological Computing research programme.

Creates a true closed-loop ENAQT reservoir: when a spike is detected on any
output channel, the system stimulates input channels with ENAQT-gated
probability p_rel = P₄(γ).  This creates a biologically realistic feedback
loop where neural activity drives its own inputs through quantum-modulated
gating.

Three p_rel conditions are tested + dephasing sweep:
  1. Coherent:    γ = 0.01 cm⁻¹     → p_rel = 0.063
  2. Body temp:   γ = 215 cm⁻¹      → p_rel = 0.180
  3. ENAQT peak:  γ = 1145 cm⁻¹     → p_rel = 0.418
  + 15-point dephasing sweep to map closed-loop MC vs γ

High tick rate (25 kHz) to capture sub-millisecond spike dynamics.

On Poisson simulator: closed-loop dynamics are limited (no synaptic
plasticity, no adaptation), so MC will be in the noise floor.
On CL1 hardware: real recurrent neural dynamics should produce meaningful
feedback effects.

Outputs:
    experiment_2d_results.png    (6-panel dashboard)
    experiment_2d_metrics.json   (full numerical results)
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import Ridge

import cl
from cl import ChannelSet, StimDesign

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_1c_enaqt_ion_channel import compute_transport_efficiency


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

TICKS_PER_SECOND = 25000       # 25 kHz — sub-millisecond resolution
DURATION_SEC = 10              # shorter duration at high tick rate
WARMUP_SEC = 1
N_TICKS = TICKS_PER_SECOND * DURATION_SEC
N_WARMUP = TICKS_PER_SECOND * WARMUP_SEC

N_INPUT_CHANNELS = 8
INPUT_CHANNELS = list(range(N_INPUT_CHANNELS))
N_OUTPUT_CHANNELS = 56         # channels 8-63 are output
OUTPUT_CHANNELS = list(range(N_INPUT_CHANNELS, 64))
N_CHANNELS = 64
MAX_DELAY = 50                 # more delays at higher tick rate
RIDGE_ALPHA = 0.01
SEED = 42

# Input: random binary signal, flipped every ~2ms (50 ticks at 25kHz)
INPUT_FLIP_INTERVAL = 50

# Named conditions (from Experiment 1c/1e)
NAMED_CONDITIONS = {
    'coherent':   {'gamma': 0.01,   'p_rel': 0.063, 'label': 'Coherent (γ=0.01)'},
    'body_temp':  {'gamma': 215.46, 'p_rel': 0.180, 'label': 'Body temp (310 K)'},
    'enaqt_peak': {'gamma': 1145.05,'p_rel': 0.418, 'label': 'ENAQT peak'},
}

# Dephasing sweep for closed-loop MC curve
N_SWEEP = 15
GAMMA_SWEEP = np.logspace(np.log10(0.01), np.log10(5000), N_SWEEP)


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
# CLOSED-LOOP BNN RUN
# ══════════════════════════════════════════════════

def run_closed_loop(p_rel, seed=SEED):
    """
    Closed-loop spike-triggered ENAQT reservoir.

    Flow:
      1. Generate random input signal (block-structured)
      2. At each tick, inject external input stim (with ENAQT gating)
      3. When a spike is detected on any output channel,
         stim input channels with probability p_rel (ENAQT feedback)
      4. Record spike matrix for readout training

    Returns: mc_total, total_spikes, feedback_stims, external_stims
    """
    rng = np.random.default_rng(seed)

    # Block-structured input: changes every INPUT_FLIP_INTERVAL ticks
    n_blocks = N_TICKS // INPUT_FLIP_INTERVAL + 1
    block_values = rng.random(n_blocks)
    input_signal = np.repeat(block_values, INPUT_FLIP_INTERVAL)[:N_TICKS]

    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)
    total_spikes = 0
    feedback_stims = 0
    external_stims = 0

    input_chs = ChannelSet(*INPUT_CHANNELS)
    stim = StimDesign(160, -1.0, 160, 1.0)

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            # ── External input stimulation (ENAQT-gated) ──
            if input_signal[tick_idx] > 0.5:
                if rng.random() < p_rel:
                    neurons.stim(input_chs, stim)
                    external_stims += 1

            # ── Record spikes + closed-loop feedback ──
            for spike in tick.analysis.spikes:
                ch = spike.channel
                if 0 <= ch < N_CHANNELS:
                    spike_matrix[tick_idx, ch] += 1.0
                    total_spikes += 1

                    # Spike-triggered ENAQT-gated feedback
                    if ch in OUTPUT_CHANNELS:
                        if rng.random() < p_rel:
                            neurons.stim(input_chs, stim)
                            feedback_stims += 1

            tick_idx += 1

    states = spike_matrix[N_WARMUP:]
    signal = input_signal[N_WARMUP:]

    # Downsample to ~1000 Hz bins for MC computation (25 bins per sample)
    bin_size = TICKS_PER_SECOND // 1000
    n_bins = len(signal) // bin_size
    states_binned = states[:n_bins * bin_size].reshape(n_bins, bin_size,
                                                       N_CHANNELS).sum(axis=1)
    signal_binned = signal[:n_bins * bin_size].reshape(n_bins,
                                                       bin_size).mean(axis=1)

    mc_total, mc_per_delay = compute_memory_capacity(
        states_binned, signal_binned, max_delay=MAX_DELAY)

    return mc_total, mc_per_delay, total_spikes, feedback_stims, external_stims


# ══════════════════════════════════════════════════
# OPEN-LOOP CONTROL (no spike-triggered feedback)
# ══════════════════════════════════════════════════

def run_open_loop(p_rel, seed=SEED):
    """
    Open-loop baseline: external ENAQT-gated input only, no feedback.
    Same as Experiment 2b but at 25 kHz with block-structured input.
    """
    rng = np.random.default_rng(seed)
    n_blocks = N_TICKS // INPUT_FLIP_INTERVAL + 1
    block_values = rng.random(n_blocks)
    input_signal = np.repeat(block_values, INPUT_FLIP_INTERVAL)[:N_TICKS]

    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)
    total_spikes = 0
    external_stims = 0

    input_chs = ChannelSet(*INPUT_CHANNELS)
    stim = StimDesign(160, -1.0, 160, 1.0)

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            if input_signal[tick_idx] > 0.5:
                if rng.random() < p_rel:
                    neurons.stim(input_chs, stim)
                    external_stims += 1
            for spike in tick.analysis.spikes:
                ch = spike.channel
                if 0 <= ch < N_CHANNELS:
                    spike_matrix[tick_idx, ch] += 1.0
                    total_spikes += 1
            tick_idx += 1

    states = spike_matrix[N_WARMUP:]
    signal = input_signal[N_WARMUP:]

    bin_size = TICKS_PER_SECOND // 1000
    n_bins = len(signal) // bin_size
    states_binned = states[:n_bins * bin_size].reshape(n_bins, bin_size,
                                                       N_CHANNELS).sum(axis=1)
    signal_binned = signal[:n_bins * bin_size].reshape(n_bins,
                                                       bin_size).mean(axis=1)

    mc_total, mc_per_delay = compute_memory_capacity(
        states_binned, signal_binned, max_delay=MAX_DELAY)

    return mc_total, mc_per_delay, total_spikes, external_stims


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_2d_results.png")
    output_json = os.path.join(script_dir, "experiment_2d_metrics.json")

    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    os.environ["CL_SDK_RANDOM_SEED"] = str(SEED)

    print("=" * 65)
    print("  EXPERIMENT 2d — CLOSED-LOOP SPIKE-TRIGGERED ENAQT (CL SDK)")
    print("  Phase 2 · Biological Computing Research Programme")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE ENAQT P₄(γ) REFERENCE CURVE
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing ENAQT P₄(γ) reference curve")
    print(f"{'─' * 65}")

    gamma_fine = np.logspace(np.log10(0.01), np.log10(5000), 60)
    p4_ref = []
    for gamma in gamma_fine:
        eta, _, _, _ = compute_transport_efficiency(gamma)
        p4_ref.append(eta)
    p4_ref = np.array(p4_ref)
    print(f"  ◎ {len(gamma_fine)} points computed, peak P₄ = {p4_ref.max():.4f}")

    # ═══════════════════════════════════════════════
    # STEP 2: NAMED CONDITIONS (closed-loop vs open-loop)
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 2: Named conditions — closed-loop vs open-loop")
    print(f"{'─' * 65}")

    named_results = {}
    for name, cond in NAMED_CONDITIONS.items():
        p_rel = cond['p_rel']

        print(f"    ◈  {cond['label']} (p_rel = {p_rel:.3f})")

        # Closed-loop
        mc_cl, mc_delay_cl, spk_cl, fb_cl, ext_cl = run_closed_loop(
            p_rel, seed=SEED)
        print(f"       CLOSED-LOOP:  MC = {mc_cl:.4f}  "
              f"({spk_cl} spikes, {fb_cl} feedback stims)")

        # Open-loop
        mc_ol, mc_delay_ol, spk_ol, ext_ol = run_open_loop(
            p_rel, seed=SEED)
        print(f"       OPEN-LOOP:    MC = {mc_ol:.4f}  "
              f"({spk_ol} spikes, {ext_ol} external stims)")

        named_results[name] = {
            'gamma': cond['gamma'],
            'p_rel': p_rel,
            'label': cond['label'],
            'closed_loop': {
                'mc': mc_cl, 'mc_per_delay': mc_delay_cl.tolist(),
                'spikes': spk_cl, 'feedback_stims': fb_cl,
                'external_stims': ext_cl,
            },
            'open_loop': {
                'mc': mc_ol, 'mc_per_delay': mc_delay_ol.tolist(),
                'spikes': spk_ol, 'external_stims': ext_ol,
            },
        }

    # ═══════════════════════════════════════════════
    # STEP 3: DEPHASING SWEEP (closed-loop only)
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: Dephasing sweep — closed-loop MC vs γ")
    print(f"{'─' * 65}")

    sweep_mc = []
    sweep_spikes = []
    sweep_feedback = []
    sweep_p_rel = []

    for i, gamma in enumerate(GAMMA_SWEEP):
        eta, _, _, _ = compute_transport_efficiency(gamma)
        p_rel = eta
        sweep_p_rel.append(float(p_rel))

        mc, _, spk, fb, ext = run_closed_loop(p_rel, seed=SEED + i)
        sweep_mc.append(mc)
        sweep_spikes.append(spk)
        sweep_feedback.append(fb)

        print(f"    [{i+1:2d}/{N_SWEEP}] γ = {gamma:8.2f} cm⁻¹  "
              f"p_rel = {p_rel:.4f}  →  MC = {mc:.4f}  "
              f"({fb} feedback stims)")

    sweep_mc = np.array(sweep_mc)

    # ═══════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════
    peak_idx = np.argmax(sweep_mc)
    peak_gamma = float(GAMMA_SWEEP[peak_idx])
    peak_mc = float(sweep_mc[peak_idx])

    # Correlation between log(P₄) and log(MC) on sweep
    sweep_p4 = np.array(sweep_p_rel)
    valid = (sweep_p4 > 0) & (sweep_mc > 0)
    if np.sum(valid) > 3:
        corr_log = np.corrcoef(np.log(sweep_p4[valid]),
                               np.log(sweep_mc[valid]))[0, 1]
    else:
        corr_log = float('nan')

    elapsed = time.time() - t_start

    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")

    for name, res in named_results.items():
        cl_mc = res['closed_loop']['mc']
        ol_mc = res['open_loop']['mc']
        delta = cl_mc - ol_mc
        print(f"  {res['label']:25s}  CL={cl_mc:.4f}  OL={ol_mc:.4f}  "
              f"Δ={delta:+.4f}")

    print(f"\n  Sweep peak: γ = {peak_gamma:.2f} cm⁻¹  (MC = {peak_mc:.4f})")
    print(f"  log-log correlation (P₄ vs MC): r = {corr_log:.4f}")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # ── Save metrics ──
    metrics = {
        'experiment': '2d_closed_loop_enaqt',
        'platform': 'CL SDK Simulator v0.29.0',
        'parameters': {
            'ticks_per_second': TICKS_PER_SECOND,
            'duration_sec': DURATION_SEC,
            'warmup_sec': WARMUP_SEC,
            'n_channels': N_CHANNELS,
            'n_input_channels': N_INPUT_CHANNELS,
            'n_output_channels': N_OUTPUT_CHANNELS,
            'input_flip_interval': INPUT_FLIP_INTERVAL,
            'max_delay': MAX_DELAY,
            'ridge_alpha': RIDGE_ALPHA,
            'random_seed': SEED,
            'n_sweep_points': N_SWEEP,
        },
        'named_conditions': named_results,
        'dephasing_sweep': {
            'gamma_values': GAMMA_SWEEP.tolist(),
            'p_rel_values': sweep_p_rel,
            'mc_values': sweep_mc.tolist(),
            'spikes': sweep_spikes,
            'feedback_stims': sweep_feedback,
            'peak_gamma': peak_gamma,
            'peak_mc': peak_mc,
            'log_log_corr': round(corr_log, 4) if not np.isnan(corr_log)
                            else None,
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(metrics, gamma_fine, p4_ref, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 2d COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(metrics, gamma_fine, p4_ref, output_path):
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
                         fontfamily=MONO, fontweight="bold",
                         loc="left", pad=10)

    named = metrics['named_conditions']
    sweep = metrics['dephasing_sweep']

    # ── Panel A: Closed-loop vs Open-loop bar comparison ──
    ax_bar = fig.add_subplot(gs[0, 0])
    style_ax(ax_bar, "◈  CLOSED-LOOP vs OPEN-LOOP MC")
    names = list(named.keys())
    labels = [named[n]['label'] for n in names]
    cl_vals = [named[n]['closed_loop']['mc'] for n in names]
    ol_vals = [named[n]['open_loop']['mc'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars_cl = ax_bar.bar(x - w/2, cl_vals, w, color=ACCENT_CYAN,
                         alpha=0.8, label='Closed-loop')
    bars_ol = ax_bar.bar(x + w/2, ol_vals, w, color=ACCENT_AMBER,
                         alpha=0.8, label='Open-loop')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=8, color=TEXT_SECONDARY,
                           fontfamily=MONO, rotation=15)
    ax_bar.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_bar.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # Add value labels
    for bar in bars_cl:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.0001,
                    f'{h:.4f}', ha='center', va='bottom',
                    fontsize=7, color=ACCENT_CYAN, fontfamily=MONO)
    for bar in bars_ol:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.0001,
                    f'{h:.4f}', ha='center', va='bottom',
                    fontsize=7, color=ACCENT_AMBER, fontfamily=MONO)

    # ── Panel B: MC per delay for named conditions (closed-loop) ──
    ax_delay = fig.add_subplot(gs[0, 1])
    style_ax(ax_delay, "◈  MC PER DELAY — CLOSED-LOOP")
    colors_named = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_RED]
    for i, name in enumerate(names):
        mc_d = named[name]['closed_loop']['mc_per_delay']
        ax_delay.plot(range(1, len(mc_d) + 1), mc_d,
                      color=colors_named[i], linewidth=1.2,
                      alpha=0.8, label=named[name]['label'])
    ax_delay.set_xlabel("Delay k", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_delay.set_ylabel("r²(k)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_delay.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel C: Dephasing sweep — MC vs γ ──
    ax_sweep = fig.add_subplot(gs[1, 0])
    style_ax(ax_sweep, "◈  CLOSED-LOOP MC vs DEPHASING γ")
    gamma_vals = np.array(sweep['gamma_values'])
    mc_vals = np.array(sweep['mc_values'])
    ax_sweep.semilogx(gamma_vals, mc_vals, 'o-', color=ACCENT_CYAN,
                      linewidth=1.5, markersize=5, alpha=0.9)
    peak_idx = np.argmax(mc_vals)
    ax_sweep.plot(gamma_vals[peak_idx], mc_vals[peak_idx], 'o',
                  color=ACCENT_RED, markersize=10, zorder=5,
                  label=f'Peak: γ={gamma_vals[peak_idx]:.1f}, '
                        f'MC={mc_vals[peak_idx]:.4f}')
    ax_sweep.axvline(215.46, color=ACCENT_AMBER, linestyle='--',
                     alpha=0.5, label='Body temp')
    ax_sweep.axvline(1145.05, color=ACCENT_RED, linestyle='--',
                     alpha=0.5, label='ENAQT peak')
    ax_sweep.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_sweep.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_sweep.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: ENAQT reference curve with sweep overlay ──
    ax_p4 = fig.add_subplot(gs[1, 1])
    style_ax(ax_p4, "◈  P₄(γ) vs CLOSED-LOOP MC(γ)")
    ax_p4.semilogx(gamma_fine, p4_ref, color=ACCENT_TEAL,
                   linewidth=2, alpha=0.9, label='P₄(γ) — ENAQT')
    ax_p4.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_p4.set_ylabel("P₄", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)

    # Overlay MC on secondary axis
    ax_mc2 = ax_p4.twinx()
    ax_mc2.semilogx(gamma_vals, mc_vals, 's-', color=ACCENT_RED,
                    linewidth=1.2, markersize=4, alpha=0.8,
                    label='MC(γ) — closed-loop')
    ax_mc2.set_ylabel("Memory Capacity", color=ACCENT_RED,
                      fontfamily=MONO, fontsize=9)
    ax_mc2.tick_params(colors=ACCENT_RED, labelsize=8)
    ax_mc2.spines["right"].set_color(ACCENT_RED)

    # Combined legend
    lines1, labels1 = ax_p4.get_legend_handles_labels()
    lines2, labels2 = ax_mc2.get_legend_handles_labels()
    ax_p4.legend(lines1 + lines2, labels1 + labels2,
                 fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Feedback stims vs γ ──
    ax_fb = fig.add_subplot(gs[2, 0])
    style_ax(ax_fb, "◈  FEEDBACK STIMS vs DEPHASING γ")
    ax_fb.semilogx(gamma_vals, sweep['feedback_stims'], 'o-',
                   color=ACCENT_CYAN, linewidth=1.5, markersize=4,
                   alpha=0.85, label='Spike-triggered feedback stims')
    ax_fb.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_fb.set_ylabel("Feedback stim count", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_fb.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  CLOSED-LOOP ENAQT — RESULTS")
    ax_sum.axis("off")

    corr_str = (f"{sweep['log_log_corr']:.4f}"
                if sweep['log_log_corr'] is not None else "N/A")

    lines_text = [
        f"PLATFORM:      CL SDK Simulator v0.29.0",
        f"TICK RATE:      {TICKS_PER_SECOND:,} Hz (25 kHz)",
        f"DURATION:       {DURATION_SEC} s ({WARMUP_SEC} s warmup)",
        "",
        f"NAMED CONDITIONS (CL = closed, OL = open):",
    ]
    for name in names:
        cl_mc = named[name]['closed_loop']['mc']
        ol_mc = named[name]['open_loop']['mc']
        lines_text.append(f"  {named[name]['label']:20s} "
                          f"CL={cl_mc:.4f} OL={ol_mc:.4f}")

    lines_text += [
        "",
        f"SWEEP: {N_SWEEP} gamma values",
        f"  Peak: γ = {sweep['peak_gamma']:.1f} cm⁻¹  "
        f"(MC = {sweep['peak_mc']:.4f})",
        f"  log-log corr(P₄, MC) = {corr_str}",
        "",
        f"NOTE: Poisson simulator lacks recurrent",
        f"dynamics. On CL1 hardware, closed-loop",
        f"feedback should enhance MC over open-loop.",
    ]

    text = "\n".join(lines_text)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "EXPERIMENT 2d ── CLOSED-LOOP SPIKE-TRIGGERED ENAQT (CL SDK)",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.97,
             f"Spike-triggered feedback with ENAQT gating · "
             f"25 kHz tick rate · {N_SWEEP}-point dephasing sweep",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
