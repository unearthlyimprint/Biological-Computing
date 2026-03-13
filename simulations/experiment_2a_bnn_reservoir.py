#!/usr/bin/env python3
"""
experiment_2a_bnn_reservoir.py — BNN Reservoir Computing Baseline (CL SDK)
==========================================================================

Phase 2, Experiment 2a of the Biological Computing research programme.

Measures the memory capacity (MC) of a simulated biological neural network
(BNN) on the Cortical Labs CL SDK, establishing a baseline for comparison
with ENAQT-gated experiments (2b).

The CL SDK simulator generates Poisson-distributed spikes across 64 channels,
providing a biologically grounded reservoir. We drive the network with
patterned stimulation encoding a random input signal, record the spike
response, and compute MC via ridge-regression readout — identical to the
methodology used in Experiments 1d and 1e.

Key design decisions:
  - Input encoding:  Binary stim/no-stim per tick based on thresholded
                     random signal (matches Experiment 1e's Bernoulli gating)
  - Reservoir state: Spike count per channel per time bin (biological
                     analogue of ESN node activations)
  - Readout:         Ridge regression (same as Phase 1)
  - MC computation:  Σ r²(k) for k = 1..max_delay

This code runs identically on the CL1 hardware — no modifications needed.

Outputs:
  - experiment_2a_results.png   (dashboard)
  - experiment_2a_metrics.json  (MC values and parameters)

Usage:
    python experiment_2a_bnn_reservoir.py
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
from cl import ChannelSet, StimDesign, BurstDesign


# ══════════════════════════════════════════════════
# EXPERIMENT PARAMETERS
# ══════════════════════════════════════════════════

# Simulation timing
TICKS_PER_SECOND = 1000       # 1 kHz sampling (1 ms bins)
DURATION_SEC = 60             # total experiment duration
WARMUP_SEC = 5                # discard initial transient
N_TICKS = TICKS_PER_SECOND * DURATION_SEC
N_WARMUP = TICKS_PER_SECOND * WARMUP_SEC

# Stimulation parameters
N_INPUT_CHANNELS = 8          # channels used for input stimulation
INPUT_CHANNELS = list(range(N_INPUT_CHANNELS))
STIM_DESIGN = StimDesign(160, -1.0, 160, 1.0)  # biphasic, 160µs, 1µA

# Reservoir readout
N_CHANNELS = 64               # total channels on CL1/simulator
BIN_SIZE_TICKS = 1            # spike count bin = 1 tick (1 ms)
MAX_DELAY = 30                # maximum delay for MC computation
RIDGE_ALPHA = 0.01            # regularisation

# Input signal
INPUT_PROB = 0.5              # probability of stim per tick (Bernoulli)
RANDOM_SEED = 42

# Release probability sweep (for comparison with Phase 1)
P_REL_VALUES = [0.063, 0.180, 0.418, 1.0]
P_REL_LABELS = ["Coherent (0.063)", "Body temp (0.180)",
                "ENAQT peak (0.418)", "Full (1.0)"]


# ══════════════════════════════════════════════════
# MEMORY CAPACITY COMPUTATION
# ══════════════════════════════════════════════════

def compute_memory_capacity(states, input_signal, max_delay=MAX_DELAY,
                            alpha=RIDGE_ALPHA):
    """
    Compute memory capacity from reservoir states and input history.

    MC = Σ_{k=1}^{max_delay} r²(k)

    where r²(k) is the squared correlation between the model's
    prediction of input(t-k) and the actual input(t-k).

    Parameters
    ----------
    states : ndarray (T × N_channels)
        Reservoir state matrix (spike counts per channel per bin)
    input_signal : ndarray (T,)
        Input signal
    max_delay : int
        Maximum delay to evaluate
    alpha : float
        Ridge regression regularisation

    Returns
    -------
    mc_total : float
        Total memory capacity
    mc_per_delay : ndarray (max_delay,)
        MC contribution per delay
    """
    T = len(input_signal)
    mc_per_delay = np.zeros(max_delay)

    # Train/test split (80/20)
    split = int(0.8 * T)

    for k in range(1, max_delay + 1):
        # Target: input shifted by k steps
        target = input_signal[max_delay - k: T - k]

        # Align states
        X = states[max_delay:]

        X_train, X_test = X[:split - max_delay], X[split - max_delay:]
        y_train, y_test = target[:split - max_delay], target[split - max_delay:]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Ridge regression
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Squared correlation
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        mc_per_delay[k - 1] = corr ** 2 if not np.isnan(corr) else 0.0

    mc_total = float(np.sum(mc_per_delay))
    return mc_total, mc_per_delay


# ══════════════════════════════════════════════════
# BNN RESERVOIR EXPERIMENT
# ══════════════════════════════════════════════════

def run_bnn_reservoir(p_rel=1.0, seed=RANDOM_SEED):
    """
    Run the BNN reservoir experiment at a given release probability.

    Parameters
    ----------
    p_rel : float
        Synaptic release probability (1.0 = all stims transmitted)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    mc_total : float
    mc_per_delay : ndarray
    spike_rates : ndarray (per channel)
    total_spikes : int
    stim_count : int
    """
    rng = np.random.default_rng(seed)

    # Generate random binary input signal
    input_signal = rng.random(N_TICKS)

    # Storage for spike counts per channel per tick
    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)

    total_spikes = 0
    stim_count = 0

    channel_set = ChannelSet(*INPUT_CHANNELS)

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            # ── Stimulation (input encoding) ──
            if input_signal[tick_idx] > (1.0 - INPUT_PROB):
                # ENAQT stochastic gating
                if rng.random() < p_rel:
                    neurons.stim(channel_set, STIM_DESIGN)
                    stim_count += 1

            # ── Record spikes ──
            for spike in tick.analysis.spikes:
                ch = spike.channel
                if 0 <= ch < N_CHANNELS:
                    spike_matrix[tick_idx, ch] += 1.0
                    total_spikes += 1

            tick_idx += 1

    # Discard warmup
    states = spike_matrix[N_WARMUP:]
    signal = input_signal[N_WARMUP:]

    # Compute memory capacity
    mc_total, mc_per_delay = compute_memory_capacity(states, signal)

    # Spike rates per channel
    duration = (N_TICKS - N_WARMUP) / TICKS_PER_SECOND
    spike_rates = spike_matrix[N_WARMUP:].sum(axis=0) / duration

    return mc_total, mc_per_delay, spike_rates, total_spikes, stim_count


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_2a_results.png")
    output_json = os.path.join(script_dir, "experiment_2a_metrics.json")

    # Use accelerated time for faster simulation
    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    os.environ["CL_SDK_RANDOM_SEED"] = str(RANDOM_SEED)

    print("=" * 60)
    print("  EXPERIMENT 2a — BNN RESERVOIR COMPUTING BASELINE")
    print("  Phase 2 · Cortical Labs CL SDK Simulator")
    print("=" * 60)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # RUN: Sweep across release probabilities
    # ═══════════════════════════════════════════════

    results = []

    for p_rel, label in zip(P_REL_VALUES, P_REL_LABELS):
        print(f"\n{'─' * 60}")
        print(f"  p_rel = {p_rel:.3f}  ({label})")
        print(f"{'─' * 60}")

        mc, mc_delays, rates, n_spikes, n_stims = run_bnn_reservoir(
            p_rel=p_rel)

        print(f"  ◎ Total spikes: {n_spikes}")
        print(f"  ◎ Stims delivered: {n_stims}")
        print(f"  ◎ Mean spike rate: {rates.mean():.2f} Hz")
        print(f"  ◎ Memory capacity: {mc:.3f}")

        results.append({
            'p_rel': p_rel,
            'label': label,
            'mc_total': round(mc, 4),
            'mc_per_delay': mc_delays.tolist(),
            'total_spikes': n_spikes,
            'stim_count': n_stims,
            'mean_spike_rate_Hz': round(float(rates.mean()), 2),
            'spike_rates': rates.tolist(),
        })

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 60}")
    for r in results:
        print(f"  ◎ p_rel = {r['p_rel']:.3f} → MC = {r['mc_total']:.3f}"
              f"  ({r['label']})")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 60}")

    # ── Save metrics ──
    metrics = {
        'experiment': '2a_bnn_reservoir_baseline',
        'platform': 'CL SDK Simulator v0.29.0',
        'parameters': {
            'ticks_per_second': TICKS_PER_SECOND,
            'duration_sec': DURATION_SEC,
            'warmup_sec': WARMUP_SEC,
            'n_channels': N_CHANNELS,
            'n_input_channels': N_INPUT_CHANNELS,
            'input_prob': INPUT_PROB,
            'max_delay': MAX_DELAY,
            'ridge_alpha': RIDGE_ALPHA,
            'random_seed': RANDOM_SEED,
        },
        'results': results,
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(results, output_img)

    print(f"\n{'═' * 60}")
    print("  EXPERIMENT 2a COMPLETE")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(results, output_path):
    """4-panel Scientific Data Observatory dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Design tokens
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

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.28)

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

    # ── Panel A: MC per delay for each p_rel ──
    ax_mc = fig.add_subplot(gs[0, 0])
    style_ax(ax_mc, "◈  MEMORY CAPACITY PER DELAY")
    delays = np.arange(1, MAX_DELAY + 1)
    for idx, r in enumerate(results):
        ax_mc.plot(delays, r['mc_per_delay'], color=colors[idx],
                   linewidth=1.5, alpha=0.85,
                   label=f"p={r['p_rel']:.3f} (MC={r['mc_total']:.2f})")
    ax_mc.set_xlabel("Delay k", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylabel("r²(k)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel B: MC vs p_rel (bar chart) ──
    ax_bar = fig.add_subplot(gs[0, 1])
    style_ax(ax_bar, "◈  TOTAL MC vs RELEASE PROBABILITY")
    p_vals = [r['p_rel'] for r in results]
    mc_vals = [r['mc_total'] for r in results]
    bars = ax_bar.bar(range(len(results)), mc_vals,
                      color=colors[:len(results)], alpha=0.85)
    ax_bar.set_xticks(range(len(results)))
    ax_bar.set_xticklabels([f"p={p:.3f}" for p in p_vals],
                           fontfamily=MONO, fontsize=8,
                           color=TEXT_SECONDARY)
    ax_bar.set_ylabel("Memory Capacity (MC)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    for bar, mc in zip(bars, mc_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{mc:.2f}", ha='center', color=TEXT_PRIMARY,
                    fontfamily=MONO, fontsize=9)

    # ── Panel C: Spike rate distribution ──
    ax_rate = fig.add_subplot(gs[1, 0])
    style_ax(ax_rate, "◈  SPIKE RATE DISTRIBUTION (p_rel = 1.0)")
    full_result = results[-1]  # p_rel = 1.0
    channels = np.arange(N_CHANNELS)
    ax_rate.bar(channels, full_result['spike_rates'],
                color=ACCENT_CYAN, alpha=0.7, width=0.8)
    ax_rate.set_xlabel("Channel", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_rate.set_ylabel("Spike rate (Hz)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)

    # ── Panel D: Summary card ──
    ax_sum = fig.add_subplot(gs[1, 1])
    style_ax(ax_sum, "◈  BNN RESERVOIR — PARAMETER SUMMARY")
    ax_sum.axis("off")

    lines = [
        f"PLATFORM:       CL SDK Simulator",
        f"CHANNELS:       {N_CHANNELS} total, {N_INPUT_CHANNELS} input",
        f"TICK RATE:       {TICKS_PER_SECOND} Hz",
        f"DURATION:        {DURATION_SEC} s ({WARMUP_SEC} s warmup)",
        f"INPUT:           Bernoulli (p = {INPUT_PROB})",
        f"READOUT:         Ridge (α = {RIDGE_ALPHA})",
        f"MAX DELAY:       {MAX_DELAY}",
        "",
    ]
    for r in results:
        lines.append(
            f"p_rel = {r['p_rel']:.3f}:  MC = {r['mc_total']:.3f}  "
            f"({r['total_spikes']} spikes, {r['stim_count']} stims)")

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=10, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 2a ── BNN RESERVOIR COMPUTING BASELINE (CL SDK)",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             f"Cortical Labs CL SDK · {N_CHANNELS} channels · "
             f"Poisson spike simulation",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
