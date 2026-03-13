#!/usr/bin/env python3
"""
experiment_2c_temperature_sweep.py — Temperature Sweep Prediction Test (CL SDK)
================================================================================

Phase 2, Experiment 2c of the Biological Computing research programme.

Tests the falsifiable prediction: varying dephasing strength (effective
temperature) should produce a non-monotonic MC curve if ENAQT is operative.

Two conditions are compared:
  1. ENAQT gating:    p_rel = P₄(γ)  — stochastic Bernoulli dropout on stim
  2. Classical ctrl:   stim amplitude scaled proportionally — no dropout

Prediction (for real BNN with synaptic dynamics):
  - ENAQT: MC peaks at p_rel ≈ 0.418 (ENAQT optimum), then declines
  - Classical: MC increases monotonically with stronger driving

On the Poisson simulator: both should be flat at noise floor (null model).

Outputs:
    experiment_2c_results.png    (6-panel dashboard)
    experiment_2c_metrics.json   (full numerical results)
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

TICKS_PER_SECOND = 1000
DURATION_SEC = 60
WARMUP_SEC = 5
N_TICKS = TICKS_PER_SECOND * DURATION_SEC
N_WARMUP = TICKS_PER_SECOND * WARMUP_SEC

N_INPUT_CHANNELS = 8
INPUT_CHANNELS = list(range(N_INPUT_CHANNELS))
N_CHANNELS = 64
MAX_DELAY = 30
RIDGE_ALPHA = 0.01
INPUT_PROB = 0.5
SEED = 42

# p_rel sweep: fine-grained across the biologically relevant range
P_REL_SWEEP = np.concatenate([
    np.linspace(0.01, 0.10, 5),    # very low release
    np.linspace(0.15, 0.50, 8),    # body temp → ENAQT peak
    np.linspace(0.55, 1.00, 5),    # above ENAQT peak
])
N_SWEEP = len(P_REL_SWEEP)

# Corresponding dephasing rates for ENAQT curve overlay
GAMMA_SWEEP = np.logspace(np.log10(0.01), np.log10(5000), 60)


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
# BNN RUN — ENAQT GATING (Bernoulli dropout)
# ══════════════════════════════════════════════════

def run_enaqt_gated(p_rel, seed=SEED):
    """ENAQT condition: stim transmitted with probability p_rel."""
    rng = np.random.default_rng(seed)
    input_signal = rng.random(N_TICKS)
    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)
    total_spikes = 0
    stim_count = 0
    channel_set = ChannelSet(*INPUT_CHANNELS)
    stim = StimDesign(160, -1.0, 160, 1.0)  # fixed amplitude

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            if input_signal[tick_idx] > (1.0 - INPUT_PROB):
                if rng.random() < p_rel:
                    neurons.stim(channel_set, stim)
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
    return mc_total, total_spikes, stim_count


# ══════════════════════════════════════════════════
# BNN RUN — CLASSICAL CONTROL (amplitude scaling)
# ══════════════════════════════════════════════════

def run_classical_control(amplitude_scale, seed=SEED):
    """Classical condition: stim amplitude scaled, always transmitted."""
    rng = np.random.default_rng(seed)
    input_signal = rng.random(N_TICKS)
    spike_matrix = np.zeros((N_TICKS, N_CHANNELS), dtype=np.float32)
    total_spikes = 0
    stim_count = 0
    channel_set = ChannelSet(*INPUT_CHANNELS)
    # Scale amplitude instead of gating
    amp = 1.0 * amplitude_scale
    stim = StimDesign(160, -amp, 160, amp)

    with cl.open() as neurons:
        tick_idx = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND,
                                 stop_after_ticks=N_TICKS):
            if input_signal[tick_idx] > (1.0 - INPUT_PROB):
                neurons.stim(channel_set, stim)
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
    return mc_total, total_spikes, stim_count


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_2c_results.png")
    output_json = os.path.join(script_dir, "experiment_2c_metrics.json")

    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    os.environ["CL_SDK_RANDOM_SEED"] = str(SEED)

    print("=" * 65)
    print("  EXPERIMENT 2c — TEMPERATURE SWEEP PREDICTION TEST (CL SDK)")
    print("  Phase 2 · Biological Computing Research Programme")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE ENAQT REFERENCE CURVE
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing ENAQT P₄(γ) reference curve")
    print(f"{'─' * 65}")

    p4_ref = []
    for gamma in GAMMA_SWEEP:
        eta, _, _, _ = compute_transport_efficiency(gamma)
        p4_ref.append(eta)
    p4_ref = np.array(p4_ref)
    print(f"  ◎ {len(GAMMA_SWEEP)} points computed, peak P₄ = {p4_ref.max():.4f}")

    # ═══════════════════════════════════════════════
    # STEP 2: ENAQT-GATED SWEEP
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 2: ENAQT-gated sweep (Bernoulli dropout)")
    print(f"{'─' * 65}")

    enaqt_mc = []
    enaqt_spikes = []
    enaqt_stims = []
    for i, p_rel in enumerate(P_REL_SWEEP):
        mc, spk, stm = run_enaqt_gated(p_rel, seed=SEED + i)
        enaqt_mc.append(mc)
        enaqt_spikes.append(spk)
        enaqt_stims.append(stm)
        if i % 3 == 0:
            print(f"    [{i+1:2d}/{N_SWEEP}] p_rel = {p_rel:.3f}  "
                  f"→ MC = {mc:.4f}  ({stm} stims)")

    enaqt_mc = np.array(enaqt_mc)

    # ═══════════════════════════════════════════════
    # STEP 3: CLASSICAL CONTROL SWEEP
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: Classical control sweep (amplitude scaling)")
    print(f"{'─' * 65}")

    classical_mc = []
    classical_spikes = []
    classical_stims = []
    for i, amp in enumerate(P_REL_SWEEP):  # use same values as scale factors
        mc, spk, stm = run_classical_control(amp, seed=SEED + i)
        classical_mc.append(mc)
        classical_spikes.append(spk)
        classical_stims.append(stm)
        if i % 3 == 0:
            print(f"    [{i+1:2d}/{N_SWEEP}] amp = {amp:.3f}  "
                  f"→ MC = {mc:.4f}  ({stm} stims)")

    classical_mc = np.array(classical_mc)

    # ═══════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════
    enaqt_peak_idx = np.argmax(enaqt_mc)
    enaqt_peak_prel = float(P_REL_SWEEP[enaqt_peak_idx])
    enaqt_peak_mc = float(enaqt_mc[enaqt_peak_idx])

    classical_peak_idx = np.argmax(classical_mc)
    classical_peak_amp = float(P_REL_SWEEP[classical_peak_idx])
    classical_peak_mc = float(classical_mc[classical_peak_idx])

    # Check monotonicity of classical control
    diffs = np.diff(classical_mc)
    n_increasing = np.sum(diffs > 0)
    classical_monotonic = n_increasing > len(diffs) * 0.5

    # Non-monotonicity of ENAQT: peak is in middle, not at edges
    enaqt_nonmonotonic = (0.1 < enaqt_peak_prel < 0.9)

    elapsed = time.time() - t_start

    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")
    print(f"  ENAQT gating:")
    print(f"    ◎ MC peak: p_rel = {enaqt_peak_prel:.3f}  "
          f"(MC = {enaqt_peak_mc:.4f})")
    print(f"    ◎ Non-monotonic: {enaqt_nonmonotonic}")
    print(f"  Classical control:")
    print(f"    ◎ MC peak: amp = {classical_peak_amp:.3f}  "
          f"(MC = {classical_peak_mc:.4f})")
    print(f"    ◎ Monotonic trend: {classical_monotonic}")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # ── Save metrics ──
    metrics = {
        'experiment': '2c_temperature_sweep_prediction',
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
            'random_seed': SEED,
            'n_sweep_points': N_SWEEP,
        },
        'p_rel_values': P_REL_SWEEP.tolist(),
        'enaqt_gating': {
            'mc_values': enaqt_mc.tolist(),
            'spikes': enaqt_spikes,
            'stims': enaqt_stims,
            'peak_p_rel': enaqt_peak_prel,
            'peak_mc': enaqt_peak_mc,
            'non_monotonic': bool(enaqt_nonmonotonic),
        },
        'classical_control': {
            'mc_values': classical_mc.tolist(),
            'spikes': classical_spikes,
            'stims': classical_stims,
            'peak_amp': classical_peak_amp,
            'peak_mc': classical_peak_mc,
            'monotonic_trend': bool(classical_monotonic),
        },
        'prediction_test': {
            'enaqt_non_monotonic': bool(enaqt_nonmonotonic),
            'classical_monotonic': bool(classical_monotonic),
            'prediction_supported': bool(enaqt_nonmonotonic
                                         and classical_monotonic),
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(metrics, p4_ref, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 2c COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(metrics, p4_ref, output_path):
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

    p_vals = np.array(metrics['p_rel_values'])
    enaqt_mc = np.array(metrics['enaqt_gating']['mc_values'])
    classical_mc = np.array(metrics['classical_control']['mc_values'])

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

    # ── Panel A: ENAQT vs Classical MC ──
    ax_cmp = fig.add_subplot(gs[0, 0])
    style_ax(ax_cmp, "◈  MC vs p_rel: ENAQT GATING vs CLASSICAL")
    ax_cmp.plot(p_vals, enaqt_mc, 'o-', color=ACCENT_CYAN, linewidth=1.5,
                markersize=4, alpha=0.85, label='ENAQT (Bernoulli gating)')
    ax_cmp.plot(p_vals, classical_mc, 's-', color=ACCENT_AMBER, linewidth=1.5,
                markersize=4, alpha=0.85, label='Classical (amplitude scaling)')
    ax_cmp.axvline(0.418, color=ACCENT_RED, linestyle='--', alpha=0.5,
                   label='ENAQT optimum (p=0.418)')
    ax_cmp.set_xlabel("p_rel / amplitude scale", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_cmp.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_cmp.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel B: ENAQT MC with peak annotation ──
    ax_enaqt = fig.add_subplot(gs[0, 1])
    style_ax(ax_enaqt, "◈  ENAQT GATING — MC(p_rel)")
    ax_enaqt.plot(p_vals, enaqt_mc, 'o-', color=ACCENT_CYAN,
                  linewidth=1.5, markersize=5, alpha=0.9)
    peak_idx = np.argmax(enaqt_mc)
    ax_enaqt.plot(p_vals[peak_idx], enaqt_mc[peak_idx], 'o',
                  color=ACCENT_RED, markersize=10, zorder=5,
                  label=f'Peak: p={p_vals[peak_idx]:.3f}, '
                        f'MC={enaqt_mc[peak_idx]:.4f}')
    ax_enaqt.axvline(0.418, color=ACCENT_RED, linestyle=':', alpha=0.4)
    ax_enaqt.set_xlabel("p_rel", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_enaqt.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_enaqt.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel C: Classical MC with trend ──
    ax_class = fig.add_subplot(gs[1, 0])
    style_ax(ax_class, "◈  CLASSICAL CONTROL — MC(amplitude)")
    ax_class.plot(p_vals, classical_mc, 's-', color=ACCENT_AMBER,
                  linewidth=1.5, markersize=5, alpha=0.9)
    # Linear fit for trend
    z = np.polyfit(p_vals, classical_mc, 1)
    ax_class.plot(p_vals, np.polyval(z, p_vals), '--',
                  color=ACCENT_AMBER, alpha=0.4,
                  label=f'Trend: slope={z[0]:.4f}')
    ax_class.set_xlabel("Amplitude scale", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_class.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_class.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: ENAQT P₄ reference curve ──
    ax_p4 = fig.add_subplot(gs[1, 1])
    style_ax(ax_p4, "◈  ENAQT P₄(γ) REFERENCE CURVE")
    ax_p4.semilogx(GAMMA_SWEEP, p4_ref, color=ACCENT_TEAL,
                   linewidth=2, alpha=0.9)
    ax_p4.axvline(215.46, color=ACCENT_AMBER, linestyle='--',
                  alpha=0.6, label='Body temp (310 K)')
    ax_p4.axvline(1145.05, color=ACCENT_RED, linestyle='--',
                  alpha=0.6, label='ENAQT peak')
    ax_p4.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_p4.set_ylabel("P₄", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_p4.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Stim count comparison ──
    ax_stim = fig.add_subplot(gs[2, 0])
    style_ax(ax_stim, "◈  STIM DELIVERY: ENAQT vs CLASSICAL")
    ax_stim.plot(p_vals, metrics['enaqt_gating']['stims'], 'o-',
                 color=ACCENT_CYAN, linewidth=1.5, markersize=3,
                 alpha=0.85, label='ENAQT (gated)')
    ax_stim.plot(p_vals, metrics['classical_control']['stims'], 's-',
                 color=ACCENT_AMBER, linewidth=1.5, markersize=3,
                 alpha=0.85, label='Classical (all transmitted)')
    ax_stim.set_xlabel("p_rel / amplitude", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_stim.set_ylabel("Stims delivered", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_stim.legend(fontsize=7, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  PREDICTION TEST — RESULTS")
    ax_sum.axis("off")

    pt = metrics['prediction_test']
    lines = [
        f"PLATFORM:         CL SDK Simulator v0.29.0",
        f"SWEEP POINTS:     {metrics['parameters']['n_sweep_points']}",
        f"DURATION:          {DURATION_SEC} s ({WARMUP_SEC} s warmup)",
        "",
        f"ENAQT GATING:",
        f"  MC peak at p_rel = {metrics['enaqt_gating']['peak_p_rel']:.3f}"
        f"  (MC = {metrics['enaqt_gating']['peak_mc']:.4f})",
        f"  Non-monotonic: {pt['enaqt_non_monotonic']}",
        "",
        f"CLASSICAL CONTROL:",
        f"  MC peak at amp = {metrics['classical_control']['peak_amp']:.3f}"
        f"  (MC = {metrics['classical_control']['peak_mc']:.4f})",
        f"  Monotonic trend: {pt['classical_monotonic']}",
        "",
        f"PREDICTION SUPPORTED: {pt['prediction_supported']}",
        "",
        f"NOTE: On Poisson simulator, both are noise.",
        f"On CL1 hardware, ENAQT should show peak",
        f"while classical shows monotonic increase.",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "EXPERIMENT 2c ── TEMPERATURE SWEEP PREDICTION TEST (CL SDK)",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.97,
             f"ENAQT gating vs classical amplitude scaling · "
             f"{N_SWEEP}-point p_rel sweep",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
