#!/usr/bin/env python3
"""
experiment_2b_enaqt_bnn.py — ENAQT-Gated BNN Reservoir (CL SDK)
================================================================

Phase 2, Experiment 2b of the Biological Computing research programme.

Applies the ENAQT-derived synaptic gating from Experiment 1e to the
biological neural network (BNN) on the Cortical Labs CL SDK — replacing
the abstract ESN reservoir with biologically simulated spiking neurons.

The bridge mechanism is identical to Phase 1:
    γ (dephasing rate) → P₄(γ) (ENAQT transport efficiency)
                       → p_rel (synaptic release probability)

At each tick, the input stim is transmitted with probability p_rel or
suppressed. This creates ENAQT-modulated driving of the BNN reservoir.

Three stages:
  1. Four-condition comparison  (Coherent / Body temp / ENAQT peak / Full)
  2. Dephasing sweep            (γ → P₄ → MC curve)
  3. Co-peak analysis           (does MC peak with P₄?)

This code runs identically on CL1 hardware — no modifications needed.

Outputs:
    experiment_2b_results.png    (6-panel dashboard)
    experiment_2b_metrics.json   (full numerical results)
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

# Import ENAQT efficiency from Experiment 1c
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_1c_enaqt_ion_channel import compute_transport_efficiency


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

# Simulation timing
TICKS_PER_SECOND = 1000       # 1 kHz sampling (1 ms bins)
DURATION_SEC = 60             # total experiment duration
WARMUP_SEC = 5                # discard initial transient
N_TICKS = TICKS_PER_SECOND * DURATION_SEC
N_WARMUP = TICKS_PER_SECOND * WARMUP_SEC

# Stimulation
N_INPUT_CHANNELS = 8
INPUT_CHANNELS = list(range(N_INPUT_CHANNELS))
STIM_DESIGN = StimDesign(160, -1.0, 160, 1.0)  # biphasic, 160µs, 1µA

# Reservoir readout
N_CHANNELS = 64
MAX_DELAY = 30
RIDGE_ALPHA = 0.01
INPUT_PROB = 0.5
SEED = 42

# ENAQT dephasing conditions (cm⁻¹)
GAMMA_COHERENT_CM = 0.01
GAMMA_BODY_CM = 215.46        # 310 K (body temperature)
GAMMA_PEAK_CM = 1145.05       # optimal from Experiment 1c
GAMMA_MAX_CM = 5000.0

# Dephasing sweep for MC(γ) curve
N_SWEEP = 40
GAMMA_MIN_SWEEP = 0.01
GAMMA_MAX_SWEEP = 5000.0


# ══════════════════════════════════════════════════
# ENAQT → P_RELEASE BRIDGE
# ══════════════════════════════════════════════════

def compute_p_release(gamma_cm):
    """
    Compute synaptic release probability from ENAQT transport efficiency.

    Calls the Experiment 1c quantum transport model to get P₄(γ).
    """
    eta, _, _, _ = compute_transport_efficiency(gamma_cm)
    return eta


# ══════════════════════════════════════════════════
# MEMORY CAPACITY
# ══════════════════════════════════════════════════

def compute_memory_capacity(states, input_signal, max_delay=MAX_DELAY,
                            alpha=RIDGE_ALPHA):
    """
    Compute MC = Σ r²(k) for k = 1..max_delay via Ridge regression.
    """
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
    """
    Run the BNN reservoir experiment at a given release probability.

    Returns
    -------
    mc_total, mc_per_delay, spike_rates, total_spikes, stim_count
    """
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
            # Stimulation with ENAQT stochastic gating
            if input_signal[tick_idx] > (1.0 - INPUT_PROB):
                if rng.random() < p_rel:
                    neurons.stim(channel_set, STIM_DESIGN)
                    stim_count += 1

            # Record spikes
            for spike in tick.analysis.spikes:
                ch = spike.channel
                if 0 <= ch < N_CHANNELS:
                    spike_matrix[tick_idx, ch] += 1.0
                    total_spikes += 1

            tick_idx += 1

    # Discard warmup
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
    output_img = os.path.join(script_dir, "experiment_2b_results.png")
    output_json = os.path.join(script_dir, "experiment_2b_metrics.json")

    os.environ["CL_SDK_ACCELERATED_TIME"] = "1"
    os.environ["CL_SDK_RANDOM_SEED"] = str(SEED)

    print("=" * 65)
    print("  EXPERIMENT 2b — ENAQT-GATED BNN RESERVOIR (CL SDK)")
    print("  Phase 2 · Biological Computing Research Programme")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE ENAQT CURVE — γ → P₄
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing ENAQT transport efficiency curve")
    print(f"{'─' * 65}")

    p_coherent = compute_p_release(GAMMA_COHERENT_CM)
    p_body = compute_p_release(GAMMA_BODY_CM)
    p_peak = compute_p_release(GAMMA_PEAK_CM)

    print(f"\n  Key P₄ values:")
    print(f"    Coherent (γ={GAMMA_COHERENT_CM:.2f}):  "
          f"P_release = {p_coherent:.4f}")
    print(f"    Body temp (γ={GAMMA_BODY_CM:.2f}): "
          f"P_release = {p_body:.4f}")
    print(f"    ENAQT peak (γ={GAMMA_PEAK_CM:.2f}): "
          f"P_release = {p_peak:.4f}")

    # ═══════════════════════════════════════════════
    # STEP 2: FOUR-CONDITION COMPARISON
    # ═══════════════════════════════════════════════
    conditions = [
        ("Coherent",    GAMMA_COHERENT_CM, p_coherent),
        ("Body temp",   GAMMA_BODY_CM,     p_body),
        ("ENAQT peak",  GAMMA_PEAK_CM,     p_peak),
        ("Full (P=1)",  None,              1.0),
    ]

    results = []
    for label, gamma, p_rel in conditions:
        print(f"\n{'─' * 65}")
        gamma_str = f"γ={gamma:.1f}" if gamma else "N/A"
        print(f"  Condition: {label} ({gamma_str}, "
              f"P_release={p_rel:.4f})")
        print(f"{'─' * 65}")

        mc, mc_delays, rates, n_spikes, n_stims = run_bnn_reservoir(
            p_rel=p_rel)

        print(f"  ◎ Total spikes: {n_spikes}")
        print(f"  ◎ Stims delivered: {n_stims}")
        print(f"  ◎ Mean spike rate: {rates.mean():.2f} Hz")
        print(f"  ◎ Memory capacity: {mc:.4f}")

        results.append({
            'label': label,
            'gamma_cm': gamma,
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
    print("  Step 3: Dephasing sweep — MC(γ) via P₄(γ)")
    print(f"{'─' * 65}")

    gamma_sweep = np.logspace(np.log10(GAMMA_MIN_SWEEP),
                               np.log10(GAMMA_MAX_SWEEP),
                               N_SWEEP)
    p4_sweep = []
    mc_sweep = []

    for i, gamma in enumerate(gamma_sweep):
        p4 = compute_p_release(gamma)
        p4_sweep.append(p4)

        mc, _, _, _, _ = run_bnn_reservoir(
            p_rel=max(p4, 0.001),  # clamp to avoid zero
            seed=SEED + i)

        mc_sweep.append(mc)

        if i % 5 == 0:
            print(f"    [{i+1:2d}/{N_SWEEP}] γ = {gamma:8.2f} cm⁻¹  "
                  f"→  P₄ = {p4:.4f}  →  MC = {mc:.4f}")

    p4_sweep = np.array(p4_sweep)
    mc_sweep = np.array(mc_sweep)

    # ── Co-peak analysis ──
    mc_peak_idx = np.argmax(mc_sweep)
    mc_peak_gamma = float(gamma_sweep[mc_peak_idx])
    mc_peak_val = float(mc_sweep[mc_peak_idx])
    p4_peak_idx = np.argmax(p4_sweep)
    p4_peak_gamma = float(gamma_sweep[p4_peak_idx])

    print(f"\n  ◎ MC peak:  γ = {mc_peak_gamma:.1f} cm⁻¹  "
          f"(MC = {mc_peak_val:.4f})")
    print(f"  ◎ P₄ peak: γ = {p4_peak_gamma:.1f} cm⁻¹  "
          f"(P₄ = {float(p4_sweep[p4_peak_idx]):.4f})")

    copeak = bool(abs(np.log10(mc_peak_gamma) - np.log10(p4_peak_gamma)) < 0.5)
    print(f"  ◎ Co-peak: {'YES — within half decade' if copeak else 'NO'}")

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")
    for r in results:
        print(f"  ◎ {r['label']:15s}  p_rel = {r['p_release']:.4f}  "
              f"→ MC = {r['mc_total']:.4f}")
    print(f"  ◎ MC peak at γ = {mc_peak_gamma:.1f} cm⁻¹")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # ── Save metrics ──
    metrics = {
        'experiment': '2b_enaqt_gated_bnn',
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
        'enaqt_conditions': {
            'gamma_coherent_cm': GAMMA_COHERENT_CM,
            'p_coherent': round(p_coherent, 6),
            'gamma_body_cm': GAMMA_BODY_CM,
            'p_body': round(p_body, 6),
            'gamma_peak_cm': GAMMA_PEAK_CM,
            'p_peak': round(p_peak, 6),
        },
        'four_condition_results': results,
        'dephasing_sweep': {
            'gamma_values': gamma_sweep.tolist(),
            'p4_values': p4_sweep.tolist(),
            'mc_values': mc_sweep.tolist(),
            'mc_peak_gamma_cm': mc_peak_gamma,
            'mc_peak_value': mc_peak_val,
            'p4_peak_gamma_cm': p4_peak_gamma,
            'co_peak': copeak,
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(results, gamma_sweep, p4_sweep, mc_sweep,
                   mc_peak_gamma, mc_peak_val, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 2b COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(results, gamma_sweep, p4_sweep, mc_sweep,
                   mc_peak_gamma, mc_peak_val, output_path):
    """6-panel Scientific Data Observatory dashboard."""
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
    TEXT_MUTED = (1, 1, 1, 0.30)
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
                         fontfamily=MONO, fontweight="bold",
                         loc="left", pad=10)

    # ── Panel A: MC per delay for each condition ──
    ax_mc = fig.add_subplot(gs[0, 0])
    style_ax(ax_mc, "◈  MEMORY CAPACITY PER DELAY")
    delays = np.arange(1, MAX_DELAY + 1)
    for idx, r in enumerate(results):
        ax_mc.plot(delays, r['mc_per_delay'], color=colors[idx],
                   linewidth=1.5, alpha=0.85,
                   label=f"{r['label']} (MC={r['mc_total']:.3f})")
    ax_mc.set_xlabel("Delay k", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylabel("r²(k)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel B: MC vs condition (bar chart) ──
    ax_bar = fig.add_subplot(gs[0, 1])
    style_ax(ax_bar, "◈  TOTAL MC vs ENAQT CONDITION")
    mc_vals = [r['mc_total'] for r in results]
    bars = ax_bar.bar(range(len(results)), mc_vals,
                      color=colors[:len(results)], alpha=0.85)
    ax_bar.set_xticks(range(len(results)))
    ax_bar.set_xticklabels([r['label'] for r in results],
                           fontfamily=MONO, fontsize=8,
                           color=TEXT_SECONDARY, rotation=15)
    ax_bar.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    for bar, mc in zip(bars, mc_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(mc_vals) * 0.03,
                    f"{mc:.3f}", ha='center', color=TEXT_PRIMARY,
                    fontfamily=MONO, fontsize=9)

    # ── Panel C: ENAQT P₄ curve ──
    ax_p4 = fig.add_subplot(gs[1, 0])
    style_ax(ax_p4, "◈  ENAQT TRANSPORT EFFICIENCY P₄(γ)")
    ax_p4.semilogx(gamma_sweep, p4_sweep, color=ACCENT_TEAL,
                   linewidth=2, alpha=0.9)
    ax_p4.axvline(215.46, color=ACCENT_AMBER, linestyle='--',
                  alpha=0.6, label='Body temp (310 K)')
    ax_p4.axvline(1145.05, color=ACCENT_RED, linestyle='--',
                  alpha=0.6, label='ENAQT peak')
    ax_p4.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_p4.set_ylabel("P₄ (transport efficiency)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_p4.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: MC(γ) sweep ──
    ax_mc_sweep = fig.add_subplot(gs[1, 1])
    style_ax(ax_mc_sweep, "◈  MEMORY CAPACITY MC(γ) — DEPHASING SWEEP")
    ax_mc_sweep.semilogx(gamma_sweep, mc_sweep, 'o-',
                         color=ACCENT_CYAN, linewidth=1.5,
                         markersize=4, alpha=0.85)
    ax_mc_sweep.axvline(mc_peak_gamma, color=ACCENT_RED, linestyle=':',
                        alpha=0.7, label=f'MC peak γ={mc_peak_gamma:.0f}')
    ax_mc_sweep.axvline(215.46, color=ACCENT_AMBER, linestyle='--',
                        alpha=0.5, label='Body temp')
    ax_mc_sweep.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                           fontfamily=MONO, fontsize=9)
    ax_mc_sweep.set_ylabel("Memory Capacity", color=TEXT_SECONDARY,
                           fontfamily=MONO, fontsize=9)
    ax_mc_sweep.legend(fontsize=7, facecolor=BG_SECONDARY,
                       edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: P₄ vs MC overlay (co-peak test) ──
    ax_overlay = fig.add_subplot(gs[2, 0])
    style_ax(ax_overlay, "◈  CO-PEAK TEST: P₄ vs MC")
    ax_l = ax_overlay
    ax_r = ax_overlay.twinx()

    ax_l.semilogx(gamma_sweep, p4_sweep, color=ACCENT_TEAL,
                  linewidth=2, alpha=0.85, label='P₄(γ)')
    ax_r.semilogx(gamma_sweep, mc_sweep, 'o-', color=ACCENT_CYAN,
                  linewidth=1.5, markersize=3, alpha=0.85, label='MC(γ)')

    ax_l.set_xlabel("Dephasing γ (cm⁻¹)", color=TEXT_SECONDARY,
                    fontfamily=MONO, fontsize=9)
    ax_l.set_ylabel("P₄", color=ACCENT_TEAL, fontfamily=MONO, fontsize=9)
    ax_r.set_ylabel("MC", color=ACCENT_CYAN, fontfamily=MONO, fontsize=9)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(BORDER)
    ax_r.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # Combined legend
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(lines_l + lines_r, labels_l + labels_r,
                fontsize=7, facecolor=BG_SECONDARY,
                edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  ENAQT BNN RESERVOIR — RESULTS")
    ax_sum.axis("off")

    lines = [
        f"PLATFORM:        CL SDK Simulator v0.29.0",
        f"CHANNELS:        {N_CHANNELS} total, {N_INPUT_CHANNELS} input",
        f"TICK RATE:        {TICKS_PER_SECOND} Hz",
        f"DURATION:         {DURATION_SEC} s ({WARMUP_SEC} s warmup)",
        f"SWEEP POINTS:    {N_SWEEP}",
        f"READOUT:          Ridge (α = {RIDGE_ALPHA})",
        "",
        f"ENAQT CONDITIONS:",
    ]
    for r in results:
        lines.append(
            f"  {r['label']:15s} p={r['p_release']:.4f}  "
            f"MC={r['mc_total']:.4f}  "
            f"({r['stim_count']} stims)")
    lines.extend([
        "",
        f"MC PEAK:          γ = {mc_peak_gamma:.0f} cm⁻¹  "
        f"(MC = {mc_peak_val:.4f})",
    ])

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 2b ── ENAQT-GATED BNN RESERVOIR (CL SDK)",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.97,
             f"Cortical Labs CL SDK · ENAQT bridge: γ → P₄ → p_rel "
             f"· {N_SWEEP}-point dephasing sweep",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
