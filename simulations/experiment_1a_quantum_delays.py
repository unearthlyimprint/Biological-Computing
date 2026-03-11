#!/usr/bin/env python3
"""
experiment_1a_quantum_delays.py — Quantum vs Classical Synaptic Delays
=======================================================================

Phase 1, Experiment 1a of the Biological Computing research programme.

Compares two identical 1000-neuron LIF networks in BRIAN2:
  • CLASSICAL: Fixed synaptic delays (1.5 ms)
  • QUANTUM:   Delays drawn from a WKB tunnelling transmission distribution
               (exponentially skewed toward shorter delays)

Both networks share:
  - Same neuron parameters (tau, thresholds, refractory)
  - Same connectivity matrix (frozen random seed)
  - Same external Poisson drive
  - Same initial conditions

Outputs:
  - experiment_1a_results.png   (5-panel comparison dashboard)
  - experiment_1a_metrics.json  (numerical metric comparison)

Usage:
    python experiment_1a_quantum_delays.py
"""

import json
import time
import sys
import os

import numpy as np

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_delay_model import (
    tunnelling_delay_distribution,
    classical_delay_distribution,
    compute_all_metrics,
    compute_synchrony_index,
    compute_kappa,
)

# ══════════════════════════════════════════════════
# EXPERIMENT PARAMETERS
# ══════════════════════════════════════════════════

SEED = 42                    # reproducibility
N_EXC = 800                  # excitatory neurons
N_INH = 200                  # inhibitory neurons
N_TOTAL = N_EXC + N_INH

# LIF neuron parameters
TAU_M = 20                   # membrane time constant (ms)
V_REST = -65                 # resting potential (mV)
V_THRESH = -50               # spike threshold (mV)
V_RESET = -70                # post-spike reset (mV)
TAU_REF = 2                  # refractory period (ms)

# Synaptic parameters
W_EXC = 1.0                  # excitatory weight (mV)
W_INH = -3.0                 # inhibitory weight (mV)
P_CONNECT = 0.1              # connection probability

# External input
INPUT_RATE = 50              # Poisson rate (Hz)
N_INPUT = 100                # number of input neurons
W_INPUT = 2.0                # input weight (mV)

# Simulation
SIM_DURATION = 2.0           # seconds

# Classical delay
CLASSICAL_DELAY_MS = 1.5     # fixed delay for classical network

# Quantum tunnelling parameters
BARRIER_HEIGHT_EV = 0.3      # synaptic cleft barrier height
PARTICLE_ENERGY_EV = 0.1     # neurotransmitter kinetic energy
MASS_FACTOR = 1.0            # mass in proton units
DELAY_MIN_MS = 0.5           # minimum plausible delay
DELAY_MAX_MS = 4.0           # maximum plausible delay


def run_network(delay_values_ms, label, seed_offset=0):
    """
    Build and run a BRIAN2 LIF network with specified synaptic delays.

    Parameters
    ----------
    delay_values_ms : ndarray or float
        Delay values in ms. If scalar, all delays are identical.
    label : str
        Network label for printing
    seed_offset : int
        Offset added to seed for BRIAN2 internal RNG (same base seed)

    Returns
    -------
    results : dict
        Spike data and rate traces
    """
    # Import brian2 inside function to allow fresh namespace per run
    from brian2 import (
        NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
        PopulationRateMonitor, run, defaultclock, ms, mV, Hz, second,
        start_scope
    )

    start_scope()
    defaultclock.dt = 0.1 * ms
    np.random.seed(SEED + seed_offset)

    print(f"\n{'═' * 60}")
    print(f"  Running {label} network")
    print(f"{'═' * 60}")

    # ── Neuron groups ──
    eqs = """
    dv/dt = (V_rest_val - v) / tau_m_val : volt (unless refractory)
    V_rest_val : volt
    tau_m_val : second
    """

    exc = NeuronGroup(
        N_EXC, eqs,
        threshold=f"v > {V_THRESH}*mV",
        reset=f"v = {V_RESET}*mV",
        refractory=TAU_REF * ms,
        method="euler"
    )
    exc.v = f"{V_REST}*mV + rand() * ({V_THRESH - V_REST})*mV"
    exc.V_rest_val = V_REST * mV
    exc.tau_m_val = TAU_M * ms

    inh = NeuronGroup(
        N_INH, eqs,
        threshold=f"v > {V_THRESH}*mV",
        reset=f"v = {V_RESET}*mV",
        refractory=TAU_REF * ms,
        method="euler"
    )
    inh.v = f"{V_REST}*mV + rand() * ({V_THRESH - V_REST})*mV"
    inh.V_rest_val = V_REST * mV
    inh.tau_m_val = TAU_M * ms

    # ── Synapses with specified delays ──
    # E → E
    np.random.seed(SEED)  # freeze connectivity across runs
    syn_ee = Synapses(exc, exc, "w : volt", on_pre="v_post += w")
    syn_ee.connect(p=P_CONNECT)
    syn_ee.w = f"{W_EXC}*mV * (0.5 + 0.5 * rand())"

    # Assign delays
    n_ee = len(syn_ee)
    if np.isscalar(delay_values_ms) or len(np.unique(delay_values_ms)) == 1:
        val = float(delay_values_ms) if np.isscalar(delay_values_ms) else float(delay_values_ms[0])
        syn_ee.delay = val * ms
        actual_delays = np.full(n_ee, val)
    else:
        # Truncate or tile to match number of synapses
        if len(delay_values_ms) >= n_ee:
            delays = delay_values_ms[:n_ee]
        else:
            delays = np.tile(delay_values_ms,
                             int(np.ceil(n_ee / len(delay_values_ms))))[:n_ee]
        syn_ee.delay = delays * ms
        actual_delays = delays

    # E → I
    syn_ei = Synapses(exc, inh, on_pre=f"v_post += {W_EXC}*mV")
    syn_ei.connect(p=P_CONNECT)
    syn_ei.delay = np.median(actual_delays) * ms

    # I → E
    syn_ie = Synapses(inh, exc, on_pre=f"v_post += {W_INH}*mV")
    syn_ie.connect(p=P_CONNECT)
    syn_ie.delay = np.median(actual_delays) * ms

    # I → I
    syn_ii = Synapses(inh, inh, on_pre=f"v_post += {W_INH}*mV")
    syn_ii.connect(p=P_CONNECT)
    syn_ii.delay = np.median(actual_delays) * ms

    # ── External input ──
    poisson = PoissonGroup(N_INPUT, rates=INPUT_RATE * Hz)
    syn_in = Synapses(poisson, exc, on_pre=f"v_post += {W_INPUT}*mV")
    syn_in.connect(p=0.1)

    # ── Monitors ──
    spike_exc = SpikeMonitor(exc)
    spike_inh = SpikeMonitor(inh)
    rate_exc = PopulationRateMonitor(exc)

    # ── Run ──
    t0 = time.time()
    run(SIM_DURATION * second, report="text")
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Exc spikes: {spike_exc.num_spikes:,}")
    print(f"  Inh spikes: {spike_inh.num_spikes:,}")

    # Extract data (convert from brian2 quantities)
    exc_i = np.array(spike_exc.i)
    exc_t = np.array(spike_exc.t / second)
    inh_i = np.array(spike_inh.i)
    inh_t = np.array(spike_inh.t / second)
    rate_t = np.array(rate_exc.t / second)
    rate_smooth = np.array(rate_exc.smooth_rate(width=10 * ms) / Hz)

    return {
        'exc_i': exc_i, 'exc_t': exc_t,
        'inh_i': inh_i, 'inh_t': inh_t,
        'rate_t': rate_t, 'rate_smooth': rate_smooth,
        'delays': actual_delays,
        'elapsed_s': elapsed,
        'n_ee_synapses': n_ee,
    }


def make_dashboard(classical, quantum, c_metrics, q_metrics,
                   c_delays, q_delays, output_path):
    """
    Generate the 5-panel comparison dashboard.
    Scientific Data Observatory design system.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ── Design tokens ──
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
    FONT_MONO = "monospace"

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25,
                  height_ratios=[2.5, 1.5, 1.5])

    def style_ax(ax, title=""):
        ax.set_facecolor(BG_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if title:
            ax.set_title(title, color=ACCENT_CYAN, fontsize=11,
                         fontfamily=FONT_MONO, fontweight="bold",
                         loc="left", pad=10)

    # ── Panel A: Raster plots ──
    ax_raster_c = fig.add_subplot(gs[0, 0])
    style_ax(ax_raster_c, "◈  CLASSICAL — RASTER PLOT")
    ax_raster_c.scatter(classical['exc_t'] * 1000, classical['exc_i'],
                        s=0.15, c=ACCENT_CYAN, alpha=0.5, rasterized=True)
    ax_raster_c.scatter(classical['inh_t'] * 1000,
                        classical['inh_i'] + N_EXC,
                        s=0.15, c=ACCENT_RED, alpha=0.5, rasterized=True)
    ax_raster_c.set_ylabel("Neuron Index", color=TEXT_SECONDARY,
                           fontfamily=FONT_MONO, fontsize=9)
    ax_raster_c.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                           fontfamily=FONT_MONO, fontsize=9)

    ax_raster_q = fig.add_subplot(gs[0, 1])
    style_ax(ax_raster_q, "◈  QUANTUM — RASTER PLOT")
    ax_raster_q.scatter(quantum['exc_t'] * 1000, quantum['exc_i'],
                        s=0.15, c=ACCENT_TEAL, alpha=0.5, rasterized=True)
    ax_raster_q.scatter(quantum['inh_t'] * 1000,
                        quantum['inh_i'] + N_EXC,
                        s=0.15, c=ACCENT_RED, alpha=0.5, rasterized=True)
    ax_raster_q.set_ylabel("Neuron Index", color=TEXT_SECONDARY,
                           fontfamily=FONT_MONO, fontsize=9)
    ax_raster_q.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                           fontfamily=FONT_MONO, fontsize=9)

    # ── Panel B: Population rate overlay ──
    ax_rate = fig.add_subplot(gs[1, 0])
    style_ax(ax_rate, "◈  POPULATION FIRING RATE")
    ax_rate.plot(classical['rate_t'] * 1000, classical['rate_smooth'],
                 color=ACCENT_CYAN, linewidth=1.0, alpha=0.8,
                 label="Classical")
    ax_rate.plot(quantum['rate_t'] * 1000, quantum['rate_smooth'],
                 color=ACCENT_TEAL, linewidth=1.0, alpha=0.8,
                 label="Quantum")
    ax_rate.fill_between(classical['rate_t'] * 1000,
                         classical['rate_smooth'],
                         alpha=0.08, color=ACCENT_CYAN)
    ax_rate.fill_between(quantum['rate_t'] * 1000,
                         quantum['rate_smooth'],
                         alpha=0.08, color=ACCENT_TEAL)
    ax_rate.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                       fontfamily=FONT_MONO, fontsize=9)
    ax_rate.set_ylabel("Rate (Hz)", color=TEXT_SECONDARY,
                       fontfamily=FONT_MONO, fontsize=9)
    ax_rate.legend(fontsize=8, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                   loc="upper right")

    # ── Panel C: ISI distribution ──
    ax_isi = fig.add_subplot(gs[1, 1])
    style_ax(ax_isi, "◈  INTER-SPIKE INTERVAL DISTRIBUTION")

    # Compute ISIs
    if len(classical['exc_t']) > 1:
        c_isis = []
        for i in range(N_EXC):
            mask = classical['exc_i'] == i
            ts = np.sort(classical['exc_t'][mask])
            if len(ts) > 1:
                c_isis.extend(np.diff(ts) * 1000)
        if c_isis:
            ax_isi.hist(c_isis, bins=80, range=(0, 200), density=True,
                        alpha=0.5, color=ACCENT_CYAN, label="Classical",
                        edgecolor="none")

    if len(quantum['exc_t']) > 1:
        q_isis = []
        for i in range(N_EXC):
            mask = quantum['exc_i'] == i
            ts = np.sort(quantum['exc_t'][mask])
            if len(ts) > 1:
                q_isis.extend(np.diff(ts) * 1000)
        if q_isis:
            ax_isi.hist(q_isis, bins=80, range=(0, 200), density=True,
                        alpha=0.5, color=ACCENT_TEAL, label="Quantum",
                        edgecolor="none")

    ax_isi.set_xlabel("ISI (ms)", color=TEXT_SECONDARY,
                      fontfamily=FONT_MONO, fontsize=9)
    ax_isi.set_ylabel("Density", color=TEXT_SECONDARY,
                      fontfamily=FONT_MONO, fontsize=9)
    ax_isi.legend(fontsize=8, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Delay distributions ──
    ax_delay = fig.add_subplot(gs[2, 0])
    style_ax(ax_delay, "◈  SYNAPTIC DELAY DISTRIBUTION")
    ax_delay.hist(c_delays, bins=60, range=(0, 5), density=True,
                  alpha=0.6, color=ACCENT_CYAN, label="Classical (fixed)",
                  edgecolor="none")
    ax_delay.hist(q_delays, bins=60, range=(0, 5), density=True,
                  alpha=0.5, color=ACCENT_TEAL, label="Quantum (tunnelling)",
                  edgecolor="none")
    ax_delay.set_xlabel("Delay (ms)", color=TEXT_SECONDARY,
                        fontfamily=FONT_MONO, fontsize=9)
    ax_delay.set_ylabel("Density", color=TEXT_SECONDARY,
                        fontfamily=FONT_MONO, fontsize=9)
    ax_delay.legend(fontsize=8, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Metrics comparison ──
    ax_metrics = fig.add_subplot(gs[2, 1])
    style_ax(ax_metrics, "◈  NETWORK METRICS COMPARISON")

    metric_names = ["Rate (Hz)", "CV_ISI", "Fano Factor",
                    "Entropy\n(norm.)", "N Spikes\n(×10³)"]
    c_vals = [
        c_metrics['mean_rate_hz'],
        c_metrics['cv_isi_mean'],
        c_metrics['fano_factor'],
        c_metrics['spike_entropy'] / max(c_metrics['max_entropy'], 1),
        c_metrics['n_spikes'] / 1000,
    ]
    q_vals = [
        q_metrics['mean_rate_hz'],
        q_metrics['cv_isi_mean'],
        q_metrics['fano_factor'],
        q_metrics['spike_entropy'] / max(q_metrics['max_entropy'], 1),
        q_metrics['n_spikes'] / 1000,
    ]

    x = np.arange(len(metric_names))
    width = 0.35
    bars_c = ax_metrics.bar(x - width / 2, c_vals, width,
                            color=ACCENT_CYAN, alpha=0.8, label="Classical")
    bars_q = ax_metrics.bar(x + width / 2, q_vals, width,
                            color=ACCENT_TEAL, alpha=0.8, label="Quantum")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_names, fontsize=8, color=TEXT_SECONDARY,
                               fontfamily=FONT_MONO)
    ax_metrics.legend(fontsize=8, facecolor=BG_SECONDARY,
                      edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # Value labels on bars
    for bar in list(bars_c) + list(bars_q):
        h = bar.get_height()
        ax_metrics.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.2f}", ha="center", va="bottom",
                        color=TEXT_SECONDARY, fontsize=7,
                        fontfamily=FONT_MONO)

    # ── Supertitle ──
    fig.suptitle(
        "EXPERIMENT 1a ── QUANTUM vs CLASSICAL SYNAPTIC DELAYS",
        color=ACCENT_CYAN, fontsize=16, fontfamily=FONT_MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             f"N={N_TOTAL} LIF neurons  ·  {SIM_DURATION}s  ·  "
             f"seed={SEED}  ·  barrier={BARRIER_HEIGHT_EV} eV",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=FONT_MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_1a_results.png")
    output_json = os.path.join(script_dir, "experiment_1a_metrics.json")

    print("=" * 60)
    print("  EXPERIMENT 1a — QUANTUM vs CLASSICAL SYNAPTIC DELAYS")
    print("  Biological Computing Research Programme · Phase 1")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    # ── Generate delay distributions ──
    kappa = compute_kappa(BARRIER_HEIGHT_EV, PARTICLE_ENERGY_EV, MASS_FACTOR)
    n_synapse_estimate = int(N_EXC * N_EXC * P_CONNECT)  # ~64000

    q_delays = tunnelling_delay_distribution(
        n_synapse_estimate,
        d_min_ms=DELAY_MIN_MS,
        d_max_ms=DELAY_MAX_MS,
        kappa=kappa,
        rng=rng
    )
    c_delays = classical_delay_distribution(
        n_synapse_estimate,
        delay_ms=CLASSICAL_DELAY_MS
    )

    print(f"\n  Tunnelling κ = {kappa:.2e} m⁻¹")
    print(f"  Quantum delays:   mean={np.mean(q_delays):.2f} ms, "
          f"std={np.std(q_delays):.2f} ms")
    print(f"  Classical delays:  mean={np.mean(c_delays):.2f} ms, "
          f"std={np.std(c_delays):.2f} ms")

    # ── Run classical network ──
    classical = run_network(c_delays, "CLASSICAL", seed_offset=0)

    # ── Run quantum network ──
    quantum = run_network(q_delays, "QUANTUM", seed_offset=0)

    # ── Compute metrics ──
    print(f"\n{'─' * 60}")
    print("  Computing network metrics...")

    c_metrics = compute_all_metrics(
        classical['exc_i'], classical['exc_t'], N_EXC, SIM_DURATION
    )
    q_metrics = compute_all_metrics(
        quantum['exc_i'], quantum['exc_t'], N_EXC, SIM_DURATION
    )

    # Compute synchrony between excitatory rate traces
    min_len = min(len(classical['rate_smooth']), len(quantum['rate_smooth']))
    if min_len > 10:
        synchrony = compute_synchrony_index(
            classical['rate_smooth'][:min_len],
            quantum['rate_smooth'][:min_len]
        )
    else:
        synchrony = 0.0

    # ── Print comparison ──
    print(f"\n{'═' * 60}")
    print("  RESULTS COMPARISON")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<25} {'Classical':>12} {'Quantum':>12} {'Δ':>10}")
    print(f"  {'─' * 59}")

    for key in ['n_spikes', 'mean_rate_hz', 'cv_isi_mean',
                'fano_factor', 'spike_entropy']:
        cv = c_metrics[key]
        qv = q_metrics[key]
        delta = qv - cv
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<25} {cv:>12.4f} {qv:>12.4f} {sign}{delta:>9.4f}")

    print(f"  {'cross_synchrony':<25} {synchrony:>12.4f}")
    print(f"{'═' * 60}")

    # ── Save metrics ──
    results = {
        'experiment': '1a_quantum_vs_classical_delays',
        'parameters': {
            'n_exc': N_EXC,
            'n_inh': N_INH,
            'p_connect': P_CONNECT,
            'sim_duration_s': SIM_DURATION,
            'seed': SEED,
            'barrier_height_eV': BARRIER_HEIGHT_EV,
            'particle_energy_eV': PARTICLE_ENERGY_EV,
            'kappa_m_inv': float(kappa),
            'classical_delay_ms': CLASSICAL_DELAY_MS,
            'quantum_delay_mean_ms': float(np.mean(q_delays)),
            'quantum_delay_std_ms': float(np.std(q_delays)),
        },
        'classical': c_metrics,
        'quantum': q_metrics,
        'cross_synchrony': synchrony,
        'timing': {
            'classical_elapsed_s': classical['elapsed_s'],
            'quantum_elapsed_s': quantum['elapsed_s'],
        }
    }

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Generate dashboard ──
    make_dashboard(classical, quantum, c_metrics, q_metrics,
                   c_delays, q_delays, output_img)

    print(f"\n{'═' * 60}")
    print("  EXPERIMENT 1a COMPLETE")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
