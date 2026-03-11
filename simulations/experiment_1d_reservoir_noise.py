#!/usr/bin/env python3
"""
experiment_1d_reservoir_noise.py — Reservoir Computing: Quantum vs Classical Noise
==================================================================================

Phase 1, Experiment 1d of the Biological Computing research programme.

Tests whether quantum-scale noise injection improves the computational
capacity of a reservoir computer compared to classical Gaussian noise
and a no-noise baseline.

Model:
  - 200-neuron Echo State Network (ESN) with tanh activation
  - Sparse random recurrent connectivity (p=0.1), spectral radius ~0.9
  - Single time-varying input signal to all neurons (weighted)
  - Linear readout (Ridge regression) from reservoir states → target

Three noise conditions (matched energy):
  1. No noise — pure recurrent dynamics
  2. Classical — Gaussian noise injection
  3. Quantum — tunnelling-distributed noise (WKB, asymmetric)

Metrics:
  - Memory capacity MC = Σ R²(k) for k=1..K  (linear delayed recall)
  - Nonlinear capacity NLC — R² for polynomial targets u²(t), u(t)·u(t-1)
  - Noise amplitude sweep — MC vs noise level (edge-of-chaos search)

Key physics:
  Quantum noise (tunnelling distribution) is inherently asymmetric and
  heavy-tailed, producing richer temporal correlations than Gaussian noise
  of the same variance. This may push the reservoir closer to the
  edge-of-chaos regime where computational capacity is maximised.

Outputs:
  - experiment_1d_results.png   (5-panel dashboard)
  - experiment_1d_metrics.json  (numerical results)

References:
  Jaeger, H. (2001). "The echo state approach to analysing and training RNNs."
  Maass, W. et al. (2002). "Real-time computing without stable states."
  Lukoševičius, M. & Jaeger, H. (2009). "Reservoir computing approaches
    to recurrent neural network training." Computer Science Review 3, 127–149.

Usage:
    python experiment_1d_reservoir_noise.py
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_delay_model import tunnelling_delay_distribution


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

SEED = 42

# Reservoir architecture (Echo State Network)
N_RESERVOIR = 200           # reservoir neurons
P_CONNECT = 0.1             # recurrent connection probability
SPECTRAL_RADIUS = 0.9       # controls echo state property
W_INPUT_SCALE = 1.5         # input weight scaling — pushes into nonlinear tanh regime
LEAK_RATE = 0.3             # leaky integration rate (0=slow, 1=instant)

# Simulation
N_STEPS = 5000              # total time steps
N_WASHOUT = 200             # discard initial transient

# Memory capacity test
MAX_DELAY = 30              # test u(t-k) for k = 1..MAX_DELAY
RIDGE_ALPHA = 0.01          # Ridge regularisation
TRAIN_FRAC = 0.7            # fraction of data for training

# Noise parameters
NOISE_BASE = 0.05           # baseline noise amplitude (units of activation)
NOISE_SWEEP_MULTIPLIERS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

# Quantum tunnelling parameters (for noise distribution shape)
BARRIER_HEIGHT_EV = 0.3
PARTICLE_ENERGY_EV = 0.1
MASS_FACTOR = 1.0


# ══════════════════════════════════════════════════
# ECHO STATE NETWORK
# ══════════════════════════════════════════════════

def create_reservoir_weights(n_neurons, p_connect, spectral_radius, rng):
    """
    Create a sparse random reservoir weight matrix with specified
    spectral radius.

    Parameters
    ----------
    n_neurons : int
    p_connect : float
        Connection probability
    spectral_radius : float
        Desired spectral radius of W
    rng : numpy.random.Generator

    Returns
    -------
    W : ndarray (n_neurons, n_neurons)
        Reservoir weight matrix
    """
    # Sparse random matrix
    W = rng.standard_normal((n_neurons, n_neurons))
    mask = rng.random((n_neurons, n_neurons)) < p_connect
    W *= mask

    # Scale to desired spectral radius
    eigenvalues = np.linalg.eigvals(W)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if max_eigenvalue > 0:
        W = W * (spectral_radius / max_eigenvalue)

    return W


def run_esn(input_signal, W_res, W_in, noise_type="none",
            noise_amplitude=0.0, rng=None):
    """
    Run an Echo State Network with leaky integration.

    x(t) = (1-α)·x(t-1) + α·tanh(W_res·x(t-1) + W_in·u(t) + noise(t))

    Parameters
    ----------
    input_signal : ndarray (n_steps,)
        Input u(t) ∈ [-0.5, 0.5]
    W_res : ndarray (N, N)
        Reservoir weight matrix
    W_in : ndarray (N,)
        Input weight vector
    noise_type : str
        "none", "classical" (Gaussian), "quantum" (tunnelling)
    noise_amplitude : float
    rng : numpy.random.Generator

    Returns
    -------
    states : ndarray (n_steps, N)
        Reservoir state at each time step
    """
    n_steps = len(input_signal)
    N = W_res.shape[0]
    states = np.zeros((n_steps, N))
    x = np.zeros(N)  # initial state

    # Pre-generate noise
    if noise_type == "classical" and noise_amplitude > 0:
        noise = rng.normal(0, noise_amplitude, size=(n_steps, N))
    elif noise_type == "quantum" and noise_amplitude > 0:
        raw = tunnelling_delay_distribution(
            n_steps * N,
            d_min_ms=0.0, d_max_ms=1.0,
            barrier_height_eV=BARRIER_HEIGHT_EV,
            particle_energy_eV=PARTICLE_ENERGY_EV,
            mass_factor=MASS_FACTOR,
            rng=rng
        )
        # Centre and scale to match classical noise variance
        raw_centred = raw - np.mean(raw)
        raw_std = np.std(raw) if np.std(raw) > 0 else 1.0
        noise = (raw_centred / raw_std * noise_amplitude).reshape(n_steps, N)
    else:
        noise = np.zeros((n_steps, N))

    # Run reservoir
    for t in range(n_steps):
        pre_activation = W_res @ x + W_in * input_signal[t] + noise[t]
        x = (1 - LEAK_RATE) * x + LEAK_RATE * np.tanh(pre_activation)
        states[t] = x

    return states


# ══════════════════════════════════════════════════
# MEMORY CAPACITY
# ══════════════════════════════════════════════════

def compute_memory_capacity(X, u, max_delay=MAX_DELAY):
    """
    Compute memory capacity via linear delayed recall.

    MC = Σ_{k=1}^{K} R²(k)

    where R²(k) is the coefficient of determination for predicting
    u(t-k) from the reservoir state X(t) via Ridge regression.
    """
    n = len(u)
    mc_per_delay = np.zeros(max_delay)

    for k in range(1, max_delay + 1):
        # Target: u(t-k) — shifted input
        target = u[:-k] if k < n else np.zeros(n)
        reservoir_states = X[k:]  # align with target

        if len(target) < 10 or len(reservoir_states) < 10:
            continue

        min_len = min(len(target), len(reservoir_states))
        target = target[:min_len]
        reservoir_states = reservoir_states[:min_len]

        n_tr = int(min_len * TRAIN_FRAC)
        if n_tr < 5:
            continue

        X_train = reservoir_states[:n_tr]
        X_test = reservoir_states[n_tr:]
        y_train = target[:n_tr]
        y_test = target[n_tr:]

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mc_per_delay[k - 1] = max(0.0, r2)

    mc_total = float(np.sum(mc_per_delay))
    return mc_total, mc_per_delay


def compute_nonlinear_capacity(X, u):
    """
    Compute nonlinear transformation capacity.

    Tests:
      1. u²(t)        — instantaneous nonlinearity
      2. u(t)·u(t-1)  — cross-temporal nonlinearity
    """
    n = len(u)
    n_tr = int(n * TRAIN_FRAC)
    nlc = {}

    # u²(t)
    target_sq = u ** 2
    min_len = min(len(target_sq), len(X))
    X_tr, X_te = X[:n_tr], X[n_tr:min_len]
    y_tr, y_te = target_sq[:n_tr], target_sq[n_tr:min_len]
    if len(y_te) > 5:
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_tr, y_tr)
        nlc['square_r2'] = max(0.0, r2_score(y_te, model.predict(X_te)))
    else:
        nlc['square_r2'] = 0.0

    # u(t)·u(t-1)
    target_cross = u[1:] * u[:-1]
    X_shifted = X[1:]
    min_len = min(len(target_cross), len(X_shifted))
    X_tr, X_te = X_shifted[:n_tr], X_shifted[n_tr:min_len]
    y_tr, y_te = target_cross[:n_tr], target_cross[n_tr:min_len]
    if len(y_te) > 5:
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_tr, y_tr)
        nlc['cross_r2'] = max(0.0, r2_score(y_te, model.predict(X_te)))
    else:
        nlc['cross_r2'] = 0.0

    return nlc


# ══════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_1d_results.png")
    output_json = os.path.join(script_dir, "experiment_1d_metrics.json")

    print("=" * 60)
    print("  EXPERIMENT 1d — RESERVOIR COMPUTING: QUANTUM vs CLASSICAL")
    print("  Biological Computing Research Programme · Phase 1")
    print("=" * 60)

    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # ── Create reservoir ──
    W_res = create_reservoir_weights(N_RESERVOIR, P_CONNECT,
                                      SPECTRAL_RADIUS, rng)
    W_in = rng.uniform(-W_INPUT_SCALE, W_INPUT_SCALE, N_RESERVOIR)

    # ── Generate input signal ──
    # Uniform in [-0.5, 0.5] — standard for MC tests
    input_signal = rng.uniform(-0.5, 0.5, N_STEPS)

    actual_sr = np.max(np.abs(np.linalg.eigvals(W_res)))
    print(f"\n  Reservoir: {N_RESERVOIR} neurons, "
          f"p={P_CONNECT}, ρ={actual_sr:.3f}")
    print(f"  Leak rate: {LEAK_RATE}")
    print(f"  Simulation: {N_STEPS} steps, {N_WASHOUT} washout")
    print(f"  Memory capacity test: k = 1..{MAX_DELAY}")

    # ═══════════════════════════════════════════════
    # MAIN COMPARISON: 3 conditions at baseline noise
    # ═══════════════════════════════════════════════
    conditions = [
        ("No noise",   "none",      0.0),
        ("Classical",  "classical", NOISE_BASE),
        ("Quantum",    "quantum",   NOISE_BASE),
    ]

    results = {}
    for label, noise_type, amplitude in conditions:
        print(f"\n{'─' * 60}")
        print(f"  Running condition: {label} "
              f"(type={noise_type}, amp={amplitude:.3f})")
        print(f"{'─' * 60}")

        t0 = time.time()
        states = run_esn(
            input_signal, W_res, W_in,
            noise_type=noise_type,
            noise_amplitude=amplitude,
            rng=np.random.default_rng(SEED + 1)
        )
        elapsed = time.time() - t0

        # Discard washout
        X = states[N_WASHOUT:]
        u = input_signal[N_WASHOUT:]

        # Memory capacity
        mc_total, mc_per_delay = compute_memory_capacity(X, u)

        # Nonlinear capacity
        nlc = compute_nonlinear_capacity(X, u)

        # State statistics
        mean_act = np.mean(np.abs(X))
        std_act = np.std(X)

        results[label] = {
            'states': states,
            'X': X, 'u': u,
            'mc_total': mc_total,
            'mc_per_delay': mc_per_delay,
            'nlc': nlc,
            'mean_activation': float(mean_act),
            'std_activation': float(std_act),
            'elapsed_s': elapsed,
        }

        print(f"  ◎ Memory capacity: MC = {mc_total:.2f}")
        print(f"  ◎ Nonlinear R²(u²): {nlc['square_r2']:.3f}")
        print(f"  ◎ Nonlinear R²(u·u₋₁): {nlc['cross_r2']:.3f}")
        print(f"  ◎ Mean |activation|: {mean_act:.4f}")
        print(f"  ◎ Elapsed: {elapsed:.2f}s")

    # ═══════════════════════════════════════════════
    # NOISE AMPLITUDE SWEEP
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep: Memory capacity vs noise amplitude")
    print(f"{'─' * 60}")

    sweep_classical = []
    sweep_quantum = []

    for mult in NOISE_SWEEP_MULTIPLIERS:
        amp = mult  # direct amplitude (not multiplied by base)

        # Classical
        states_c = run_esn(
            input_signal, W_res, W_in,
            noise_type="classical", noise_amplitude=amp,
            rng=np.random.default_rng(SEED + 2)
        )
        X_c = states_c[N_WASHOUT:]
        mc_c, _ = compute_memory_capacity(X_c, input_signal[N_WASHOUT:])
        sweep_classical.append({'amplitude': amp, 'mc': mc_c})

        # Quantum
        states_q = run_esn(
            input_signal, W_res, W_in,
            noise_type="quantum", noise_amplitude=amp,
            rng=np.random.default_rng(SEED + 3)
        )
        X_q = states_q[N_WASHOUT:]
        mc_q, _ = compute_memory_capacity(X_q, input_signal[N_WASHOUT:])
        sweep_quantum.append({'amplitude': amp, 'mc': mc_q})

        print(f"    amp={amp:6.3f}  "
              f"MC_classical={mc_c:.2f}  MC_quantum={mc_q:.2f}")

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed_total = time.time() - t_start
    print(f"\n{'═' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 60}")

    for label in ["No noise", "Classical", "Quantum"]:
        r = results[label]
        print(f"  {label:12s}  MC={r['mc_total']:5.2f}  "
              f"NLC_sq={r['nlc']['square_r2']:.3f}  "
              f"NLC_cross={r['nlc']['cross_r2']:.3f}  "
              f"|act|={r['mean_activation']:.4f}")

    best_q = max(sweep_quantum, key=lambda x: x['mc'])
    best_c = max(sweep_classical, key=lambda x: x['mc'])
    print(f"\n  Best classical MC: {best_c['mc']:.2f} "
          f"at amp={best_c['amplitude']:.3f}")
    print(f"  Best quantum MC:   {best_q['mc']:.2f} "
          f"at amp={best_q['amplitude']:.3f}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")
    print(f"{'═' * 60}")

    # ── Save metrics ──
    metrics = {
        'experiment': '1d_reservoir_noise',
        'parameters': {
            'n_reservoir': N_RESERVOIR,
            'p_connect': P_CONNECT,
            'spectral_radius': SPECTRAL_RADIUS,
            'w_input_scale': W_INPUT_SCALE,
            'leak_rate': LEAK_RATE,
            'n_steps': N_STEPS,
            'n_washout': N_WASHOUT,
            'noise_base': NOISE_BASE,
            'max_delay': MAX_DELAY,
            'ridge_alpha': RIDGE_ALPHA,
            'seed': SEED,
        },
        'conditions': {},
        'noise_sweep': {
            'amplitudes': NOISE_SWEEP_MULTIPLIERS,
            'classical': [{'amp': s['amplitude'], 'mc': round(s['mc'], 4)}
                          for s in sweep_classical],
            'quantum': [{'amp': s['amplitude'], 'mc': round(s['mc'], 4)}
                        for s in sweep_quantum],
            'best_classical': {
                'amplitude': best_c['amplitude'],
                'mc': round(best_c['mc'], 4),
            },
            'best_quantum': {
                'amplitude': best_q['amplitude'],
                'mc': round(best_q['mc'], 4),
            },
        },
        'elapsed_s': round(elapsed_total, 1),
    }

    for label in ["No noise", "Classical", "Quantum"]:
        r = results[label]
        metrics['conditions'][label] = {
            'mc_total': round(r['mc_total'], 4),
            'mc_per_delay': [round(x, 4) for x in r['mc_per_delay']],
            'nlc_square_r2': round(r['nlc']['square_r2'], 4),
            'nlc_cross_r2': round(r['nlc']['cross_r2'], 4),
            'mean_activation': round(r['mean_activation'], 4),
            'std_activation': round(r['std_activation'], 4),
        }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Generate dashboard ──
    make_dashboard(results, sweep_classical, sweep_quantum, output_img)

    print(f"\n{'═' * 60}")
    print("  EXPERIMENT 1d COMPLETE")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(results, sweep_classical, sweep_quantum, output_path):
    """5-panel Scientific Data Observatory dashboard."""
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

    condition_colors = {
        "No noise": ACCENT_AMBER,
        "Classical": ACCENT_CYAN,
        "Quantum": ACCENT_TEAL,
    }

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 2, hspace=0.38, wspace=0.28,
                  height_ratios=[1.2, 1.5, 1.5])

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

    # ── Panel A: Reservoir state trajectories ──
    ax_traj = fig.add_subplot(gs[0, :])
    style_ax(ax_traj, "◈  RESERVOIR STATE TRAJECTORIES — FIRST 5 NEURONS")

    t_show = 500  # show first 500 steps after washout
    for label in ["No noise", "Classical", "Quantum"]:
        X = results[label]['X']
        color = condition_colors[label]
        for neuron_idx in range(5):
            alpha = 0.6 if neuron_idx > 0 else 0.9
            lw = 1.0 if neuron_idx > 0 else 1.5
            lab = label if neuron_idx == 0 else None
            ax_traj.plot(np.arange(t_show), X[:t_show, neuron_idx],
                        color=color, alpha=alpha, linewidth=lw, label=lab)

    ax_traj.set_xlabel("Time step", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_traj.set_ylabel("Activation", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_traj.legend(fontsize=8, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                   loc="upper right")

    # ── Panel B: Memory capacity MC(k) curves ──
    ax_mc = fig.add_subplot(gs[1, 0])
    style_ax(ax_mc, "◈  MEMORY CAPACITY MC(k)")

    delays = np.arange(1, MAX_DELAY + 1)
    for label in ["No noise", "Classical", "Quantum"]:
        mc_k = results[label]['mc_per_delay']
        color = condition_colors[label]
        ax_mc.plot(delays, mc_k, color=color, linewidth=2, alpha=0.85,
                   marker='o', markersize=4, label=f"{label} "
                   f"(MC={results[label]['mc_total']:.2f})")
    ax_mc.set_xlabel("Delay k (steps)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylabel("R²(k)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylim(-0.02, 1.02)
    ax_mc.legend(fontsize=8, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY, loc="upper right")

    # ── Panel C: Noise amplitude sweep ──
    ax_sweep = fig.add_subplot(gs[1, 1])
    style_ax(ax_sweep, "◈  NOISE AMPLITUDE SWEEP — MC vs NOISE LEVEL")

    amps_c = [s['amplitude'] for s in sweep_classical]
    mc_c = [s['mc'] for s in sweep_classical]
    amps_q = [s['amplitude'] for s in sweep_quantum]
    mc_q = [s['mc'] for s in sweep_quantum]

    ax_sweep.plot(amps_c, mc_c, color=ACCENT_CYAN, linewidth=2.0,
                  marker='s', markersize=6, alpha=0.85, label="Classical")
    ax_sweep.plot(amps_q, mc_q, color=ACCENT_TEAL, linewidth=2.0,
                  marker='D', markersize=6, alpha=0.85, label="Quantum")
    ax_sweep.set_xlabel("Noise amplitude", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_sweep.set_ylabel("Memory capacity MC", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_sweep.legend(fontsize=8, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Nonlinear capacity (bar chart) ──
    ax_nlc = fig.add_subplot(gs[2, 0])
    style_ax(ax_nlc, "◈  NONLINEAR CAPACITY")

    labels_cond = ["No noise", "Classical", "Quantum"]
    sq_vals = [results[l]['nlc']['square_r2'] for l in labels_cond]
    cross_vals = [results[l]['nlc']['cross_r2'] for l in labels_cond]

    x = np.arange(len(labels_cond))
    width = 0.35
    bars_sq = ax_nlc.bar(x - width / 2, sq_vals, width, alpha=0.8,
                         label="R²(u²)", color=ACCENT_GREEN)
    bars_cr = ax_nlc.bar(x + width / 2, cross_vals, width, alpha=0.8,
                         label="R²(u·u₋₁)", color=ACCENT_AMBER)

    ax_nlc.set_xticks(x)
    ax_nlc.set_xticklabels(labels_cond, fontsize=9, color=TEXT_SECONDARY,
                           fontfamily=MONO)
    ax_nlc.set_ylabel("R²", color=TEXT_SECONDARY, fontfamily=MONO, fontsize=9)
    ax_nlc.legend(fontsize=8, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # Value labels
    for bars in [bars_sq, bars_cr]:
        for bar in bars:
            h = bar.get_height()
            ax_nlc.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom",
                        color=TEXT_SECONDARY, fontsize=7, fontfamily=MONO)

    # ── Panel E: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  RESERVOIR COMPUTING — NOISE COMPARISON")
    ax_sum.axis("off")

    r_nn = results["No noise"]
    r_cl = results["Classical"]
    r_qu = results["Quantum"]

    best_c = max(sweep_classical, key=lambda x: x['mc'])
    best_q = max(sweep_quantum, key=lambda x: x['mc'])

    lines = [
        f"RESERVOIR:  {N_RESERVOIR} neurons, ρ={SPECTRAL_RADIUS}",
        f"LEAK RATE:  {LEAK_RATE}  ·  "
        f"MAX DELAY:  k={MAX_DELAY}",
        "",
        f"NO NOISE:    MC = {r_nn['mc_total']:5.2f}  "
        f"NLC(u²) = {r_nn['nlc']['square_r2']:.3f}  "
        f"|act| = {r_nn['mean_activation']:.4f}",
        f"CLASSICAL:   MC = {r_cl['mc_total']:5.2f}  "
        f"NLC(u²) = {r_cl['nlc']['square_r2']:.3f}  "
        f"|act| = {r_cl['mean_activation']:.4f}",
        f"QUANTUM:     MC = {r_qu['mc_total']:5.2f}  "
        f"NLC(u²) = {r_qu['nlc']['square_r2']:.3f}  "
        f"|act| = {r_qu['mean_activation']:.4f}",
        "",
        f"SWEEP BEST CLASSICAL:  MC = {best_c['mc']:.2f}  "
        f"at amp={best_c['amplitude']:.3f}",
        f"SWEEP BEST QUANTUM:    MC = {best_q['mc']:.2f}  "
        f"at amp={best_q['amplitude']:.3f}",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=10, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 1d ── RESERVOIR COMPUTING: QUANTUM vs CLASSICAL NOISE",
        color=ACCENT_CYAN, fontsize=15, fontfamily=MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             f"{N_RESERVOIR}-neuron ESN · ρ={SPECTRAL_RADIUS} · "
             f"Ridge readout · MC test",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
