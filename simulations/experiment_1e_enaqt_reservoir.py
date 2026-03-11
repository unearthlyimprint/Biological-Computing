#!/usr/bin/env python3
"""
experiment_1e_enaqt_reservoir.py — ENAQT-Modulated Synaptic Reservoir Computing
=================================================================================

Phase 1, Experiment 1e of the Biological Computing research programme.

BRIDGE EXPERIMENT: Uses quantum transport efficiency (P₄) from Experiment 1c
as the synaptic release probability in the reservoir computer from Experiment 1d.

Hypothesis:
  If ion channel transport is quantum-enhanced by ENAQT, and ion channels
  control synaptic vesicle release, then the ENAQT peak should correspond
  to a peak in network-level computational capacity.

Model:
  - 200-neuron Echo State Network (ESN) with tanh activation
  - Synaptic release probability = P₄(γ_dephasing) from ENAQT model
  - Each recurrent connection fires with probability P_release per timestep
  - P_release = transport efficiency of the 4-site ion channel model

Physics:
  Ca²⁺ channels in presynaptic terminals control vesicle fusion.
  If these channels exhibit ENAQT, the transport efficiency P₄ directly
  modulates the probability of neurotransmitter release. At body temperature
  (310 K, γ ≈ 215 cm⁻¹), P₄ ≈ 0.18 — within the biological range of
  synaptic release probabilities (0.1–0.9).

Conditions:
  1. Coherent (γ ≈ 0):     P_release ≈ 0.063  — quantum interference limits transport
  2. Body temp (310 K):    P_release ≈ 0.180  — biologically relevant ENAQT
  3. ENAQT peak:           P_release ≈ 0.418  — maximum quantum-classical synergy
  4. Full connectivity:    P_release = 1.000  — deterministic (classical) baseline

Plus: dephasing sweep γ → P₄ → MC to see if MC mirrors the ENAQT curve.

Outputs:
  - experiment_1e_results.png   (6-panel dashboard)
  - experiment_1e_metrics.json  (numerical results)

References:
  Plenio, M.B. & Huelga, S.F. (2008). New J. Phys. 10, 113019.
  Jaeger, H. (2001). "The echo state approach to analysing and training RNNs."
  Lukoševičius, M. & Jaeger, H. (2009). Computer Science Review 3, 127–149.

Usage:
    python experiment_1e_enaqt_reservoir.py
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
from experiment_1c_enaqt_ion_channel import compute_transport_efficiency


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

SEED = 42

# Reservoir architecture (Echo State Network — from Exp 1d)
N_RESERVOIR = 200
P_CONNECT = 0.1
SPECTRAL_RADIUS = 0.9
W_INPUT_SCALE = 1.5
LEAK_RATE = 0.3

# Simulation
N_STEPS = 5000
N_WASHOUT = 200

# Memory capacity test
MAX_DELAY = 30
RIDGE_ALPHA = 0.01
TRAIN_FRAC = 0.7

# Dephasing sweep for P₄ computation
N_DEPHASING_SWEEP = 40
GAMMA_MIN_CM = 0.01
GAMMA_MAX_CM = 1e5

# Key dephasing regimes (from Experiment 1c results)
GAMMA_COHERENT_CM = 0.01         # near-zero dephasing
GAMMA_BODY_CM = 215.46           # body temperature (310 K)
GAMMA_PEAK_CM = 1145.05          # ENAQT peak


# ══════════════════════════════════════════════════
# ECHO STATE NETWORK WITH SYNAPTIC DROPOUT
# ══════════════════════════════════════════════════

def create_reservoir_weights(n_neurons, p_connect, spectral_radius, rng):
    """Create sparse random reservoir weight matrix with specified spectral radius."""
    W = rng.standard_normal((n_neurons, n_neurons))
    mask = rng.random((n_neurons, n_neurons)) < p_connect
    W *= mask
    eigenvalues = np.linalg.eigvals(W)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if max_eigenvalue > 0:
        W = W * (spectral_radius / max_eigenvalue)
    return W


def run_esn_with_release(input_signal, W_res, W_in, p_release=1.0,
                          rng=None):
    """
    Run ESN with stochastic synaptic release (Bernoulli dropout on recurrence).

    x(t) = (1-α)·x(t-1) + α·tanh((W_res ⊙ mask(P_rel))·x(t-1) + W_in·u(t))

    Parameters
    ----------
    input_signal : ndarray (n_steps,)
    W_res : ndarray (N, N)
    W_in : ndarray (N,)
    p_release : float
        Synaptic release probability (0 to 1). Each connection fires
        independently with this probability at each timestep.
        Derived from ENAQT P₄.
    rng : numpy.random.Generator

    Returns
    -------
    states : ndarray (n_steps, N)
    """
    n_steps = len(input_signal)
    N = W_res.shape[0]
    states = np.zeros((n_steps, N))
    x = np.zeros(N)

    # Identify which connections exist (non-zero in W_res)
    W_nonzero = W_res != 0

    for t in range(n_steps):
        if p_release < 1.0:
            # Bernoulli mask: each existing synapse fires with probability p_release
            release_mask = (rng.random(W_res.shape) < p_release) & W_nonzero
            W_effective = np.where(release_mask, W_res, 0.0)
            # Scale weights to compensate for dropout (inverted dropout)
            # This keeps the expected input magnitude constant
            W_effective = W_effective / max(p_release, 1e-6)
        else:
            W_effective = W_res

        pre = W_effective @ x + W_in * input_signal[t]
        x = (1 - LEAK_RATE) * x + LEAK_RATE * np.tanh(pre)
        states[t] = x

    return states


# ══════════════════════════════════════════════════
# MEMORY AND NONLINEAR CAPACITY
# ══════════════════════════════════════════════════

def compute_memory_capacity(X, u, max_delay=MAX_DELAY):
    """Compute MC = Σ R²(k) for k=1..K via Ridge regression."""
    n = len(u)
    mc_per_delay = np.zeros(max_delay)

    for k in range(1, max_delay + 1):
        target = u[:-k] if k < n else np.zeros(n)
        states = X[k:]
        if len(target) < 10 or len(states) < 10:
            continue
        min_len = min(len(target), len(states))
        target, states = target[:min_len], states[:min_len]
        n_tr = int(min_len * TRAIN_FRAC)
        if n_tr < 5:
            continue

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(states[:n_tr], target[:n_tr])
        r2 = r2_score(target[n_tr:], model.predict(states[n_tr:]))
        mc_per_delay[k - 1] = max(0.0, r2)

    return float(np.sum(mc_per_delay)), mc_per_delay


def compute_nonlinear_capacity(X, u):
    """Compute NLC for u²(t) and u(t)·u(t-1)."""
    n = len(u)
    n_tr = int(n * TRAIN_FRAC)
    nlc = {}

    # u²(t)
    target_sq = u ** 2
    min_len = min(len(target_sq), len(X))
    if min_len - n_tr > 5:
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X[:n_tr], target_sq[:n_tr])
        nlc['square_r2'] = max(0.0, r2_score(target_sq[n_tr:min_len],
                                              model.predict(X[n_tr:min_len])))
    else:
        nlc['square_r2'] = 0.0

    # u(t)·u(t-1)
    target_cross = u[1:] * u[:-1]
    X_s = X[1:]
    min_len = min(len(target_cross), len(X_s))
    if min_len - n_tr > 5:
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_s[:n_tr], target_cross[:n_tr])
        nlc['cross_r2'] = max(0.0, r2_score(target_cross[n_tr:min_len],
                                             model.predict(X_s[n_tr:min_len])))
    else:
        nlc['cross_r2'] = 0.0

    return nlc


# ══════════════════════════════════════════════════
# ENAQT → P_RELEASE BRIDGE
# ══════════════════════════════════════════════════

def compute_p_release(gamma_cm):
    """
    Compute synaptic release probability from ENAQT transport efficiency.

    Calls the Experiment 1c quantum transport model to get P₄(γ).

    Parameters
    ----------
    gamma_cm : float
        Dephasing rate in cm⁻¹

    Returns
    -------
    p_release : float
        Transport efficiency = P₄ = synaptic release probability
    """
    eta, _, _, _ = compute_transport_efficiency(gamma_cm)
    return eta


# ══════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_1e_results.png")
    output_json = os.path.join(script_dir, "experiment_1e_metrics.json")

    print("=" * 65)
    print("  EXPERIMENT 1e — ENAQT-MODULATED SYNAPTIC RESERVOIR COMPUTING")
    print("  Biological Computing Research Programme · Phase 1")
    print("=" * 65)

    t_start = time.time()
    rng = np.random.default_rng(SEED)

    # ── Create reservoir (same as 1d) ──
    W_res = create_reservoir_weights(N_RESERVOIR, P_CONNECT,
                                      SPECTRAL_RADIUS, rng)
    W_in = rng.uniform(-W_INPUT_SCALE, W_INPUT_SCALE, N_RESERVOIR)
    input_signal = rng.uniform(-0.5, 0.5, N_STEPS)

    actual_sr = np.max(np.abs(np.linalg.eigvals(W_res)))
    print(f"\n  Reservoir: {N_RESERVOIR} neurons, "
          f"p={P_CONNECT}, ρ={actual_sr:.3f}")
    print(f"  Leak rate: {LEAK_RATE}, steps: {N_STEPS}")

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE ENAQT CURVE — γ → P₄
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing ENAQT transport efficiency curve")
    print(f"{'─' * 65}")

    gamma_sweep = np.logspace(np.log10(GAMMA_MIN_CM),
                               np.log10(GAMMA_MAX_CM),
                               N_DEPHASING_SWEEP)
    p4_sweep = []
    for i, gamma in enumerate(gamma_sweep):
        p4 = compute_p_release(gamma)
        p4_sweep.append(p4)
        if i % 10 == 0:
            print(f"    γ = {gamma:10.2f} cm⁻¹  →  P₄ = {p4:.4f}")
    p4_sweep = np.array(p4_sweep)

    # Key regimes
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
        ("Coherent",       GAMMA_COHERENT_CM,  p_coherent),
        ("Body temp",      GAMMA_BODY_CM,      p_body),
        ("ENAQT peak",     GAMMA_PEAK_CM,      p_peak),
        ("Full (P=1.0)",   None,               1.0),
    ]

    results = {}
    for label, gamma, p_rel in conditions:
        print(f"\n{'─' * 65}")
        gamma_str = f"γ={gamma:.1f}" if gamma else "N/A"
        print(f"  Condition: {label} ({gamma_str}, "
              f"P_release={p_rel:.4f})")
        print(f"{'─' * 65}")

        t0 = time.time()
        states = run_esn_with_release(
            input_signal, W_res, W_in,
            p_release=p_rel,
            rng=np.random.default_rng(SEED + 1)
        )
        elapsed = time.time() - t0

        X = states[N_WASHOUT:]
        u = input_signal[N_WASHOUT:]
        mc_total, mc_per_delay = compute_memory_capacity(X, u)
        nlc = compute_nonlinear_capacity(X, u)

        results[label] = {
            'gamma_cm': gamma,
            'p_release': p_rel,
            'states': states, 'X': X, 'u': u,
            'mc_total': mc_total,
            'mc_per_delay': mc_per_delay,
            'nlc': nlc,
            'mean_act': float(np.mean(np.abs(X))),
            'std_act': float(np.std(X)),
            'elapsed_s': elapsed,
        }

        print(f"  ◎ Memory capacity: MC = {mc_total:.2f}")
        print(f"  ◎ NLC(u²) = {nlc['square_r2']:.3f}  "
              f"NLC(u·u₋₁) = {nlc['cross_r2']:.3f}")
        print(f"  ◎ |act| = {results[label]['mean_act']:.4f}")
        print(f"  ◎ Elapsed: {elapsed:.2f}s")

    # ═══════════════════════════════════════════════
    # STEP 3: DEPHASING SWEEP — γ → P₄ → MC
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: Dephasing sweep — MC(γ) via P₄(γ)")
    print(f"{'─' * 65}")

    mc_sweep = []
    for i, (gamma, p4) in enumerate(zip(gamma_sweep, p4_sweep)):
        states = run_esn_with_release(
            input_signal, W_res, W_in,
            p_release=max(p4, 0.001),  # clamp to avoid zero connectivity
            rng=np.random.default_rng(SEED + 2)
        )
        X = states[N_WASHOUT:]
        mc, _ = compute_memory_capacity(X, input_signal[N_WASHOUT:])
        mc_sweep.append(mc)
        if i % 10 == 0:
            print(f"    γ={gamma:10.2f}  P₄={p4:.4f}  MC={mc:.2f}")

    mc_sweep = np.array(mc_sweep)

    # Find MC peak
    peak_idx = np.argmax(mc_sweep)
    mc_peak_gamma = gamma_sweep[peak_idx]
    mc_peak_val = mc_sweep[peak_idx]

    # Find ENAQT peak
    enaqt_peak_idx = np.argmax(p4_sweep)
    enaqt_peak_gamma = gamma_sweep[enaqt_peak_idx]

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed_total = time.time() - t_start
    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")

    for label in ["Coherent", "Body temp", "ENAQT peak", "Full (P=1.0)"]:
        r = results[label]
        print(f"  {label:16s}  P_rel={r['p_release']:.3f}  "
              f"MC={r['mc_total']:5.2f}  "
              f"NLC_sq={r['nlc']['square_r2']:.3f}")

    print(f"\n  MC peak:    γ = {mc_peak_gamma:.1f} cm⁻¹  "
          f"(MC = {mc_peak_val:.2f})")
    print(f"  ENAQT peak: γ = {enaqt_peak_gamma:.1f} cm⁻¹  "
          f"(P₄ = {p4_sweep[enaqt_peak_idx]:.3f})")
    print(f"  Correlation: MC peak {'≈' if abs(np.log10(mc_peak_gamma/enaqt_peak_gamma)) < 0.5 else '≠'} ENAQT peak")
    print(f"  Total elapsed: {elapsed_total:.1f}s")
    print(f"{'═' * 65}")

    # ── Save metrics ──
    metrics = {
        'experiment': '1e_enaqt_reservoir',
        'parameters': {
            'n_reservoir': N_RESERVOIR,
            'spectral_radius': SPECTRAL_RADIUS,
            'w_input_scale': W_INPUT_SCALE,
            'leak_rate': LEAK_RATE,
            'n_steps': N_STEPS,
            'max_delay': MAX_DELAY,
            'seed': SEED,
        },
        'enaqt_bridge': {
            'p_coherent': round(p_coherent, 6),
            'p_body': round(p_body, 6),
            'p_peak': round(p_peak, 6),
            'gamma_coherent_cm': GAMMA_COHERENT_CM,
            'gamma_body_cm': GAMMA_BODY_CM,
            'gamma_peak_cm': GAMMA_PEAK_CM,
        },
        'conditions': {},
        'dephasing_sweep': {
            'gamma_values': [round(g, 4) for g in gamma_sweep.tolist()],
            'p4_values': [round(p, 6) for p in p4_sweep.tolist()],
            'mc_values': [round(m, 4) for m in mc_sweep.tolist()],
            'mc_peak_gamma_cm': round(mc_peak_gamma, 4),
            'mc_peak_value': round(mc_peak_val, 4),
            'enaqt_peak_gamma_cm': round(enaqt_peak_gamma, 4),
        },
        'elapsed_s': round(elapsed_total, 1),
    }

    for label in ["Coherent", "Body temp", "ENAQT peak", "Full (P=1.0)"]:
        r = results[label]
        metrics['conditions'][label] = {
            'p_release': round(r['p_release'], 6),
            'mc_total': round(r['mc_total'], 4),
            'mc_per_delay': [round(x, 4) for x in r['mc_per_delay']],
            'nlc_square_r2': round(r['nlc']['square_r2'], 4),
            'nlc_cross_r2': round(r['nlc']['cross_r2'], 4),
            'mean_activation': round(r['mean_act'], 4),
        }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Generate dashboard ──
    make_dashboard(results, gamma_sweep, p4_sweep, mc_sweep,
                   mc_peak_gamma, mc_peak_val, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 1e COMPLETE")
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
    ACCENT_PURPLE = "#a78bfa"
    TEXT_PRIMARY = (1, 1, 1, 0.95)
    TEXT_SECONDARY = (1, 1, 1, 0.55)
    TEXT_MUTED = (1, 1, 1, 0.30)
    BORDER = (0, 0.824, 1.0, 0.08)
    MONO = "monospace"

    cond_colors = {
        "Coherent": ACCENT_PURPLE,
        "Body temp": ACCENT_GREEN,
        "ENAQT peak": ACCENT_CYAN,
        "Full (P=1.0)": ACCENT_AMBER,
    }

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 2, hspace=0.40, wspace=0.28,
                  height_ratios=[1.0, 1.2, 1.2])

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

    # ── Panel A: ENAQT curve η(γ) ──
    ax_enaqt = fig.add_subplot(gs[0, 0])
    style_ax(ax_enaqt, "◈  ENAQT TRANSPORT EFFICIENCY P₄(γ)")

    ax_enaqt.semilogx(gamma_sweep, p4_sweep, color=ACCENT_CYAN,
                       linewidth=2.5, alpha=0.9)
    # Mark key regimes
    for label, gamma, p_rel in [("Coherent", GAMMA_COHERENT_CM,
                                  results["Coherent"]['p_release']),
                                 ("Body", GAMMA_BODY_CM,
                                  results["Body temp"]['p_release']),
                                 ("Peak", GAMMA_PEAK_CM,
                                  results["ENAQT peak"]['p_release'])]:
        color = {"Coherent": ACCENT_PURPLE, "Body": ACCENT_GREEN,
                 "Peak": ACCENT_CYAN}[label]
        ax_enaqt.axvline(gamma, color=color, alpha=0.4, linestyle="--",
                          linewidth=1)
        ax_enaqt.scatter([gamma], [p_rel], color=color, s=80, zorder=5,
                          edgecolors='white', linewidth=0.5)
        ax_enaqt.annotate(f"{label}\nP₄={p_rel:.3f}",
                           (gamma, p_rel), textcoords="offset points",
                           xytext=(15, 5), fontsize=7, color=color,
                           fontfamily=MONO)

    ax_enaqt.set_xlabel("Dephasing rate γ (cm⁻¹)", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=9)
    ax_enaqt.set_ylabel("Transport efficiency P₄", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=9)

    # ── Panel B: MC(γ) sweep ──
    ax_mc_sweep = fig.add_subplot(gs[0, 1])
    style_ax(ax_mc_sweep, "◈  MEMORY CAPACITY MC(γ) VIA ENAQT")

    ax_mc_sweep.semilogx(gamma_sweep, mc_sweep, color=ACCENT_TEAL,
                          linewidth=2.5, alpha=0.9)
    ax_mc_sweep.axvline(mc_peak_gamma, color=ACCENT_RED, alpha=0.6,
                         linestyle="--", linewidth=1.5)
    ax_mc_sweep.scatter([mc_peak_gamma], [mc_peak_val], color=ACCENT_RED,
                         s=100, zorder=5, edgecolors='white', linewidth=0.5)
    ax_mc_sweep.annotate(f"MC peak\nγ={mc_peak_gamma:.0f}\nMC={mc_peak_val:.2f}",
                          (mc_peak_gamma, mc_peak_val),
                          textcoords="offset points", xytext=(15, -15),
                          fontsize=8, color=ACCENT_RED, fontfamily=MONO)
    # Mark body temp
    ax_mc_sweep.axvline(GAMMA_BODY_CM, color=ACCENT_GREEN, alpha=0.4,
                         linestyle=":", linewidth=1)
    ax_mc_sweep.set_xlabel("Dephasing rate γ (cm⁻¹)", color=TEXT_SECONDARY,
                            fontfamily=MONO, fontsize=9)
    ax_mc_sweep.set_ylabel("Memory capacity MC", color=TEXT_SECONDARY,
                            fontfamily=MONO, fontsize=9)

    # ── Panel C: OVERLAY — η(γ) vs MC(γ) ──
    ax_overlay = fig.add_subplot(gs[1, 0])
    style_ax(ax_overlay, "◈  DUAL AXIS — ENAQT vs MEMORY CAPACITY")

    ax_overlay.semilogx(gamma_sweep, p4_sweep, color=ACCENT_CYAN,
                         linewidth=2, alpha=0.8, label="P₄ (ENAQT)")
    ax_overlay.set_ylabel("Transport efficiency P₄", color=ACCENT_CYAN,
                           fontfamily=MONO, fontsize=9)
    ax_overlay.set_xlabel("Dephasing rate γ (cm⁻¹)", color=TEXT_SECONDARY,
                           fontfamily=MONO, fontsize=9)
    ax_overlay.tick_params(axis='y', labelcolor=ACCENT_CYAN)

    ax2 = ax_overlay.twinx()
    ax2.semilogx(gamma_sweep, mc_sweep, color=ACCENT_TEAL,
                  linewidth=2, alpha=0.8, linestyle="--", label="MC")
    ax2.set_ylabel("Memory capacity MC", color=ACCENT_TEAL,
                    fontfamily=MONO, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=ACCENT_TEAL)
    ax2.spines["right"].set_color(BORDER)
    ax2.spines["top"].set_visible(False)
    ax2.set_facecolor("none")

    # Combined legend
    lines1, labels1 = ax_overlay.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_overlay.legend(lines1 + lines2, labels1 + labels2,
                       fontsize=8, facecolor=BG_SECONDARY,
                       edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                       loc="upper right")

    # ── Panel D: MC(k) per-delay curves ──
    ax_mc = fig.add_subplot(gs[1, 1])
    style_ax(ax_mc, "◈  MEMORY CAPACITY MC(k) — PER CONDITION")

    delays = np.arange(1, MAX_DELAY + 1)
    for label in ["Coherent", "Body temp", "ENAQT peak", "Full (P=1.0)"]:
        mc_k = results[label]['mc_per_delay']
        color = cond_colors[label]
        ax_mc.plot(delays, mc_k, color=color, linewidth=2, alpha=0.85,
                   marker='o', markersize=3,
                   label=f"{label} (MC={results[label]['mc_total']:.2f})")
    ax_mc.set_xlabel("Delay k (steps)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylabel("R²(k)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylim(-0.02, 1.02)
    ax_mc.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY, loc="upper right")

    # ── Panel E: NLC bar chart ──
    ax_nlc = fig.add_subplot(gs[2, 0])
    style_ax(ax_nlc, "◈  NONLINEAR CAPACITY")

    labels_cond = ["Coherent", "Body temp", "ENAQT peak", "Full (P=1.0)"]
    sq_vals = [results[l]['nlc']['square_r2'] for l in labels_cond]
    cross_vals = [results[l]['nlc']['cross_r2'] for l in labels_cond]

    x = np.arange(len(labels_cond))
    width = 0.35
    bars_sq = ax_nlc.bar(x - width / 2, sq_vals, width, alpha=0.8,
                         label="R²(u²)", color=ACCENT_GREEN)
    bars_cr = ax_nlc.bar(x + width / 2, cross_vals, width, alpha=0.8,
                         label="R²(u·u₋₁)", color=ACCENT_AMBER)

    ax_nlc.set_xticks(x)
    ax_nlc.set_xticklabels([l.replace(" ", "\n") for l in labels_cond],
                           fontsize=8, color=TEXT_SECONDARY, fontfamily=MONO)
    ax_nlc.set_ylabel("R²", color=TEXT_SECONDARY, fontfamily=MONO, fontsize=9)
    ax_nlc.legend(fontsize=8, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    for bars in [bars_sq, bars_cr]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                ax_nlc.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom",
                            color=TEXT_SECONDARY, fontsize=7, fontfamily=MONO)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  ENAQT → RESERVOIR — BRIDGE RESULTS")
    ax_sum.axis("off")

    lines = [
        f"RESERVOIR:  {N_RESERVOIR} neurons, ρ={SPECTRAL_RADIUS}",
        f"SYNAPTIC RELEASE = P₄(γ) FROM ENAQT MODEL",
        "",
    ]
    for label in ["Coherent", "Body temp", "ENAQT peak", "Full (P=1.0)"]:
        r = results[label]
        lines.append(
            f"{label:16s}  P={r['p_release']:.3f}  "
            f"MC={r['mc_total']:5.2f}  "
            f"NLC={r['nlc']['square_r2']:.3f}"
        )
    lines += [
        "",
        f"MC PEAK:    γ = {mc_peak_gamma:.0f} cm⁻¹  (MC = {mc_peak_val:.2f})",
        f"ENAQT PEAK: γ = {gamma_sweep[np.argmax(p4_sweep)]:.0f} cm⁻¹  "
        f"(P₄ = {np.max(p4_sweep):.3f})",
        "",
        f"FINDING: MC peak {'COINCIDES WITH' if abs(np.log10(mc_peak_gamma/gamma_sweep[np.argmax(p4_sweep)])) < 0.5 else 'DIFFERS FROM'} ENAQT peak",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 1e ── ENAQT-MODULATED SYNAPTIC RESERVOIR COMPUTING",
        color=ACCENT_CYAN, fontsize=15, fontfamily=MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             "P₄(γ) from quantum ion channel → synaptic release → "
             "reservoir memory capacity",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
