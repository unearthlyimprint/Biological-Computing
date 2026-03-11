#!/usr/bin/env python3
"""
experiment_1c_enaqt_ion_channel.py — ENAQT in a Model Ion Channel
==================================================================

Phase 1, Experiment 1c of the Biological Computing research programme.

Simulates Environment-Assisted Quantum Transport (ENAQT) through a
4-site tight-binding model of a K⁺ channel selectivity filter. Tests
whether noise at biological temperatures can *enhance* quantum transport
efficiency — the same phenomenon proven in FMO photosynthesis.

Model:
  - 4 binding sites in the selectivity filter (single-file K⁺ passage)
  - Tight-binding Hamiltonian with nearest-neighbour hopping J ≈ 25 meV
  - Pure dephasing noise (Lindblad) parameterised by rate Γ
  - Irreversible sink at site 4 (ion exits channel)
  - Site energy disorder Δ (thermal fluctuations)

Key physics (Plenio & Huelga 2008, Mohseni et al. 2008):
  - Too little noise → Anderson localisation traps the ion
  - Too much noise → quantum Zeno effect freezes transport
  - Optimal noise → ENAQT peak — constructive quantum-classical interplay

Outputs:
  - experiment_1c_results.png   (5-panel dashboard)
  - experiment_1c_metrics.json  (transport efficiencies and parameters)

References:
  Plenio, M.B. & Huelga, S.F. (2008). "Dephasing-assisted transport:
    quantum and classical." New J. Phys. 10, 113019.
  Mohseni, M. et al. (2008). "Environment-assisted quantum walks in
    photosynthetic energy transfer." J. Chem. Phys. 129, 174106.
  Doyle, D.A. et al. (1998). "The structure of the potassium channel."
    Science 280, 69–77.

Usage:
    python experiment_1c_enaqt_ion_channel.py
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import qutip as qt


# ══════════════════════════════════════════════════
# PHYSICS CONSTANTS & MODEL PARAMETERS
# ══════════════════════════════════════════════════

# Energy units: cm⁻¹ (spectroscopy convention, 1 meV ≈ 8.066 cm⁻¹)
# Time units: picoseconds (natural for these energy scales)

K_B_CM = 0.695035       # Boltzmann constant in cm⁻¹/K
HBAR_CM_PS = 5.3088     # ℏ in cm⁻¹ · ps

# K⁺ channel selectivity filter (Doyle et al. 1998)
N_SITES = 4             # 4 binding sites in selectivity filter
SITE_SPACING_A = 3.4    # Å between binding sites
FILTER_LENGTH_A = 12.0  # total filter length

# Tight-binding parameters
J_HOPPING_CM = 100.0    # nearest-neighbour hopping ≈ 12.4 meV ≈ 100 cm⁻¹
SINK_RATE_CM = 5.0      # sink rate κ at site 4 (weak extraction)

# DONOR–BRIDGE–ACCEPTOR energy landscape.
# The selectivity filter confines K⁺ ions in single file with binding sites
# at different electrostatic depths. The carbonyl oxygens at intermediate
# sites create a barrier that K⁺ must cross.
#
# Canonical ENAQT requires:
#   - Donor and acceptor near-resonant (or acceptor lower)
#   - Bridge sites at HIGHER energy (creates tunnelling barrier)
#   - Coherent transport is slow (off-resonance tunnelling)
#   - Dephasing breaks phase coherence → enables sequential hopping
#
# Model:  ε = [0, +1500, +1500, 0] cm⁻¹
# Bridge barrier = 1500 cm⁻¹ (= 15 J, deep off-resonance localisation)
# At this barrier, coherent tunnelling is exponentially suppressed,
# requiring dephasing to enable classical hopping → canonical ENAQT.
BRIDGE_HEIGHT_CM = 1500.0
SITE_ENERGIES_CM = np.array([0.0, BRIDGE_HEIGHT_CM, BRIDGE_HEIGHT_CM, 0.0])

# Measurement time: probe at intermediate time.
# ℏ/J = 5.3 ps / 100 ≈ 0.053 ps. Use 5 ps (≈100 × ℏ/J).
# Long enough for dephasing-assisted hopping, short enough that
# coherent tunnelling through the large barrier is still incomplete.
TRANSIT_TIME_PS = 5.0    # measurement window
N_TPOINTS = 500

# Dephasing sweep
N_DEPHASING = 120
DEPHASING_MIN_CM = 0.01
DEPHASING_MAX_CM = 1e5

# Temperature markers
TEMPERATURES_K = [4, 77, 293, 310]
TEMP_LABELS = ["4 K (cryo)", "77 K (LN₂)", "293 K (room)", "310 K (body)"]

# Barrier heights for comparison sweep (cm⁻¹)
# Replaces the old "disorder" sweep — now compares different barrier strengths
BARRIER_VALUES = [0, 250, 500, 1000, 1500]
N_DISORDER_SAMPLES = 1  # no averaging needed (deterministic)

# Map temperature to dephasing rate: Γ(T) = α·k_B·T
# α ≈ 1 gives Γ(310K) ≈ 215 cm⁻¹ (reasonable for protein environment)
ALPHA_DEPHASING = 1.0


# ══════════════════════════════════════════════════
# HAMILTONIAN CONSTRUCTION
# ══════════════════════════════════════════════════

def build_hamiltonian(site_energies=None):
    """
    Build 4-site tight-binding Hamiltonian for the selectivity filter.

    H = Σᵢ εᵢ|i⟩⟨i| + Σ⟨i,j⟩ J(|i⟩⟨j| + |j⟩⟨i|)

    Parameters
    ----------
    site_energies : array-like or None
        Site energies εᵢ in cm⁻¹. If None, use the default gradient.

    Returns
    -------
    H : Qobj
        Hamiltonian in cm⁻¹ (multiply by 2π/ℏ for angular frequency)
    """
    if site_energies is None:
        site_energies = SITE_ENERGIES_CM.copy()

    # Site energies (diagonal)
    H = sum(site_energies[i] * qt.projection(N_SITES, i, i)
            for i in range(N_SITES))

    # Nearest-neighbour hopping (off-diagonal)
    for i in range(N_SITES - 1):
        H += J_HOPPING_CM * (qt.projection(N_SITES, i, i + 1) +
                              qt.projection(N_SITES, i + 1, i))

    return H


# ══════════════════════════════════════════════════
# LINDBLAD OPERATORS
# ══════════════════════════════════════════════════

def build_lindblad_ops(gamma_cm, kappa_cm=SINK_RATE_CM):
    """
    Build Lindblad collapse operators for dephasing and sink.

    Parameters
    ----------
    gamma_cm : float
        Pure dephasing rate Γ in cm⁻¹
    kappa_cm : float
        Sink rate κ at the exit site (site N-1) in cm⁻¹

    Returns
    -------
    c_ops : list of Qobj
    """
    c_ops = []

    # Convert rates from cm⁻¹ to ps⁻¹:  rate_ps = rate_cm / ℏ_cm_ps
    gamma_ps = gamma_cm / HBAR_CM_PS
    kappa_ps = kappa_cm / HBAR_CM_PS

    # Pure dephasing on each site: √Γ |i⟩⟨i|
    if gamma_ps > 0:
        for i in range(N_SITES):
            c_ops.append(np.sqrt(gamma_ps) * qt.projection(N_SITES, i, i))

    # Irreversible sink at last site: √κ |sink⟩⟨N-1|
    # We model the sink by removing population from site N-1
    # using a lowering-type operator to a "ground" state (site 0 re-used
    # is wrong — instead we track population loss as transport efficiency)
    # Correct approach: add an extra "sink" state, OR track population
    # loss via a non-Hermitian decay operator.
    # Here: use the trace-loss approach: c_sink = √κ |0⟩⟨N-1| where we
    # track Tr(ρ) decrease as successful transport.
    # Better: add a 5th state as sink.

    return c_ops, kappa_ps


def build_system_with_sink(site_energies=None, gamma_cm=0.0):
    """
    Build the full system with N_SITES + 1 states (extra sink state).

    The sink state |sink⟩ is the (N_SITES)th basis state.
    Transport is measured as population in |sink⟩.
    """
    n = N_SITES + 1  # 4 sites + 1 sink

    # Site energies on the physical sites — USE the barrier landscape by default
    if site_energies is None:
        site_energies = SITE_ENERGIES_CM.copy()

    # Hamiltonian: only acts on physical sites
    H = sum(site_energies[i] * qt.projection(n, i, i)
            for i in range(N_SITES))
    for i in range(N_SITES - 1):
        H += J_HOPPING_CM * (qt.projection(n, i, i + 1) +
                              qt.projection(n, i + 1, i))

    # Lindblad operators
    c_ops = []

    # Convert rates
    gamma_ps = gamma_cm / HBAR_CM_PS
    kappa_ps = SINK_RATE_CM / HBAR_CM_PS

    # Pure dephasing on physical sites
    if gamma_ps > 0:
        for i in range(N_SITES):
            c_ops.append(np.sqrt(gamma_ps) * qt.projection(n, i, i))

    # Sink: irreversible transfer from site (N_SITES-1) to sink state
    c_ops.append(np.sqrt(kappa_ps) * qt.projection(n, N_SITES, N_SITES - 1))

    # Initial state: ion enters at site 0
    rho_0 = qt.projection(n, 0, 0)

    # Observable: population in sink state (= transport efficiency)
    P_sink = qt.projection(n, N_SITES, N_SITES)

    return H, c_ops, rho_0, P_sink


# ══════════════════════════════════════════════════
# TRANSPORT EFFICIENCY
# ══════════════════════════════════════════════════

def compute_transport_efficiency(gamma_cm, site_energies=None,
                                  tmax=TRANSIT_TIME_PS,
                                  n_points=N_TPOINTS):
    """
    Run Lindblad master equation and return transport efficiency.

    Transport efficiency η = population in sink state at time tmax.

    Returns
    -------
    eta : float
        Transport efficiency (0 to 1)
    times : ndarray
        Time points (ps)
    sink_pop : ndarray
        Sink population vs time
    site_pops : ndarray (n_points × N_SITES)
        Population on each physical site vs time
    """
    n = N_SITES + 1
    H, c_ops, rho_0, P_sink = build_system_with_sink(
        site_energies=site_energies, gamma_cm=gamma_cm)

    tlist = np.linspace(0, tmax, n_points)

    # Build site population observables
    P_sites = [qt.projection(n, i, i) for i in range(N_SITES)]

    e_ops = [P_sink] + P_sites

    # Solve
    opts = {"atol": 1e-10, "rtol": 1e-8}
    try:
        result = qt.mesolve(H / HBAR_CM_PS, rho_0, tlist, c_ops, e_ops,
                            options=opts)
    except TypeError:
        result = qt.mesolve(H / HBAR_CM_PS, rho_0, tlist, c_ops, e_ops,
                            options=qt.Options(atol=1e-10, rtol=1e-8))

    sink_pop = np.real(result.expect[0])
    site_pops = np.array([np.real(result.expect[i + 1])
                          for i in range(N_SITES)])

    eta = float(sink_pop[-1])
    return eta, tlist, sink_pop, site_pops


# ══════════════════════════════════════════════════
# DISORDER AVERAGING
# ══════════════════════════════════════════════════

def compute_efficiency_with_disorder(gamma_cm, disorder_cm,
                                      n_samples=N_DISORDER_SAMPLES,
                                      rng=None):
    """
    Average transport efficiency over random site-energy disorder.

    Parameters
    ----------
    gamma_cm : float
        Dephasing rate (cm⁻¹)
    disorder_cm : float
        Disorder width Δ — site energies drawn from Uniform[-Δ/2, Δ/2]
    n_samples : int
        Number of disorder realisations

    Returns
    -------
    eta_mean : float
    eta_std : float
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if disorder_cm == 0:
        eta, _, _, _ = compute_transport_efficiency(
            gamma_cm, site_energies=SITE_ENERGIES_CM)
        return eta, 0.0

    etas = []
    for _ in range(n_samples):
        # Add disorder on top of the baseline gradient
        noise = rng.uniform(-disorder_cm / 2, disorder_cm / 2,
                            size=N_SITES)
        energies = SITE_ENERGIES_CM + noise
        eta, _, _, _ = compute_transport_efficiency(
            gamma_cm, site_energies=energies)
        etas.append(eta)

    return float(np.mean(etas)), float(np.std(etas))


# ══════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_1c_results.png")
    output_json = os.path.join(script_dir, "experiment_1c_metrics.json")

    print("=" * 60)
    print("  EXPERIMENT 1c — ENAQT IN A MODEL ION CHANNEL")
    print("  Biological Computing Research Programme · Phase 1")
    print("=" * 60)

    t_start = time.time()

    # ── Model summary ──
    print(f"\n  Model: {N_SITES}-site tight-binding chain + sink")
    print(f"  Hopping J = {J_HOPPING_CM:.0f} cm⁻¹ "
          f"({J_HOPPING_CM / 8.066:.1f} meV)")
    print(f"  Sink rate κ = {SINK_RATE_CM:.0f} cm⁻¹")
    print(f"  Site energies: {SITE_ENERGIES_CM} cm⁻¹")
    print(f"  Measurement time = {TRANSIT_TIME_PS:.1f} ps")
    print(f"  Filter geometry: {N_SITES} sites × "
          f"{SITE_SPACING_A:.1f} Å = {FILTER_LENGTH_A:.0f} Å")

    # ═══════════════════════════════════════════════
    # SWEEP 1: ENAQT curve — η vs Γ (ordered lattice)
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep 1: Transport efficiency vs dephasing rate (Δ = 0)")
    print(f"{'─' * 60}")

    dephasing_rates = np.logspace(np.log10(DEPHASING_MIN_CM),
                                  np.log10(DEPHASING_MAX_CM),
                                  N_DEPHASING)
    efficiencies = []
    selected_curves = {}
    # Representative rates covering: coherent → dip → ENAQT peak → Zeno
    plot_rates = [0.1, 10.0, 100.0, 1000.0, 10000.0]

    for idx, gamma in enumerate(dephasing_rates):
        eta, tlist, sink_pop, site_pops = compute_transport_efficiency(gamma)
        efficiencies.append(eta)

        # Store selected time-evolution curves for plotting (one per target)
        nearest = min(plot_rates, key=lambda r: abs(np.log10(r) -
                                                     np.log10(gamma)))
        if (abs(np.log10(nearest) - np.log10(gamma)) < 0.08
                and f"Γ={nearest:.0e}" not in
                    ''.join(selected_curves.keys())):
            selected_curves[f"Γ = {nearest:.0e} cm⁻¹"] = {
                'times': tlist,
                'sink_pop': sink_pop,
                'site_pops': site_pops,
            }

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    [{idx+1:3d}/{N_DEPHASING}] "
                  f"Γ = {gamma:.2e} cm⁻¹  →  η = {eta:.4f}")

    efficiencies = np.array(efficiencies)

    # Find ENAQT peak
    peak_idx = np.argmax(efficiencies)
    optimal_gamma = dephasing_rates[peak_idx]
    max_eta = efficiencies[peak_idx]
    eta_no_noise = efficiencies[0]
    eta_high_noise = efficiencies[-1]

    print(f"\n  ◎ ENAQT peak: Γ_opt = {optimal_gamma:.1f} cm⁻¹, "
          f"η_max = {max_eta:.4f}")
    print(f"  ◎ No-noise limit: η(Γ→0) = {eta_no_noise:.4f}")
    print(f"  ◎ High-noise limit: η(Γ→∞) = {eta_high_noise:.4f}")
    print(f"  ◎ ENAQT enhancement: "
          f"{max_eta / max(eta_no_noise, 1e-10):.1f}× over coherent")

    # ═══════════════════════════════════════════════
    # SWEEP 2: Temperature mapping
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep 2: Mapping temperatures to dephasing rates")
    print(f"{'─' * 60}")

    temp_results = []
    for T_K, label in zip(TEMPERATURES_K, TEMP_LABELS):
        # Γ(T) = α · k_B · T (linear phonon coupling)
        gamma_T = ALPHA_DEPHASING * K_B_CM * T_K
        eta_T, _, _, _ = compute_transport_efficiency(gamma_T)
        print(f"    {label}: Γ = {gamma_T:.1f} cm⁻¹  →  η = {eta_T:.4f}")
        temp_results.append({
            'temperature_K': T_K,
            'label': label,
            'gamma_cm': round(gamma_T, 2),
            'efficiency': round(eta_T, 6),
        })

    # ═══════════════════════════════════════════════
    # SWEEP 3: Barrier height comparison
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep 3: Effect of bridge barrier height")
    print(f"{'─' * 60}")

    # Use fewer dephasing points for this sweep
    dephasing_sparse = np.logspace(np.log10(DEPHASING_MIN_CM),
                                   np.log10(DEPHASING_MAX_CM), 40)

    barrier_curves = {}
    for barrier in BARRIER_VALUES:
        energies = np.array([0.0, barrier, barrier, 0.0])
        print(f"    Barrier = {barrier} cm⁻¹ ... ", end="", flush=True)
        etas_b = []
        for gamma in dephasing_sparse:
            eta, _, _, _ = compute_transport_efficiency(
                gamma, site_energies=energies)
            etas_b.append(eta)
        barrier_curves[barrier] = np.array(etas_b)

        peak_b = np.argmax(etas_b)
        print(f"peak at Γ = {dephasing_sparse[peak_b]:.1f} cm⁻¹, "
              f"η = {etas_b[peak_b]:.4f}")

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 60}")

    body_result = [r for r in temp_results if r['temperature_K'] == 310][0]
    # Is body temperature near ENAQT peak?
    ratio = body_result['gamma_cm'] / optimal_gamma
    if 0.1 < ratio < 10:
        proximity = "NEAR the ENAQT peak"
    elif ratio <= 0.1:
        proximity = "BELOW the ENAQT peak (under-damped)"
    else:
        proximity = "ABOVE the ENAQT peak (over-damped)"

    print(f"  ◎ Optimal dephasing: Γ_opt = {optimal_gamma:.1f} cm⁻¹")
    print(f"  ◎ Body temp dephasing: Γ(310K) = "
          f"{body_result['gamma_cm']:.1f} cm⁻¹")
    print(f"  ◎ Body temperature is {proximity}")
    print(f"  ◎ Enhancement ratio: {ratio:.2f}× of Γ_opt")
    print(f"  ◎ Transport at 310 K: η = {body_result['efficiency']:.4f}")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 60}")

    # ── Save metrics ──
    metrics = {
        'experiment': '1c_enaqt_ion_channel',
        'model': {
            'n_sites': N_SITES,
            'hopping_cm': J_HOPPING_CM,
            'hopping_meV': round(J_HOPPING_CM / 8.066, 2),
            'sink_rate_cm': SINK_RATE_CM,
            'transit_time_ps': TRANSIT_TIME_PS,
            'site_spacing_A': SITE_SPACING_A,
            'filter_length_A': FILTER_LENGTH_A,
        },
        'enaqt_peak': {
            'optimal_dephasing_rate_cm': round(float(optimal_gamma), 2),
            'max_transport_efficiency': round(float(max_eta), 6),
            'coherent_limit_efficiency': round(float(eta_no_noise), 6),
            'classical_limit_efficiency': round(float(eta_high_noise), 6),
            'enhancement_ratio': round(float(max_eta / max(eta_no_noise,
                                                            1e-10)), 2),
        },
        'temperature_mapping': temp_results,
        'body_temperature_assessment': {
            'gamma_body_cm': round(body_result['gamma_cm'], 2),
            'efficiency_body': round(body_result['efficiency'], 6),
            'proximity_to_peak': proximity,
            'ratio_to_optimal': round(ratio, 3),
        },
        'barrier_comparison': {
            str(barrier): {
                'peak_gamma_cm': round(float(
                    dephasing_sparse[np.argmax(barrier_curves[barrier])]), 2),
                'peak_efficiency': round(float(
                    np.max(barrier_curves[barrier])), 6),
            }
            for barrier in BARRIER_VALUES
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Generate dashboard ──
    make_dashboard(
        dephasing_rates, efficiencies, selected_curves,
        temp_results, dephasing_sparse, barrier_curves,
        optimal_gamma, max_eta, body_result, output_img
    )

    print(f"\n{'═' * 60}")
    print("  EXPERIMENT 1c COMPLETE")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(dephasing_rates, efficiencies, selected_curves,
                   temp_results, dephasing_sparse, barrier_curves,
                   optimal_gamma, max_eta, body_result, output_path):
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
    ACCENT_PURPLE = "#a78bfa"
    TEXT_PRIMARY = (1, 1, 1, 0.95)
    TEXT_SECONDARY = (1, 1, 1, 0.55)
    TEXT_MUTED = (1, 1, 1, 0.30)
    BORDER = (0, 0.824, 1.0, 0.08)
    MONO = "monospace"

    colors_cycle = [ACCENT_CYAN, ACCENT_TEAL, ACCENT_GREEN,
                    ACCENT_AMBER, ACCENT_RED, ACCENT_PURPLE]

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 2, hspace=0.38, wspace=0.28,
                  height_ratios=[1.5, 1.5, 1.5])

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

    # ── Panel A: Transport time evolution ──
    ax_evo = fig.add_subplot(gs[0, 0])
    style_ax(ax_evo, "◈  SINK POPULATION η(t) AT SELECTED Γ")
    for idx, (label, data) in enumerate(sorted(
            selected_curves.items(),
            key=lambda x: float(x[0].split("=")[1].split()[0]))):
        color = colors_cycle[idx % len(colors_cycle)]
        ax_evo.plot(data['times'], data['sink_pop'],
                    color=color, linewidth=1.8, alpha=0.85, label=label)
    ax_evo.set_xlabel("Time (ps)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_evo.set_ylabel("Sink population η", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_evo.set_ylim(-0.02, 1.02)
    ax_evo.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                  loc="lower right")

    # ── Panel B: ENAQT curve (THE key result) ──
    ax_enaqt = fig.add_subplot(gs[0, 1])
    style_ax(ax_enaqt, "◈  ENAQT CURVE — TRANSPORT EFFICIENCY vs NOISE")
    ax_enaqt.semilogx(dephasing_rates, efficiencies,
                       color=ACCENT_CYAN, linewidth=2.5, alpha=0.9)
    # Mark the peak
    ax_enaqt.axvline(x=optimal_gamma, color=ACCENT_AMBER,
                      linestyle="--", alpha=0.6, linewidth=1)
    ax_enaqt.plot(optimal_gamma, max_eta, 'o', color=ACCENT_AMBER,
                   markersize=10, zorder=5)
    ax_enaqt.annotate(
        f"Γ_opt = {optimal_gamma:.0f} cm⁻¹\nη = {max_eta:.3f}",
        xy=(optimal_gamma, max_eta),
        xytext=(optimal_gamma * 5, max_eta - 0.08),
        color=ACCENT_AMBER, fontsize=8, fontfamily=MONO,
        arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=1),
    )
    # Mark body temperature
    gamma_body = body_result['gamma_cm']
    eta_body = body_result['efficiency']
    ax_enaqt.plot(gamma_body, eta_body, 's', color=ACCENT_RED,
                   markersize=10, zorder=5, label=f"310 K (body)")
    ax_enaqt.annotate(
        f"310 K\nΓ = {gamma_body:.0f}\nη = {eta_body:.3f}",
        xy=(gamma_body, eta_body),
        xytext=(gamma_body / 8, eta_body + 0.05),
        color=ACCENT_RED, fontsize=8, fontfamily=MONO,
        arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1),
    )
    ax_enaqt.set_xlabel("Dephasing rate Γ (cm⁻¹)", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=9)
    ax_enaqt.set_ylabel("Transport efficiency η", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=9)
    ax_enaqt.legend(fontsize=7, facecolor=BG_SECONDARY,
                     edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel C: Temperature markers on ENAQT curve ──
    ax_temp = fig.add_subplot(gs[1, 0])
    style_ax(ax_temp, "◈  TEMPERATURE MAPPING TO DEPHASING")
    ax_temp.semilogx(dephasing_rates, efficiencies,
                      color=ACCENT_CYAN, linewidth=1.5, alpha=0.4)
    markers = ['D', 's', '^', 'o']
    temp_colors = [ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
    for idx, res in enumerate(temp_results):
        ax_temp.plot(res['gamma_cm'], res['efficiency'],
                      marker=markers[idx], color=temp_colors[idx],
                      markersize=12, zorder=5, label=res['label'])
    ax_temp.axvline(x=optimal_gamma, color=ACCENT_AMBER,
                     linestyle=":", alpha=0.3, linewidth=1)
    ax_temp.set_xlabel("Dephasing rate Γ (cm⁻¹)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_temp.set_ylabel("Transport efficiency η", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_temp.legend(fontsize=8, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Barrier height comparison ──
    ax_dis = fig.add_subplot(gs[1, 1])
    style_ax(ax_dis, "◈  BRIDGE BARRIER HEIGHT EFFECT ON ENAQT")
    for idx, barrier in enumerate(BARRIER_VALUES):
        color = colors_cycle[idx % len(colors_cycle)]
        ls = "-" if barrier > 0 else "--"
        ax_dis.semilogx(dephasing_sparse, barrier_curves[barrier],
                         color=color, linewidth=1.8, alpha=0.85,
                         linestyle=ls, label=f"Barrier = {barrier} cm⁻¹")
    ax_dis.set_xlabel("Dephasing rate Γ (cm⁻¹)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_dis.set_ylabel("Transport efficiency η", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_dis.legend(fontsize=8, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Summary card ──
    ax_sum = fig.add_subplot(gs[2, :])
    style_ax(ax_sum, "◈  SELECTIVITY FILTER — ENAQT ASSESSMENT")
    ax_sum.axis("off")

    body_res = [r for r in temp_results if r['temperature_K'] == 310][0]
    cryo_res = [r for r in temp_results if r['temperature_K'] == 4][0]
    ratio = body_res['gamma_cm'] / optimal_gamma

    lines = [
        f"K⁺ SELECTIVITY FILTER:  {N_SITES} sites × "
        f"{SITE_SPACING_A} Å  =  {FILTER_LENGTH_A:.0f} Å",
        f"HOPPING INTEGRAL:  J = {J_HOPPING_CM:.0f} cm⁻¹  "
        f"({J_HOPPING_CM / 8.066:.1f} meV)",
        f"SINK RATE:  κ = {SINK_RATE_CM:.0f} cm⁻¹",
        "",
        f"ENAQT PEAK:  Γ_opt = {optimal_gamma:.0f} cm⁻¹  →  "
        f"η_max = {max_eta:.4f}",
        f"COHERENT LIMIT:  η(Γ→0) = {efficiencies[0]:.4f}",
        f"CLASSICAL LIMIT:  η(Γ→∞) = {efficiencies[-1]:.4f}",
        f"ENHANCEMENT:  {max_eta / max(efficiencies[0], 1e-10):.1f}× "
        f"over purely coherent transport",
        "",
        f"BODY TEMPERATURE (310 K):  Γ = {body_res['gamma_cm']:.0f} cm⁻¹  "
        f"→  η = {body_res['efficiency']:.4f}  "
        f"({ratio:.1f}× of Γ_opt)",
        f"CRYOGENIC (4 K):  Γ = {cryo_res['gamma_cm']:.1f} cm⁻¹  "
        f"→  η = {cryo_res['efficiency']:.4f}",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=10, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 1c ── ENAQT IN A MODEL K⁺ ION CHANNEL",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             f"{N_SITES}-site tight-binding · QuTiP Lindblad · "
             f"Plenio & Huelga (2008) model",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
