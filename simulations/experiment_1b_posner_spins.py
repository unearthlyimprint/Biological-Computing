#!/usr/bin/env python3
"""
experiment_1b_posner_spins.py — Posner Molecule ³¹P Nuclear Spin Dynamics
==========================================================================

Phase 1, Experiment 1b of the Biological Computing research programme.

Simulates 6 coupled ³¹P nuclear spins (spin-1/2) inside a Posner molecule
Ca₉(PO₄)₆ using QuTiP's Lindblad master equation. Tests Fisher's hypothesis
that nuclear spin coherence can survive at body temperature long enough to
influence calcium signalling.

Hilbert space: (2)⁶ = 64 dimensions (tractable on a laptop).

Outputs:
  - experiment_1b_results.png   (5-panel dashboard)
  - experiment_1b_metrics.json  (coherence lifetimes and parameters)

Reference:
  Fisher, M.P.A. (2015). "Quantum cognition: The possibility of processing
  with nuclear spins in the brain." Annals of Physics 362, 593–602.

Usage:
    python experiment_1b_posner_spins.py
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
# PHYSICS CONSTANTS
# ══════════════════════════════════════════════════

HBAR = 1.054571817e-34       # ℏ (J·s)
K_B = 1.380649e-23           # Boltzmann constant (J/K)
MU_0 = 1.2566370614e-6      # vacuum permeability (T·m/A)

# ³¹P nuclear spin properties
GAMMA_31P = 1.08394e8        # gyromagnetic ratio (rad·s⁻¹·T⁻¹)
SPIN_31P = 0.5               # spin quantum number

# Posner molecule geometry (Ca₉(PO₄)₆)
N_SPINS = 6                  # number of ³¹P nuclei
# P-P distances in Ångströms (from Posner cluster X-ray crystallography)
# Approximate: 3 pairs of nearest neighbours + 3 pairs of next-nearest
PP_DISTANCES_A = {
    (0, 1): 4.0, (2, 3): 4.0, (4, 5): 4.0,   # nearest pairs
    (0, 2): 5.2, (1, 3): 5.2, (0, 4): 5.5,    # next-nearest
    (1, 5): 5.5, (2, 4): 4.8, (3, 5): 4.8,    # next-nearest
}

# Earth's magnetic field
B_EARTH = 50e-6              # 50 µT

# Simulation time parameters
T_MAX_S = 0.1                # simulate up to 100 ms
N_TPOINTS = 500              # number of time points

# Temperature sweep
TEMPERATURES_K = [4, 77, 293, 310]
TEMP_LABELS = ["4 K (cryo)", "77 K (LN₂)", "293 K (room)", "310 K (body)"]

# Dephasing rate sweep (Hz) — from very slow to very fast
DEPHASING_RATES_HZ = np.logspace(1, 6, 12)  # 10 Hz → 10⁶ Hz


# ══════════════════════════════════════════════════
# HAMILTONIAN CONSTRUCTION
# ══════════════════════════════════════════════════

def spin_op(op_type, spin_idx, n_spins=N_SPINS):
    """
    Build a spin-1/2 operator acting on spin `spin_idx` in the
    full N-spin Hilbert space.

    Parameters
    ----------
    op_type : str
        'x', 'y', 'z', '+', '-'
    spin_idx : int
        Which spin (0 to n_spins-1)
    n_spins : int
        Total number of spins
    """
    op_map = {
        'x': qt.sigmax() / 2,
        'y': qt.sigmay() / 2,
        'z': qt.sigmaz() / 2,
        '+': qt.sigmap(),
        '-': qt.sigmam(),
    }
    op = op_map[op_type]
    # tensor product: identity on all spins except spin_idx
    op_list = [qt.qeye(2)] * n_spins
    op_list[spin_idx] = op
    return qt.tensor(op_list)


def build_zeeman_hamiltonian(B0=B_EARTH, gamma=GAMMA_31P):
    """
    Zeeman Hamiltonian: H_Z = -γ B₀ Σᵢ Iᵢᶻ

    In angular frequency units (rad/s).
    """
    omega_L = gamma * B0  # Larmor frequency (rad/s)
    # Build from tensor-product operators to maintain dims structure
    H_Z = sum(-omega_L * spin_op('z', i) for i in range(N_SPINS))
    return H_Z, omega_L


def dipolar_coupling_Hz(r_angstrom, gamma=GAMMA_31P):
    """
    Dipole-dipole coupling constant between two ³¹P spins.

    J_dd = (µ₀ γ² ℏ) / (4π r³)  [in rad/s]
    Convert to Hz for readability.
    """
    r_m = r_angstrom * 1e-10
    J_rad_s = (MU_0 * gamma**2 * HBAR) / (4 * np.pi * r_m**3)
    return J_rad_s / (2 * np.pi)  # convert to Hz


def build_dipolar_hamiltonian():
    """
    Secular dipole-dipole Hamiltonian (truncated to high-field form):
    H_dd = Σᵢ<ⱼ Jᵢⱼ [3Iᵢᶻ Iⱼᶻ − Iᵢ·Iⱼ]

    In a weak field like Earth's, the full dipolar Hamiltonian is more
    appropriate, but the secular approximation captures the essence.
    We use the isotropic (scalar) coupling as a simpler but physically
    representative model: H_dd = Σᵢ<ⱼ Jᵢⱼ Iᵢ·Iⱼ
    """
    coupling_info = []
    H_dd = None  # will be initialised from first coupling term

    for (i, j), r_A in PP_DISTANCES_A.items():
        J_Hz = dipolar_coupling_Hz(r_A)
        J_rad = J_Hz * 2 * np.pi
        # Scalar coupling: J * (Ix·Ix + Iy·Iy + Iz·Iz)
        term = J_rad * (
            spin_op('x', i) * spin_op('x', j) +
            spin_op('y', i) * spin_op('y', j) +
            spin_op('z', i) * spin_op('z', j)
        )
        H_dd = term if H_dd is None else H_dd + term
        coupling_info.append({
            'spins': (i, j),
            'distance_A': r_A,
            'coupling_Hz': round(J_Hz, 2),
        })

    return H_dd, coupling_info


def build_full_hamiltonian():
    """Build the total Hamiltonian: Zeeman + dipole-dipole."""
    H_Z, omega_L = build_zeeman_hamiltonian()
    H_dd, couplings = build_dipolar_hamiltonian()
    H_total = H_Z + H_dd
    return H_total, omega_L, couplings


# ══════════════════════════════════════════════════
# DECOHERENCE (Lindblad collapse operators)
# ══════════════════════════════════════════════════

def build_collapse_operators(gamma_1, gamma_2):
    """
    Build Lindblad collapse operators for T₁ relaxation and T₂ dephasing.

    Parameters
    ----------
    gamma_1 : float
        T₁ relaxation rate (1/T₁) in s⁻¹
    gamma_2 : float
        Pure dephasing rate in s⁻¹.
        Note: total T₂ rate = gamma_2 + gamma_1/2

    Returns
    -------
    c_ops : list of Qobj
        Collapse operators for mesolve
    """
    c_ops = []
    for i in range(N_SPINS):
        # T₁ relaxation: σ⁻ with rate √Γ₁
        if gamma_1 > 0:
            c_ops.append(np.sqrt(gamma_1) * spin_op('-', i))
        # Pure T₂ dephasing: σ_z with rate √(Γ₂/2)
        if gamma_2 > 0:
            c_ops.append(np.sqrt(gamma_2 / 2) * spin_op('z', i))
    return c_ops


# ══════════════════════════════════════════════════
# INITIAL STATE
# ══════════════════════════════════════════════════

def build_initial_state():
    """
    Create initial state: singlet entangled pair on spins 0-1,
    remaining spins in |↑⟩ (ground state).

    |ψ₀₁⟩ = (|↑↓⟩ − |↓↑⟩) / √2  (singlet)
    |ψ₂₋₅⟩ = |↑↑↑↑⟩
    """
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)

    # Singlet on spins 0, 1
    singlet = (qt.tensor(up, down) - qt.tensor(down, up)).unit()
    rho_01 = qt.ket2dm(singlet)

    # Ground state for remaining spins
    rho_rest = qt.ket2dm(qt.tensor([up] * (N_SPINS - 2)))

    # Full state
    rho_0 = qt.tensor(rho_01, rho_rest)
    return rho_0


def compute_concurrence_01(rho_full):
    """
    Compute concurrence of spins 0 and 1 by tracing out spins 2-5.

    Concurrence = 0 (separable) to 1 (maximally entangled).
    """
    # Partial trace over spins 2,3,4,5 (indices 2-5)
    rho_01 = rho_full.ptrace([0, 1])
    return qt.concurrence(rho_01)


# ══════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════

def run_coherence_decay(H, rho_0, gamma_1, gamma_2, tmax=T_MAX_S,
                        n_points=N_TPOINTS):
    """
    Run Lindblad master equation and track coherence + entanglement.

    We track TWO coherence measures:
    1. Purity of the 2-spin reduced state: Tr(ρ₀₁²)
       - 1.0 = pure state (maximum coherence)
       - 0.25 = maximally mixed (no coherence)
    2. Concurrence of spins 0-1 (entanglement)

    Returns
    -------
    times : ndarray (seconds)
    purity : ndarray (purity of 2-spin subsystem)
    concurrence_times : ndarray
    concurrence : ndarray
    """
    tlist = np.linspace(0, tmax, n_points)
    c_ops = build_collapse_operators(gamma_1, gamma_2)

    # No expectation operators needed — we compute from stored states
    opts = {"store_states": True, "atol": 1e-8, "rtol": 1e-6}
    try:
        result = qt.mesolve(H, rho_0, tlist, c_ops, [], options=opts)
    except TypeError:
        result = qt.mesolve(H, rho_0, tlist, c_ops, [],
                            options=qt.Options(atol=1e-8, rtol=1e-6,
                                               store_states=True))

    # Compute purity of 2-spin subsystem at every time step
    purity = np.zeros(n_points)
    for t_idx in range(n_points):
        rho_t = result.states[t_idx]
        rho_01 = rho_t.ptrace([0, 1])
        purity[t_idx] = (rho_01 * rho_01).tr().real

    # Compute concurrence at subsampled time points (expensive)
    subsample = max(1, n_points // 50)
    concurrence_times = tlist[::subsample]
    concurrence_vals = []
    for t_idx in range(0, n_points, subsample):
        rho_t = result.states[t_idx]
        try:
            c = compute_concurrence_01(rho_t)
            concurrence_vals.append(c)
        except Exception:
            concurrence_vals.append(0.0)

    return tlist, purity, concurrence_times, np.array(concurrence_vals)


def find_coherence_lifetime(times, values, threshold_frac=None):
    """Find time at which values drop below threshold.

    For purity: threshold = 0.25 + (initial - 0.25) / e
    (decays from ~1.0 toward 0.25 for maximally mixed 2-qubit state)
    For concurrence: threshold = initial / e
    """
    if len(values) == 0 or values[0] == 0:
        return 0.0
    if threshold_frac is not None:
        threshold = threshold_frac
    else:
        # Default: 1/e of the dynamic range above the floor
        floor = 0.25  # maximally mixed 2-qubit purity
        threshold = floor + (values[0] - floor) / np.e
    below = np.where(values < threshold)[0]
    if len(below) == 0:
        return times[-1]
    return float(times[below[0]])


# ══════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_1b_results.png")
    output_json = os.path.join(script_dir, "experiment_1b_metrics.json")

    print("=" * 60)
    print("  EXPERIMENT 1b — POSNER MOLECULE ³¹P SPIN DYNAMICS")
    print("  Biological Computing Research Programme · Phase 1")
    print("=" * 60)

    t_start = time.time()

    # ── Build Hamiltonian ──
    print("\n  Building 6-spin Hamiltonian (64×64)...")
    H, omega_L, couplings = build_full_hamiltonian()
    freq_L_Hz = omega_L / (2 * np.pi)
    print(f"  ³¹P Larmor frequency: {freq_L_Hz:.1f} Hz ({omega_L:.1f} rad/s)")
    print(f"  Earth field: {B_EARTH * 1e6:.0f} µT")
    print(f"  Dipolar couplings:")
    for c in couplings:
        print(f"    Spins {c['spins']}: {c['distance_A']:.1f} Å → "
              f"J = {c['coupling_Hz']:.1f} Hz")

    # ── Build initial state ──
    print("\n  Preparing singlet state on spins [0,1]...")
    rho_0 = build_initial_state()
    c0 = compute_concurrence_01(rho_0)
    print(f"  Initial concurrence: {c0:.4f}")

    # ═══════════════════════════════════════════════
    # SWEEP 1: Dephasing rates at body temperature
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep 1: Dephasing rate at T = 310 K")
    print(f"{'─' * 60}")

    # At 310 K, T₁ for ³¹P in solids: typically ~1–10 s
    # We use T₁ = 5 s (conservative for Ca₉(PO₄)₆ cage)
    T1_body = 5.0  # seconds
    gamma_1_body = 1.0 / T1_body

    dephasing_results = []
    selected_curves = {}  # store a few curves for plotting

    plot_rates = [1e2, 1e3, 1e4, 1e5]  # rates to plot curves for

    for gamma_2 in DEPHASING_RATES_HZ:
        gamma_2_rad = gamma_2 * 2 * np.pi  # convert Hz to rad/s

        print(f"    Γ₂ = {gamma_2:.0e} Hz ... ", end="", flush=True)

        # Adapt tmax based on expected coherence time
        est_tau = 1.0 / (gamma_2_rad + gamma_1_body)
        tmax = min(T_MAX_S, max(est_tau * 5, 1e-4))

        tlist, purity, conc_t, concurrence = run_coherence_decay(
            H, rho_0, gamma_1_body, gamma_2_rad, tmax=tmax
        )
        tau_c = find_coherence_lifetime(tlist, purity)
        tau_ent = find_coherence_lifetime(conc_t, concurrence,
                                          threshold_frac=1.0/np.e)

        print(f"τ_coh = {tau_c:.2e} s, τ_ent = {tau_ent:.2e} s")

        dephasing_results.append({
            'gamma_2_Hz': float(gamma_2),
            'tau_coherence_s': tau_c,
            'tau_entanglement_s': tau_ent,
        })

        # Store selected curves
        nearest_rate = min(plot_rates, key=lambda r: abs(r - gamma_2))
        if abs(nearest_rate - gamma_2) / nearest_rate < 0.2:
            selected_curves[f"{gamma_2:.0e} Hz"] = {
                'times': tlist,
                'purity': purity,
                'conc_times': conc_t,
                'concurrence': concurrence,
            }

    # ═══════════════════════════════════════════════
    # SWEEP 2: Temperature at fixed dephasing
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("  Sweep 2: Temperature at Γ₂ = 1000 Hz")
    print(f"{'─' * 60}")

    gamma_2_fixed = 1000 * 2 * np.pi  # 1 kHz dephasing
    temp_results = []
    temp_curves = {}

    for T_K, label in zip(TEMPERATURES_K, TEMP_LABELS):
        # T₁ scales roughly linearly with temperature in solids:
        # T₁(T) ≈ T₁(310K) × (310/T)  (simplified Arrhenius-like)
        T1_T = T1_body * (310.0 / T_K)
        gamma_1_T = 1.0 / T1_T

        # Dephasing rate also increases with temperature:
        # Γ₂(T) ∝ T via thermal phonon population
        gamma_2_T = gamma_2_fixed * (T_K / 310.0)

        print(f"    {label}: T₁ = {T1_T:.1f} s, "
              f"Γ₂ = {gamma_2_T / (2*np.pi):.0f} Hz ... ", end="",
              flush=True)

        est_tau = 1.0 / (gamma_2_T + gamma_1_T)
        tmax = min(T_MAX_S, max(est_tau * 5, 1e-4))

        tlist, purity, conc_t, concurrence = run_coherence_decay(
            H, rho_0, gamma_1_T, gamma_2_T, tmax=tmax
        )
        tau_c = find_coherence_lifetime(tlist, purity)

        print(f"τ_coh = {tau_c:.2e} s")

        temp_results.append({
            'temperature_K': T_K,
            'label': label,
            'T1_s': T1_T,
            'gamma_2_Hz': gamma_2_T / (2 * np.pi),
            'tau_coherence_s': tau_c,
        })
        temp_curves[label] = {
            'times': tlist,
            'purity': purity,
        }

    # ═══════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 60}")

    # Find the "quantum window"
    for r in dephasing_results:
        if r['tau_coherence_s'] >= 1e-3:
            window_rate = r['gamma_2_Hz']
            window_tau = r['tau_coherence_s']
            print(f"  ◎ Quantum window opens at Γ₂ ≤ {window_rate:.0e} Hz")
            print(f"    → Coherence survives ≥ {window_tau*1000:.1f} ms")
            break
    else:
        print("  ◎ No ms-scale coherence found in parameter range")

    body_result = [r for r in temp_results if r['temperature_K'] == 310][0]
    print(f"  ◎ At body temperature (310K): τ_coh = "
          f"{body_result['tau_coherence_s']*1000:.2f} ms")
    print(f"  ◎ Total simulation time: {elapsed:.1f} s")
    print(f"{'═' * 60}")

    # ── Save metrics ──
    metrics = {
        'experiment': '1b_posner_spin_dynamics',
        'physics': {
            'n_spins': N_SPINS,
            'hilbert_dim': 2**N_SPINS,
            'larmor_freq_Hz': round(freq_L_Hz, 1),
            'B_earth_uT': B_EARTH * 1e6,
            'T1_body_s': T1_body,
            'dipolar_couplings': couplings,
        },
        'dephasing_sweep': dephasing_results,
        'temperature_sweep': temp_results,
        'elapsed_s': round(elapsed, 1),
    }
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Generate dashboard ──
    make_dashboard(selected_curves, dephasing_results, temp_results,
                   temp_curves, couplings, freq_L_Hz, output_img)

    print(f"\n{'═' * 60}")
    print("  EXPERIMENT 1b COMPLETE")
    print(f"{'═' * 60}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(selected_curves, dephasing_results, temp_results,
                   temp_curves, couplings, larmor_Hz, output_path):
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
    BORDER = (0, 0.824, 1.0, 0.08)
    MONO = "monospace"

    colors_cycle = [ACCENT_CYAN, ACCENT_TEAL, ACCENT_GREEN, ACCENT_AMBER,
                    ACCENT_RED, "#a78bfa"]

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

    # ── Panel A: Coherence decay at different Γ₂ ──
    ax_coh = fig.add_subplot(gs[0, 0])
    style_ax(ax_coh, "◈  PURITY Tr(ρ₀₁²) AT 310 K")
    for idx, (label, data) in enumerate(sorted(selected_curves.items())):
        color = colors_cycle[idx % len(colors_cycle)]
        ax_coh.plot(data['times'] * 1000, data['purity'],
                    color=color, linewidth=1.5, alpha=0.85,
                    label=f"Γ₂={label}")
    ax_coh.axhline(y=0.25, color=ACCENT_RED, linestyle="--",
                   alpha=0.4, linewidth=0.8, label="max mixed (0.25)")
    ax_coh.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_coh.set_ylabel("Purity Tr(ρ²)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_coh.set_ylim(0.2, 1.05)
    ax_coh.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                  loc="upper right")

    # ── Panel B: Entanglement (concurrence) decay ──
    ax_ent = fig.add_subplot(gs[0, 1])
    style_ax(ax_ent, "◈  ENTANGLEMENT (CONCURRENCE) SPINS 0–1")
    for idx, (label, data) in enumerate(sorted(selected_curves.items())):
        color = colors_cycle[idx % len(colors_cycle)]
        if len(data['concurrence']) > 0:
            ax_ent.plot(data['conc_times'] * 1000, data['concurrence'],
                        color=color, linewidth=1.5, alpha=0.85,
                        label=f"Γ₂={label}", marker=".", markersize=3)
    ax_ent.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_ent.set_ylabel("Concurrence", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_ent.set_ylim(-0.05, 1.05)
    ax_ent.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                  loc="upper right")

    # ── Panel C: Coherence lifetime vs dephasing rate (log-log) ──
    ax_tau = fig.add_subplot(gs[1, 0])
    style_ax(ax_tau, "◈  PURITY LIFETIME vs DEPHASING RATE")
    rates = [r['gamma_2_Hz'] for r in dephasing_results]
    taus_coh = [r['tau_coherence_s'] for r in dephasing_results]
    taus_ent = [r['tau_entanglement_s'] for r in dephasing_results]
    ax_tau.loglog(rates, taus_coh, color=ACCENT_CYAN, marker="o",
                  markersize=5, linewidth=1.5, label="τ_purity")
    ax_tau.loglog(rates, taus_ent, color=ACCENT_TEAL, marker="s",
                  markersize=5, linewidth=1.5, label="τ_entanglement")
    # Mark 1 ms line
    ax_tau.axhline(y=1e-3, color=ACCENT_AMBER, linestyle="--",
                   alpha=0.5, linewidth=1, label="1 ms (bio threshold)")
    ax_tau.set_xlabel("Dephasing rate Γ₂ (Hz)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_tau.set_ylabel("Lifetime (s)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_tau.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Temperature sweep ──
    ax_temp = fig.add_subplot(gs[1, 1])
    style_ax(ax_temp, "◈  PURITY DECAY vs TEMPERATURE (Γ₂=1 kHz)")
    for idx, (label, data) in enumerate(temp_curves.items()):
        color = colors_cycle[idx % len(colors_cycle)]
        ax_temp.plot(data['times'] * 1000, data['purity'],
                     color=color, linewidth=1.5, alpha=0.85, label=label)
    ax_temp.set_xlabel("Time (ms)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_temp.set_ylabel("Purity Tr(ρ²)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_temp.set_ylim(0.2, 1.05)
    ax_temp.legend(fontsize=7, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY,
                   loc="upper right")

    # ── Panel E: Summary metrics ──
    ax_sum = fig.add_subplot(gs[2, :])
    style_ax(ax_sum, "◈  POSNER MOLECULE — QUANTUM PARAMETER SPACE")
    ax_sum.axis("off")

    # Build summary text
    body_res = [r for r in temp_results if r['temperature_K'] == 310][0]
    cryo_res = [r for r in temp_results if r['temperature_K'] == 4][0]

    # Find threshold rate for 1 ms coherence
    threshold_rate = None
    for r in sorted(dephasing_results, key=lambda x: x['gamma_2_Hz']):
        if r['tau_coherence_s'] >= 1e-3:
            threshold_rate = r['gamma_2_Hz']
            break

    lines = [
        f"³¹P LARMOR FREQUENCY:  {larmor_Hz:.1f} Hz  "
        f"(B₀ = {B_EARTH*1e6:.0f} µT)",
        "",
        "DIPOLAR COUPLINGS:",
    ]
    for c in couplings[:4]:
        lines.append(
            f"  Spins {c['spins']}:  {c['distance_A']:.1f} Å  →  "
            f"J = {c['coupling_Hz']:.1f} Hz"
        )
    lines += [
        "",
        f"BODY TEMPERATURE (310 K):  τ_coh = "
        f"{body_res['tau_coherence_s']*1000:.2f} ms",
        f"CRYOGENIC (4 K):           τ_coh = "
        f"{cryo_res['tau_coherence_s']*1000:.2f} ms",
        "",
        f"QUANTUM WINDOW:  Γ₂ ≤ {threshold_rate:.0e} Hz  →  "
        f"τ > 1 ms (biologically relevant)"
        if threshold_rate else
        "QUANTUM WINDOW:  Not found in parameter range",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=10, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    # Supertitle
    fig.suptitle(
        "EXPERIMENT 1b ── POSNER MOLECULE ³¹P SPIN DYNAMICS",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.98
    )
    fig.text(0.5, 0.955,
             f"6 spins · 64-dim Hilbert space · QuTiP Lindblad · "
             f"Fisher (2015) model",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
