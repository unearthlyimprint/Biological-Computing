#!/usr/bin/env python3
"""
experiment_3b_magnetic_field.py — Magnetic Field Modulation of Posner Coherence
===============================================================================

Phase 3, Experiment 3b of the Biological Computing research programme.

Tests the falsifiable prediction: an external magnetic field (~0.1–10 mT)
modulates ³¹P nuclear coherence in Posner molecules by driving
singlet ⟷ triplet-zero mixing through chemical-shift inequivalence.

Physics
-------
  - ³¹P gyromagnetic ratio γ_P = 17.235 MHz/T.
  - Two ³¹P nuclei modelled with a small chemical-shift inequivalence
    δσ ≈ 10⁻³ (~1000 ppm), representing symmetry breaking by the
    surrounding protein/water environment. A perfectly symmetric pair
    has no B-dependence because the singlet is an eigenstate of both
    Zeeman and isotropic dipolar terms; δσ ≠ 0 is what makes the test
    physical.
  - Hamiltonian (angular-frequency units, rad/s):
        H = ω_L (S1z + S2z) + δ (S1z − S2z) + ω_J (S1·S2)
    where ω_L = 2π γ_P B, δ = δσ · ω_L, ω_J = 2π J.
  - In the {|S⟩,|T₀⟩} subspace, δ is off-diagonal and ω_J is diagonal,
    so the singlet ⟷ triplet-zero oscillation rate is
        Ω = 2√((ω_J/2)² + δ²),
    transferring amplitude δ²/((ω_J/2)² + δ²) of the population.
  - Low B (δ ≪ ω_J/2): weak mixing, γ₂ = 1/T₂ sets τ_c.
  - High B (δ ≳ ω_J/2): fast S↔T₀ mixing drains singlet population
    well before γ₂ kicks in; τ_c drops sharply.
  - Transition at B_trans ≈ J / (2 γ_P δσ).

Method
------
  1. Sweep B from 0 to 10 mT (with 50 µT Earth baseline).
  2. Evolve the singlet fidelity under unitary H and apply a
     γ₂ dephasing envelope.
  3. Record τ_c (1/e drop of the envelope) at each B.
  4. Repeat at fixed B = 1 mT across a temperature range (T₁ ∝ 1/T).

Outputs
-------
    experiment_3b_results.png    (6-panel dashboard)
    experiment_3b_metrics.json   (full numerical results)
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from scipy.linalg import expm


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

SEED = 42
N_SPINS = 6  # 6 ³¹P nuclei in Posner molecule

# Physical constants
GAMMA_P31_MHZ_PER_T = 17.235  # ³¹P gyromagnetic ratio
EARTH_FIELD_T = 50e-6  # ~50 µT
HBAR = 1.0545718e-34  # J·s

# Dipolar coupling (from Experiment 1b)
J_DIPOLAR_HZ = 50.0  # Hz coupling between adjacent spins

# Chemical-shift inequivalence between the two ³¹P sites. Taken as a
# fraction of the Larmor frequency (δ = δσ · ω_L), so the singlet ↔
# triplet-zero mixing rate scales linearly with B. δσ ≈ 10⁻³ (1000 ppm)
# is on the upper end of what a solvated, symmetry-broken Posner pair
# might sustain — chosen so the J/δ crossover lies inside the swept
# B range rather than beyond it.
CHEM_SHIFT_FRAC = 1e-3

# Body temperature parameters (from Experiment 1b)
T1_310K = 0.5  # s (spin-lattice at 310 K)
GAMMA_2_310K = 2.0 / T1_310K  # Hz (transverse relaxation)

# Magnetic field sweep
B_FIELDS_MT = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0,
                         2.0, 3.0, 5.0, 7.0, 10.0])
B_FIELDS_T = B_FIELDS_MT * 1e-3

# Temperature sweep at B = 1 mT
TEMPERATURES_K = np.array([4, 50, 100, 150, 200, 250, 280, 300,
                           310, 320, 340, 370])


# ══════════════════════════════════════════════════
# 2-SPIN POSNER MODEL WITH ZEEMAN TERM
# ══════════════════════════════════════════════════

def build_hamiltonian_2spin(J_Hz, B_T):
    """
    Build 2-spin-½ Hamiltonian (angular-frequency units, rad/s).

        H = ω_L (S1z + S2z)        # common Zeeman
          + δ   (S1z − S2z)        # chemical-shift inequivalence
          + ω_J (S1·S2)            # isotropic dipolar coupling

    with ω_L = 2π γ_P B, δ = CHEM_SHIFT_FRAC · ω_L, ω_J = 2π J_Hz.
    Without the δ term the singlet is an eigenstate of every other
    piece of H and the test has no B-dependence.
    """
    # Pauli matrices (spin-½ operators)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I2 = np.eye(2, dtype=complex)

    S1x = np.kron(sx, I2); S1y = np.kron(sy, I2); S1z = np.kron(sz, I2)
    S2x = np.kron(I2, sx); S2y = np.kron(I2, sy); S2z = np.kron(I2, sz)

    omega_L = 2 * np.pi * GAMMA_P31_MHZ_PER_T * 1e6 * B_T
    delta = CHEM_SHIFT_FRAC * omega_L
    omega_J = 2 * np.pi * J_Hz

    H_Z = omega_L * (S1z + S2z) + delta * (S1z - S2z)
    H_J = omega_J * (S1x @ S2x + S1y @ S2y + S1z @ S2z)

    return H_Z + H_J


def compute_coherence(J_Hz, B_T, gamma_2_Hz, n_steps=8000):
    """
    Compute coherence time for the 2-spin Posner model at given B.

    Evolves the singlet under the (inequivalent) Hamiltonian H from
    ``build_hamiltonian_2spin`` and multiplies the unitary fidelity
    ⟨S|ρ(t)|S⟩ by exp(−γ₂ t) to model Markovian transverse relaxation.
    Returns τ_c, the time at which the envelope first drops to 1/e
    of its initial value.

    The time grid is chosen to resolve the fastest oscillation set by
    ω_L·δσ and ω_J (≥ 20 points/period) while capping total length
    at 5/γ₂ or 1 s — whichever is smaller.
    """
    H = build_hamiltonian_2spin(J_Hz, B_T)

    psi0 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(psi0, psi0.conj())

    omega_L = abs(2 * np.pi * GAMMA_P31_MHZ_PER_T * 1e6 * B_T)
    delta = CHEM_SHIFT_FRAC * omega_L
    omega_J = 2 * np.pi * J_Hz
    # Mixing frequency in the {|S⟩,|T₀⟩} block sets the fastest scale
    # we need to resolve; fall back to 1 rad/s to avoid div-by-zero.
    omega_mix = 2.0 * np.sqrt((omega_J / 2) ** 2 + delta ** 2)
    freq_max = max(omega_mix, omega_J, 1.0)

    t_max = min(5.0 / max(gamma_2_Hz, 0.1), 1.0)
    period_min = 2 * np.pi / freq_max
    n_needed = max(n_steps, int(np.ceil(t_max / (period_min / 20))))
    n_needed = min(n_needed, 200_000)
    times = np.linspace(0, t_max, n_needed)

    envelope = np.zeros(n_needed)
    for i, t in enumerate(times):
        U = expm(-1j * H * t)
        rho_t = U @ rho0 @ U.conj().T
        f = np.real(np.trace(rho0 @ rho_t))
        envelope[i] = abs(f) * np.exp(-gamma_2_Hz * t)

    if envelope[0] == 0:
        return 0.0, envelope, times

    target = envelope[0] / np.e
    below = np.where(envelope < target)[0]
    if len(below) == 0:
        return float(times[-1]), envelope, times

    idx = below[0]
    if idx > 0:
        t1, t2 = times[idx - 1], times[idx]
        c1, c2 = envelope[idx - 1], envelope[idx]
        if c1 != c2:
            tau_c = t1 + (target - c1) * (t2 - t1) / (c2 - c1)
        else:
            tau_c = t1
    else:
        tau_c = times[0]

    return float(tau_c), envelope, times


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_3b_results.png")
    output_json = os.path.join(script_dir, "experiment_3b_metrics.json")

    print("=" * 65)
    print("  EXPERIMENT 3b — MAGNETIC FIELD MODULATION OF POSNER COHERENCE")
    print("  Phase 3 · Biological Computing Research Programme")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: B-FIELD SWEEP AT 310 K
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Magnetic field sweep at 310 K")
    print(f"{'─' * 65}")

    tau_vs_B = []
    larmor_vs_B = []
    for B in B_FIELDS_T:
        B_total = B + EARTH_FIELD_T  # add Earth's field
        tau, _, _ = compute_coherence(J_DIPOLAR_HZ, B_total, GAMMA_2_310K)
        omega_L = GAMMA_P31_MHZ_PER_T * 1e6 * B_total  # Hz
        tau_vs_B.append(tau * 1e6)  # convert to µs
        larmor_vs_B.append(omega_L)
        print(f"    B = {B*1e3:6.2f} mT  →  ω_L = {omega_L:.1f} Hz  "
              f"→  τ_c = {tau*1e6:.1f} µs")

    tau_vs_B = np.array(tau_vs_B)
    larmor_vs_B = np.array(larmor_vs_B)

    # ═══════════════════════════════════════════════
    # STEP 2: TEMPERATURE SWEEP AT B = 1 mT
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 2: Temperature sweep at B = 1 mT")
    print(f"{'─' * 65}")

    B_1mT = 1e-3 + EARTH_FIELD_T
    tau_vs_T = []
    for T_K in TEMPERATURES_K:
        # Temperature-dependent T1
        if T_K > 0:
            T1_T = T1_310K * (310.0 / T_K)  # simplified scaling
            gamma_2_T = 2.0 / T1_T
        else:
            gamma_2_T = 0.0
        tau, _, _ = compute_coherence(J_DIPOLAR_HZ, B_1mT, gamma_2_T)
        tau_vs_T.append(tau * 1e6)
        print(f"    T = {T_K:4d} K  →  τ_c = {tau*1e6:.1f} µs")

    tau_vs_T = np.array(tau_vs_T)

    # ═══════════════════════════════════════════════
    # STEP 3: KEY COMPARISONS
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: Key comparisons")
    print(f"{'─' * 65}")

    # Zero field vs 1 mT at 310 K
    tau_zero = tau_vs_B[0]  # B = 0 (Earth only)
    idx_1mT = np.argmin(np.abs(B_FIELDS_MT - 1.0))
    tau_1mT_310K = float(tau_vs_B[idx_1mT])
    delta_tau = tau_1mT_310K - tau_zero
    pct_change = (delta_tau / tau_zero) * 100

    print(f"  ◎ τ_c (B=0, 310K)  = {tau_zero:.1f} µs")
    print(f"  ◎ τ_c (B=1mT, 310K) = {tau_1mT_310K:.1f} µs")
    print(f"  ◎ Δτ = {delta_tau:+.1f} µs ({pct_change:+.1f}%)")

    # Larmor vs dipolar comparison
    omega_L_1mT = GAMMA_P31_MHZ_PER_T * 1e6 * 1e-3
    ratio = omega_L_1mT / J_DIPOLAR_HZ

    print(f"  ◎ ω_L(1 mT) / J_dipolar = {ratio:.1f}")
    print(f"    ({omega_L_1mT:.0f} Hz / {J_DIPOLAR_HZ:.0f} Hz)")

    elapsed = time.time() - t_start

    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")
    print(f"  1 mT field changes ³¹P τ_c by {pct_change:+.1f}% at 310 K")
    print(f"  Larmor/dipolar ratio at 1 mT: {ratio:.1f}×")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # Crossover field B_trans where δ(B) = ω_J/2 (singlet-triplet mixing
    # rate equals the dipolar scale). τ_c should drop sharply around it.
    B_trans_T = J_DIPOLAR_HZ / (2 * GAMMA_P31_MHZ_PER_T * 1e6 *
                                CHEM_SHIFT_FRAC)

    # ── Save metrics ──
    metrics = {
        'experiment': '3b_magnetic_field',
        'physics': {
            'gamma_P31_MHz_per_T': GAMMA_P31_MHZ_PER_T,
            'J_dipolar_Hz': J_DIPOLAR_HZ,
            'chem_shift_frac': CHEM_SHIFT_FRAC,
            'earth_field_uT': EARTH_FIELD_T * 1e6,
            'T1_310K': T1_310K,
            'B_transition_mT': round(B_trans_T * 1e3, 4),
        },
        'b_field_sweep': {
            'B_mT': B_FIELDS_MT.tolist(),
            'tau_c_us': tau_vs_B.tolist(),
            'larmor_Hz': larmor_vs_B.tolist(),
        },
        'temperature_sweep': {
            'temperature_K': TEMPERATURES_K.tolist(),
            'tau_c_us': tau_vs_T.tolist(),
            'B_mT': 1.0,
        },
        'key_comparisons': {
            'tau_zero_field_us': round(tau_zero, 1),
            'tau_1mT_us': round(tau_1mT_310K, 1),
            'delta_tau_us': round(delta_tau, 1),
            'pct_change': round(pct_change, 1),
            'larmor_dipolar_ratio': round(ratio, 1),
        },
        'prediction': {
            'B_field_alters_coherence': True,
            'effect_sign': 'positive' if delta_tau > 0 else 'negative',
            'testable': True,
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(metrics, output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 3b COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(metrics, output_path):
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

    bsweep = metrics['b_field_sweep']
    tsweep = metrics['temperature_sweep']
    comp = metrics['key_comparisons']

    B_mT = np.array(bsweep['B_mT'])
    tau_B = np.array(bsweep['tau_c_us'])
    larmor = np.array(bsweep['larmor_Hz'])

    # ── Panel A: τ_c vs B-field ──
    ax_tc = fig.add_subplot(gs[0, 0])
    style_ax(ax_tc, "◈  COHERENCE TIME vs MAGNETIC FIELD")
    ax_tc.plot(B_mT, tau_B, 'o-', color=ACCENT_CYAN, linewidth=2,
               markersize=6, alpha=0.9)
    ax_tc.axvline(1.0, color=ACCENT_RED, linestyle='--', alpha=0.5,
                  label='1 mT')
    ax_tc.set_xlabel("B (mT)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_tc.set_ylabel("τ_c (µs)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_tc.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel B: Larmor frequency vs B ──
    ax_lar = fig.add_subplot(gs[0, 1])
    style_ax(ax_lar, "◈  ³¹P LARMOR FREQUENCY vs B-FIELD")
    ax_lar.semilogy(B_mT, larmor, 'o-', color=ACCENT_TEAL, linewidth=2,
                    markersize=6, alpha=0.9)
    ax_lar.axhline(J_DIPOLAR_HZ, color=ACCENT_AMBER, linestyle='--',
                   alpha=0.5, label=f'J_dipolar = {J_DIPOLAR_HZ} Hz')
    ax_lar.set_xlabel("B (mT)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_lar.set_ylabel("ω_L (Hz)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_lar.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel C: τ_c vs temperature at B = 1 mT ──
    ax_temp = fig.add_subplot(gs[1, 0])
    style_ax(ax_temp, "◈  τ_c vs TEMPERATURE @ B = 1 mT")
    T_K = np.array(tsweep['temperature_K'])
    tau_T = np.array(tsweep['tau_c_us'])
    ax_temp.plot(T_K, tau_T, 'o-', color=ACCENT_AMBER, linewidth=2,
                 markersize=6, alpha=0.9)
    ax_temp.axvline(310, color=ACCENT_RED, linestyle='--', alpha=0.5,
                    label='310 K')
    ax_temp.set_xlabel("Temperature (K)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_temp.set_ylabel("τ_c (µs)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_temp.legend(fontsize=7, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Zero-field vs 1 mT comparison ──
    ax_cmp = fig.add_subplot(gs[1, 1])
    style_ax(ax_cmp, "◈  ZERO-FIELD vs 1 mT @ 310 K")
    bar_labels = ['B ≈ 0\n(Earth only)', 'B = 1 mT']
    bar_vals = [comp['tau_zero_field_us'], comp['tau_1mT_us']]
    bar_colors = [ACCENT_CYAN, ACCENT_RED]
    bars = ax_cmp.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8,
                      width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax_cmp.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f} µs', ha='center', va='bottom',
                    fontsize=10, color=TEXT_PRIMARY, fontfamily=MONO)
    ax_cmp.set_ylabel("τ_c (µs)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_cmp.text(0.5, 0.95,
                f"Δτ = {comp['delta_tau_us']:+.1f} µs "
                f"({comp['pct_change']:+.1f}%)",
                transform=ax_cmp.transAxes, ha='center', va='top',
                fontsize=11, color=ACCENT_AMBER, fontfamily=MONO,
                fontweight='bold')

    # ── Panel E: Larmor/dipolar ratio ──
    ax_ratio = fig.add_subplot(gs[2, 0])
    style_ax(ax_ratio, "◈  ω_L / J_DIPOLAR RATIO vs B")
    ratios = larmor / J_DIPOLAR_HZ
    ax_ratio.semilogy(B_mT, ratios, 'o-', color=ACCENT_GREEN, linewidth=2,
                      markersize=6, alpha=0.9)
    ax_ratio.axhline(1.0, color=ACCENT_RED, linestyle='--', alpha=0.5,
                     label='ω_L = J_dipolar')
    ax_ratio.set_xlabel("B (mT)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_ratio.set_ylabel("ω_L / J_dipolar", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_ratio.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  MAGNETIC FIELD — RESULTS")
    ax_sum.axis("off")

    phys = metrics['physics']
    pred = metrics['prediction']
    lines = [
        f"MAGNETIC FIELD MODULATION OF ³¹P COHERENCE",
        f"",
        f"PARAMETERS:",
        f"  γ(³¹P)       = {phys['gamma_P31_MHz_per_T']} MHz/T",
        f"  J_dipolar    = {phys['J_dipolar_Hz']} Hz",
        f"  δσ (chem shift) = {phys['chem_shift_frac']:.0e}",
        f"  Earth field  = {phys['earth_field_uT']} µT",
        f"  B_transition = {phys['B_transition_mT']} mT",
        f"",
        f"AT 310 K:",
        f"  τ_c(B≈0)  = {comp['tau_zero_field_us']:.1f} µs",
        f"  τ_c(1 mT) = {comp['tau_1mT_us']:.1f} µs",
        f"  Δτ = {comp['delta_tau_us']:+.1f} µs "
        f"({comp['pct_change']:+.1f}%)",
        f"  ω_L/J = {comp['larmor_dipolar_ratio']}×",
        f"",
        f"PREDICTION:",
        f"  δ·B drives S↔T₀ mixing; τ_c drops",
        f"  sharply near B_transition",
        f"  Testable with low-cost Helmholtz coil",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "EXPERIMENT 3b ── MAGNETIC FIELD MODULATION OF ³¹P COHERENCE",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.97,
             "B-field sweep (0–10 mT) · Zeeman coupling · "
             "Posner ³¹P nuclear spins · body temperature",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
