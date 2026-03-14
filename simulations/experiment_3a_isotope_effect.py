#!/usr/bin/env python3
"""
experiment_3a_isotope_effect.py — D₂O Isotope Effect on ENAQT
=============================================================

Phase 3, Experiment 3a of the Biological Computing research programme.

Tests the falsifiable prediction: replacing H₂O with D₂O should alter
the effective dephasing rate and shift the ENAQT transport curve.

Physics:
  - D₂O has ~1.37× higher vibrational mass than H₂O
  - O-H stretching at ~3400 cm⁻¹ → O-D at ~2500 cm⁻¹
  - Protein bath spectral density shifts proportionally
  - Effective dephasing at body temp (310 K) changes

Method:
  1. Compute ENAQT P₄(γ) for H₂O (standard, as in Exp 1c)
  2. Apply isotope mass scaling to γ: γ_D₂O = γ_H₂O × √(m_H/m_D)
     This follows from ω ∝ √(k/m) → γ ∝ ω
  3. Compare transport efficiency at body temperature
  4. Predict MC shift using the 1e bridge (p_rel = P₄)

Prediction:
  - D₂O reduces effective dephasing → moves biology further from ENAQT peak
  - P₄(310 K, D₂O) < P₄(310 K, H₂O)
  - MC should decrease if organoids are cultured in D₂O

Outputs:
    experiment_3a_results.png    (6-panel dashboard)
    experiment_3a_metrics.json   (full numerical results)
"""

import json
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_1c_enaqt_ion_channel import compute_transport_efficiency


# ══════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════

SEED = 42

# Mass ratio: deuterium / hydrogen
MASS_RATIO_DH = 2.014 / 1.008  # ≈ 1.998

# Frequency scaling: ω_D = ω_H × √(m_H / m_D)
FREQ_SCALE_D2O = np.sqrt(1.0 / MASS_RATIO_DH)  # ≈ 0.707

# Body temperature dephasing (from Experiment 1c)
GAMMA_BODY_TEMP_H2O = 215.46  # cm⁻¹ at 310 K

# D₂O-shifted body temperature dephasing
GAMMA_BODY_TEMP_D2O = GAMMA_BODY_TEMP_H2O * FREQ_SCALE_D2O

# Mixed solvent fractions (for dose-response prediction)
D2O_FRACTIONS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0])

# Fine gamma sweep for curves
GAMMA_FINE = np.logspace(np.log10(0.01), np.log10(5000), 200)


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_img = os.path.join(script_dir, "experiment_3a_results.png")
    output_json = os.path.join(script_dir, "experiment_3a_metrics.json")

    print("=" * 65)
    print("  EXPERIMENT 3a — D₂O ISOTOPE EFFECT ON ENAQT")
    print("  Phase 3 · Biological Computing Research Programme")
    print("=" * 65)

    t_start = time.time()

    # ═══════════════════════════════════════════════
    # STEP 1: COMPUTE H₂O and D₂O ENAQT CURVES
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 1: Computing ENAQT P₄(γ) for H₂O and D₂O")
    print(f"{'─' * 65}")
    print(f"  ◎ Frequency scaling factor: {FREQ_SCALE_D2O:.4f}")
    print(f"  ◎ γ_body(H₂O) = {GAMMA_BODY_TEMP_H2O:.2f} cm⁻¹")
    print(f"  ◎ γ_body(D₂O) = {GAMMA_BODY_TEMP_D2O:.2f} cm⁻¹")

    # H₂O curve (standard)
    p4_h2o = np.array([compute_transport_efficiency(g)[0]
                       for g in GAMMA_FINE])

    # D₂O curve: the spectral density shifts frequencies down
    # Effectively, at each physical temperature, the dephasing
    # rate maps to a different gamma
    gamma_d2o = GAMMA_FINE * FREQ_SCALE_D2O
    p4_d2o = np.array([compute_transport_efficiency(g)[0]
                       for g in gamma_d2o])

    # Find peaks
    peak_h2o_idx = np.argmax(p4_h2o)
    peak_d2o_idx = np.argmax(p4_d2o)
    peak_h2o_gamma = float(GAMMA_FINE[peak_h2o_idx])
    peak_d2o_gamma = float(GAMMA_FINE[peak_d2o_idx])
    peak_h2o_p4 = float(p4_h2o[peak_h2o_idx])
    peak_d2o_p4 = float(p4_d2o[peak_d2o_idx])

    print(f"  ◎ H₂O peak: γ = {peak_h2o_gamma:.2f} cm⁻¹, "
          f"P₄ = {peak_h2o_p4:.4f}")
    print(f"  ◎ D₂O peak: γ = {peak_d2o_gamma:.2f} cm⁻¹, "
          f"P₄ = {peak_d2o_p4:.4f}")

    # ═══════════════════════════════════════════════
    # STEP 2: BODY TEMPERATURE COMPARISON
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 2: Body temperature comparison")
    print(f"{'─' * 65}")

    p4_body_h2o, _, _, _ = compute_transport_efficiency(GAMMA_BODY_TEMP_H2O)
    p4_body_d2o, _, _, _ = compute_transport_efficiency(GAMMA_BODY_TEMP_D2O)

    delta_p4 = p4_body_d2o - p4_body_h2o
    pct_change = (delta_p4 / p4_body_h2o) * 100

    print(f"  ◎ P₄(310 K, H₂O) = {p4_body_h2o:.4f}")
    print(f"  ◎ P₄(310 K, D₂O) = {p4_body_d2o:.4f}")
    print(f"  ◎ ΔP₄ = {delta_p4:+.4f}  ({pct_change:+.1f}%)")

    # ═══════════════════════════════════════════════
    # STEP 3: DOSE-RESPONSE (mixed solvent)
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 3: D₂O dose-response (mixed solvent fractions)")
    print(f"{'─' * 65}")

    dose_p4 = []
    dose_gamma = []
    for frac in D2O_FRACTIONS:
        # Linear interpolation of effective dephasing
        gamma_eff = GAMMA_BODY_TEMP_H2O * (1.0 - frac) + \
                    GAMMA_BODY_TEMP_D2O * frac
        p4_eff, _, _, _ = compute_transport_efficiency(gamma_eff)
        dose_p4.append(float(p4_eff))
        dose_gamma.append(float(gamma_eff))
        print(f"    D₂O = {frac:.0%} → γ = {gamma_eff:.2f} cm⁻¹, "
              f"P₄ = {p4_eff:.4f}")

    dose_p4 = np.array(dose_p4)

    # ═══════════════════════════════════════════════
    # STEP 4: MC PREDICTION (using 1e bridge)
    # ═══════════════════════════════════════════════
    print(f"\n{'─' * 65}")
    print("  Step 4: MC prediction via ENAQT bridge")
    print(f"{'─' * 65}")

    # From Experiment 1e: MC ∝ P₄ through the bridge
    # Normalise to the H₂O MC value
    mc_ratio = dose_p4 / dose_p4[0]  # relative to pure H₂O

    for i, frac in enumerate(D2O_FRACTIONS):
        print(f"    D₂O = {frac:.0%} → MC_ratio = {mc_ratio[i]:.4f}")

    elapsed = time.time() - t_start

    print(f"\n{'═' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 65}")
    print(f"  D₂O reduces effective dephasing by "
          f"{(1 - FREQ_SCALE_D2O) * 100:.1f}%")
    print(f"  At 310 K: ΔP₄ = {delta_p4:+.4f} ({pct_change:+.1f}%)")
    print(f"  At 100% D₂O: MC predicted to change by "
          f"{(mc_ratio[-1] - 1) * 100:+.1f}%")
    print(f"  ◎ Elapsed: {elapsed:.1f} s")
    print(f"{'═' * 65}")

    # ── Save metrics ──
    metrics = {
        'experiment': '3a_isotope_effect',
        'physics': {
            'mass_ratio_DH': round(MASS_RATIO_DH, 4),
            'freq_scale_D2O': round(FREQ_SCALE_D2O, 4),
            'gamma_body_H2O': GAMMA_BODY_TEMP_H2O,
            'gamma_body_D2O': round(GAMMA_BODY_TEMP_D2O, 2),
        },
        'body_temperature': {
            'P4_H2O': round(p4_body_h2o, 4),
            'P4_D2O': round(p4_body_d2o, 4),
            'delta_P4': round(delta_p4, 4),
            'pct_change': round(pct_change, 1),
        },
        'enaqt_peaks': {
            'H2O': {'gamma': peak_h2o_gamma, 'P4': peak_h2o_p4},
            'D2O': {'gamma': peak_d2o_gamma, 'P4': peak_d2o_p4},
        },
        'dose_response': {
            'D2O_fractions': D2O_FRACTIONS.tolist(),
            'gamma_effective': dose_gamma,
            'P4_values': dose_p4.tolist(),
            'MC_ratio': mc_ratio.tolist(),
        },
        'prediction': {
            'D2O_reduces_dephasing': True,
            'P4_decreases_at_body_temp': bool(delta_p4 < 0),
            'MC_change_at_100pct_D2O': round(
                (mc_ratio[-1] - 1) * 100, 1),
        },
        'elapsed_s': round(elapsed, 1),
    }

    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ◎ Metrics saved: {output_json}")

    # ── Dashboard ──
    make_dashboard(metrics, GAMMA_FINE, p4_h2o, gamma_d2o, p4_d2o,
                   D2O_FRACTIONS, dose_p4, dose_gamma, mc_ratio,
                   output_img)

    print(f"\n{'═' * 65}")
    print("  EXPERIMENT 3a COMPLETE")
    print(f"{'═' * 65}\n")


# ══════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════

def make_dashboard(metrics, gamma_h2o, p4_h2o, gamma_d2o, p4_d2o,
                   fractions, dose_p4, dose_gamma, mc_ratio, output_path):
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
    ACCENT_PURPLE = "#a855f7"
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

    bt = metrics['body_temperature']
    phys = metrics['physics']

    # ── Panel A: H₂O vs D₂O ENAQT curves ──
    ax_cmp = fig.add_subplot(gs[0, 0])
    style_ax(ax_cmp, "◈  ENAQT CURVES — H₂O vs D₂O")
    ax_cmp.semilogx(gamma_h2o, p4_h2o, color=ACCENT_CYAN, linewidth=2,
                    alpha=0.9, label='H₂O')
    ax_cmp.semilogx(gamma_h2o, p4_d2o, color=ACCENT_PURPLE, linewidth=2,
                    alpha=0.9, linestyle='--', label='D₂O')
    ax_cmp.axvline(phys['gamma_body_H2O'], color=ACCENT_AMBER,
                   linestyle=':', alpha=0.5, label='310 K (H₂O)')
    ax_cmp.axvline(phys['gamma_body_D2O'], color=ACCENT_RED,
                   linestyle=':', alpha=0.5, label='310 K (D₂O)')
    ax_cmp.set_xlabel("γ (cm⁻¹)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_cmp.set_ylabel("P₄", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=9)
    ax_cmp.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel B: Body temperature detail ──
    ax_body = fig.add_subplot(gs[0, 1])
    style_ax(ax_body, "◈  BODY TEMPERATURE COMPARISON")
    bar_labels = ['H₂O', 'D₂O']
    bar_vals = [bt['P4_H2O'], bt['P4_D2O']]
    bar_colors = [ACCENT_CYAN, ACCENT_PURPLE]
    bars = ax_body.bar(bar_labels, bar_vals, color=bar_colors, alpha=0.8,
                       width=0.5)
    for bar, val in zip(bars, bar_vals):
        ax_body.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f'{val:.4f}', ha='center', va='bottom',
                     fontsize=10, color=TEXT_PRIMARY, fontfamily=MONO)
    ax_body.set_ylabel("P₄ at 310 K", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_body.text(0.5, 0.95,
                 f"ΔP₄ = {bt['delta_P4']:+.4f} ({bt['pct_change']:+.1f}%)",
                 transform=ax_body.transAxes, ha='center', va='top',
                 fontsize=11, color=ACCENT_RED, fontfamily=MONO,
                 fontweight='bold')

    # ── Panel C: Dose-response P₄ ──
    ax_dose = fig.add_subplot(gs[1, 0])
    style_ax(ax_dose, "◈  D₂O DOSE-RESPONSE — P₄")
    ax_dose.plot(fractions * 100, dose_p4, 'o-', color=ACCENT_PURPLE,
                 linewidth=2, markersize=6, alpha=0.9)
    ax_dose.axhline(dose_p4[0], color=ACCENT_CYAN, linestyle=':',
                    alpha=0.4, label=f'Pure H₂O (P₄={dose_p4[0]:.4f})')
    ax_dose.set_xlabel("D₂O fraction (%)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_dose.set_ylabel("P₄ at 310 K", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=9)
    ax_dose.legend(fontsize=7, facecolor=BG_SECONDARY,
                   edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Dose-response MC ratio ──
    ax_mc = fig.add_subplot(gs[1, 1])
    style_ax(ax_mc, "◈  D₂O DOSE-RESPONSE — MC PREDICTION")
    ax_mc.plot(fractions * 100, mc_ratio, 'o-', color=ACCENT_RED,
               linewidth=2, markersize=6, alpha=0.9)
    ax_mc.axhline(1.0, color=ACCENT_CYAN, linestyle=':', alpha=0.4,
                  label='Baseline (pure H₂O)')
    ax_mc.set_xlabel("D₂O fraction (%)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.set_ylabel("MC / MC_H₂O", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=9)
    ax_mc.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Effective γ shift ──
    ax_gamma = fig.add_subplot(gs[2, 0])
    style_ax(ax_gamma, "◈  EFFECTIVE DEPHASING vs D₂O FRACTION")
    ax_gamma.plot(fractions * 100, dose_gamma, 'o-', color=ACCENT_TEAL,
                  linewidth=2, markersize=6, alpha=0.9)
    ax_gamma.set_xlabel("D₂O fraction (%)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)
    ax_gamma.set_ylabel("γ_eff (cm⁻¹)", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=9)

    # ── Panel F: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 1])
    style_ax(ax_sum, "◈  ISOTOPE EFFECT — RESULTS")
    ax_sum.axis("off")

    pred = metrics['prediction']
    lines = [
        f"D₂O ISOTOPE EFFECT ON ENAQT",
        f"",
        f"MASS RATIO:    m_D / m_H = {phys['mass_ratio_DH']:.3f}",
        f"FREQ SCALE:    ω_D / ω_H = {phys['freq_scale_D2O']:.4f}",
        f"",
        f"AT 310 K:",
        f"  γ(H₂O) = {phys['gamma_body_H2O']:.2f} cm⁻¹",
        f"  γ(D₂O) = {phys['gamma_body_D2O']:.2f} cm⁻¹",
        f"  ΔP₄    = {bt['delta_P4']:+.4f} ({bt['pct_change']:+.1f}%)",
        f"",
        f"PREDICTION:",
        f"  D₂O reduces dephasing → moves",
        f"  biology away from ENAQT optimum",
        f"  MC change @ 100% D₂O: "
        f"{pred['MC_change_at_100pct_D2O']:+.1f}%",
        f"",
        f"TESTABLE: Culture organoids in",
        f"  D₂O/H₂O mixtures, measure MC",
    ]

    text = "\n".join(lines)
    ax_sum.text(0.05, 0.95, text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "EXPERIMENT 3a ── D₂O ISOTOPE EFFECT ON ENAQT",
        color=ACCENT_CYAN, fontsize=16, fontfamily=MONO,
        fontweight="bold", y=0.99
    )
    fig.text(0.5, 0.97,
             "Deuterium substitution shifts spectral density → "
             "alters effective dephasing → changes transport efficiency",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"\n  ◎ Dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
