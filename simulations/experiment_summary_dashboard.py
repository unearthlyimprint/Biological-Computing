#!/usr/bin/env python3
"""
experiment_summary_dashboard.py — Cross-Experiment Summary
==========================================================

Aggregates key metrics from all 9 experiments (1a–1e, 2a–2d) into a
single-page Scientific Data Observatory dashboard.

Outputs:
    experiment_summary_dashboard.png
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_1c_enaqt_ion_channel import compute_transport_efficiency


# ══════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════

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


def style_ax(ax, title=""):
    ax.set_facecolor(BG_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, color=ACCENT_CYAN, fontsize=10,
                     fontfamily=MONO, fontweight="bold",
                     loc="left", pad=10)


def load_json(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"experiment_{name}_metrics.json")
    with open(path) as f:
        return json.load(f)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "experiment_summary_dashboard.png")

    # ── Load all metrics ──
    m1b = load_json("1b")
    m1c = load_json("1c")
    m1d = load_json("1d")
    m1e = load_json("1e")
    m2d = load_json("2d")

    # ══════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(BG_PRIMARY)
    gs = GridSpec(3, 3, hspace=0.40, wspace=0.30,
                  left=0.05, right=0.95, top=0.92, bottom=0.05)

    # ── Panel A: Key metric per experiment (hero) ──
    ax_hero = fig.add_subplot(gs[0, :])
    style_ax(ax_hero, "◈  KEY METRICS — ALL EXPERIMENTS")

    experiments = [
        ("1a", "Quantum delays",    "CV_ISI +44%\nFano +89%"),
        ("1b", "Posner spins",      "τ_c = 346 µs\n@ 310 K"),
        ("1c", "ENAQT channel",     "P₄ = 0.418\n6.7× enhance"),
        ("1d", "Reservoir noise",   "QN/CN = 5.7×\n@ σ=2.0"),
        ("1e", "ENAQT reservoir",   "r = 0.98\nMC co-peaks"),
        ("2a", "BNN baseline",      "MC ≈ 0.003\nnull model"),
        ("2b", "ENAQT BNN",         "MC ≈ 0.003\nnull model"),
        ("2c", "Temp. sweep",       "Both flat\nnull model"),
        ("2d", "Closed-loop",       "Δ = +0.008\n@ ENAQT opt"),
    ]

    colors = [ACCENT_CYAN] * 5 + [ACCENT_TEAL] * 4
    x = np.arange(len(experiments))
    scores = [0.44, 0.35, 0.67, 0.57, 0.98, 0.03, 0.03, 0.03, 0.08]
    bars = ax_hero.bar(x, scores, color=colors, alpha=0.8, width=0.7,
                       edgecolor=[(0, 0.82, 1, 0.15)] * 9, linewidth=0.5)

    for i, (eid, name, result) in enumerate(experiments):
        ax_hero.text(i, scores[i] + 0.02, result,
                     ha='center', va='bottom', fontsize=7.5,
                     color=TEXT_PRIMARY, fontfamily=MONO,
                     linespacing=1.4)
        ax_hero.text(i, -0.07, f"{eid}\n{name}",
                     ha='center', va='top', fontsize=7.5,
                     color=TEXT_SECONDARY, fontfamily=MONO,
                     linespacing=1.3)

    ax_hero.set_ylim(-0.14, 1.15)
    ax_hero.set_xlim(-0.6, len(experiments) - 0.4)
    ax_hero.set_ylabel("Effect magnitude (normalised)", color=TEXT_SECONDARY,
                       fontfamily=MONO, fontsize=8)
    ax_hero.set_xticks([])
    ax_hero.axvline(4.5, color=ACCENT_AMBER, linestyle='--', alpha=0.4)
    ax_hero.text(2.0, 1.05, "PHASE 1", ha='center', fontsize=9,
                 color=ACCENT_CYAN, fontfamily=MONO, fontweight='bold',
                 alpha=0.6)
    ax_hero.text(6.5, 1.05, "PHASE 2", ha='center', fontsize=9,
                 color=ACCENT_TEAL, fontfamily=MONO, fontweight='bold',
                 alpha=0.6)

    # ── Panel B: ENAQT P₄ curve (compute from 1c model) ──
    ax_p4 = fig.add_subplot(gs[1, 0])
    style_ax(ax_p4, "◈  ENAQT P₄(γ) — EXP 1c")

    gamma_fine = np.logspace(np.log10(0.01), np.log10(5000), 60)
    p4_fine = [compute_transport_efficiency(g)[0] for g in gamma_fine]
    p4_fine = np.array(p4_fine)

    ax_p4.semilogx(gamma_fine, p4_fine, color=ACCENT_TEAL, linewidth=2,
                   alpha=0.9)
    peak_idx = np.argmax(p4_fine)
    ax_p4.plot(gamma_fine[peak_idx], p4_fine[peak_idx], 'o',
               color=ACCENT_RED, markersize=8, zorder=5,
               label=f'Peak: {p4_fine[peak_idx]:.3f}')
    ax_p4.axvline(215.46, color=ACCENT_AMBER, linestyle='--', alpha=0.5,
                  label='310 K')
    ax_p4.set_xlabel("γ (cm⁻¹)", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=8)
    ax_p4.set_ylabel("P₄", color=TEXT_SECONDARY,
                     fontfamily=MONO, fontsize=8)
    ax_p4.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel C: MC co-peak (from 1e) ──
    ax_copeak = fig.add_subplot(gs[1, 1])
    style_ax(ax_copeak, "◈  MC CO-PEAK — EXP 1e")

    sweep = m1e['dephasing_sweep']
    gamma_1e = np.array(sweep['gamma_values'])
    mc_1e = np.array(sweep['mc_values'])
    p4_1e = np.array(sweep['p4_values'])

    ax_copeak.semilogx(gamma_1e, p4_1e / max(p4_1e), color=ACCENT_TEAL,
                       linewidth=1.5, alpha=0.9, label='P₄ (norm)')
    ax_copeak.semilogx(gamma_1e, mc_1e / max(mc_1e), color=ACCENT_RED,
                       linewidth=1.5, alpha=0.9, label='MC (norm)')
    ax_copeak.set_xlabel("γ (cm⁻¹)", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=8)
    ax_copeak.set_ylabel("Normalised value", color=TEXT_SECONDARY,
                         fontfamily=MONO, fontsize=8)
    ax_copeak.legend(fontsize=7, facecolor=BG_SECONDARY,
                     edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel D: Closed-loop result (2d) ──
    ax_cl = fig.add_subplot(gs[1, 2])
    style_ax(ax_cl, "◈  CLOSED vs OPEN-LOOP — EXP 2d")

    named = m2d['named_conditions']
    names = list(named.keys())
    labels = [named[n]['label'].split('(')[0].strip() for n in names]
    cl_vals = [named[n]['closed_loop']['mc'] for n in names]
    ol_vals = [named[n]['open_loop']['mc'] for n in names]

    x_cl = np.arange(len(names))
    w = 0.35
    ax_cl.bar(x_cl - w/2, cl_vals, w, color=ACCENT_CYAN, alpha=0.8,
              label='Closed')
    ax_cl.bar(x_cl + w/2, ol_vals, w, color=ACCENT_AMBER, alpha=0.8,
              label='Open')
    ax_cl.set_xticks(x_cl)
    ax_cl.set_xticklabels(labels, fontsize=7, color=TEXT_SECONDARY,
                          fontfamily=MONO, rotation=15)
    ax_cl.set_ylabel("MC", color=TEXT_SECONDARY, fontfamily=MONO, fontsize=8)
    ax_cl.legend(fontsize=7, facecolor=BG_SECONDARY,
                 edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel E: Posner coherence (1b) ──
    ax_pos = fig.add_subplot(gs[2, 0])
    style_ax(ax_pos, "◈  POSNER COHERENCE — EXP 1b")

    temp_data = m1b['temperature_sweep']
    temps = [d['temperature_K'] for d in temp_data]
    taus = [d['tau_coherence_s'] * 1e6 for d in temp_data]  # convert to µs
    ax_pos.plot(temps, taus, 'o-', color=ACCENT_AMBER, linewidth=1.5,
                markersize=6, alpha=0.9)
    ax_pos.axvline(310, color=ACCENT_RED, linestyle='--', alpha=0.5,
                   label='Body temp')
    ax_pos.set_xlabel("Temperature (K)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=8)
    ax_pos.set_ylabel("τ_c (µs)", color=TEXT_SECONDARY,
                      fontfamily=MONO, fontsize=8)
    ax_pos.legend(fontsize=7, facecolor=BG_SECONDARY,
                  edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel F: Noise resilience (1d) ──
    ax_noise = fig.add_subplot(gs[2, 1])
    style_ax(ax_noise, "◈  NOISE RESILIENCE — EXP 1d")

    classical_data = m1d['noise_sweep']['classical']
    quantum_data = m1d['noise_sweep']['quantum']
    amps_cl = [d['amp'] for d in classical_data]
    mc_gauss = [d['mc'] for d in classical_data]
    amps_qu = [d['amp'] for d in quantum_data]
    mc_cauchy = [d['mc'] for d in quantum_data]

    ax_noise.plot(amps_cl, mc_gauss, 's-', color=ACCENT_AMBER, linewidth=1.5,
                  markersize=4, alpha=0.9, label='Gaussian')
    ax_noise.plot(amps_qu, mc_cauchy, 'o-', color=ACCENT_CYAN, linewidth=1.5,
                  markersize=4, alpha=0.9, label='Cauchy (quantum)')
    ax_noise.set_xlabel("Noise amplitude σ", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=8)
    ax_noise.set_ylabel("MC", color=TEXT_SECONDARY,
                        fontfamily=MONO, fontsize=8)
    ax_noise.legend(fontsize=7, facecolor=BG_SECONDARY,
                    edgecolor=BORDER, labelcolor=TEXT_PRIMARY)

    # ── Panel G: Summary card ──
    ax_sum = fig.add_subplot(gs[2, 2])
    style_ax(ax_sum, "◈  PROGRAMME SUMMARY")
    ax_sum.axis("off")

    summary_text = (
        "QUANTUM EFFECTS IN BIOLOGICAL COMPUTING\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "PHASE 1 — Abstract models\n"
        "  5 experiments · BRIAN2, QuTiP, ESN\n"
        "  Key: MC co-peaks with ENAQT (r=0.98)\n"
        "\n"
        "PHASE 2 — Biological neural networks\n"
        "  4 experiments · CL SDK simulator\n"
        "  Key: CL feedback enhances MC\n"
        "       only at ENAQT-optimal p_rel\n"
        "\n"
        "TOTAL: 9 experiments\n"
        "DOI: 10.5281/zenodo.19019549\n"
        "\n"
        "STATUS: Ready for CL1 hardware\n"
    )

    ax_sum.text(0.05, 0.95, summary_text, transform=ax_sum.transAxes,
                fontfamily=MONO, fontsize=9.5, color=TEXT_PRIMARY,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=1", facecolor=BG_SECONDARY,
                          edgecolor=BORDER, alpha=0.9))

    fig.suptitle(
        "QUANTUM EFFECTS IN BIOLOGICAL COMPUTING ── PROGRAMME OVERVIEW",
        color=ACCENT_CYAN, fontsize=18, fontfamily=MONO,
        fontweight="bold", y=0.97
    )
    fig.text(0.5, 0.945,
             "9 experiments · 2 phases · Phase 1 (abstract models) + "
             "Phase 2 (biological neural networks)",
             ha="center", color=TEXT_SECONDARY, fontsize=9,
             fontfamily=MONO)

    plt.savefig(output_path, dpi=150, facecolor=BG_PRIMARY,
                bbox_inches="tight")
    print(f"◎ Summary dashboard saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
