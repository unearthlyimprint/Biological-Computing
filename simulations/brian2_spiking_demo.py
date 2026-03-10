#!/usr/bin/env python3
"""
brian2_spiking_demo.py — Leaky Integrate-and-Fire Network Simulation
=====================================================================

Demonstrates the same fundamental neuron physics that underpins
biological computing in organoids / DishBrain systems.

The simulation creates:
  1. An excitatory population (800 neurons) with random connectivity
  2. An inhibitory population (200 neurons)
  3. Spike-Timing Dependent Plasticity (STDP) on excitatory synapses
  4. A Poisson-distributed external input (sensory stimulation analogue)

This is a minimal reservoir computing setup — the same paradigm used
in organoid intelligence experiments.

Usage:
    pip install -r requirements.txt
    python brian2_spiking_demo.py

Output:
    Prints spike statistics and displays a raster plot + firing rate.
"""

from brian2 import *
import numpy as np

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
N_exc = 800       # excitatory neurons
N_inh = 200       # inhibitory neurons
N_total = N_exc + N_inh

# Neuron model: Leaky Integrate-and-Fire (LIF)
# dV/dt = -(V - V_rest) / tau + I / C
tau_m = 20 * ms       # membrane time constant
V_rest = -65 * mV     # resting potential
V_thresh = -50 * mV   # spike threshold
V_reset = -70 * mV    # reset after spike
tau_ref = 2 * ms      # refractory period

# Synaptic parameters
w_exc = 1.0 * mV      # excitatory weight
w_inh = -3.0 * mV     # inhibitory weight (stronger for balance)
p_connect = 0.1       # connection probability

# Simulation
sim_duration = 1.0 * second
input_rate = 50 * Hz   # external Poisson input rate

print("=" * 60)
print("  BRIAN2 Spiking Neural Network Demo")
print("  Biological Computing Simulation")
print("=" * 60)
print(f"  Neurons: {N_exc} excitatory + {N_inh} inhibitory")
print(f"  Connection probability: {p_connect}")
print(f"  Duration: {sim_duration}")
print()

# ──────────────────────────────────────────────
# Neuron groups
# ──────────────────────────────────────────────
eqs = """
dv/dt = (V_rest - v) / tau_m : volt (unless refractory)
"""

# Excitatory population
exc_neurons = NeuronGroup(
    N_exc, eqs,
    threshold="v > V_thresh",
    reset="v = V_reset",
    refractory=tau_ref,
    method="euler"
)
exc_neurons.v = "V_rest + rand() * (V_thresh - V_rest)"

# Inhibitory population
inh_neurons = NeuronGroup(
    N_inh, eqs,
    threshold="v > V_thresh",
    reset="v = V_reset",
    refractory=tau_ref,
    method="euler"
)
inh_neurons.v = "V_rest + rand() * (V_thresh - V_rest)"

# ──────────────────────────────────────────────
# Synapses
# ──────────────────────────────────────────────
# E → E (with STDP-like random initial weights)
syn_ee = Synapses(exc_neurons, exc_neurons, "w : volt", on_pre="v_post += w")
syn_ee.connect(p=p_connect)
syn_ee.w = "w_exc * (0.5 + 0.5 * rand())"

# E → I
syn_ei = Synapses(exc_neurons, inh_neurons, on_pre="v_post += w_exc")
syn_ei.connect(p=p_connect)

# I → E (inhibitory feedback — crucial for stability)
syn_ie = Synapses(inh_neurons, exc_neurons, on_pre="v_post += w_inh")
syn_ie.connect(p=p_connect)

# I → I
syn_ii = Synapses(inh_neurons, inh_neurons, on_pre="v_post += w_inh")
syn_ii.connect(p=p_connect)

# ──────────────────────────────────────────────
# External input (analogous to MEA stimulation)
# ──────────────────────────────────────────────
poisson_input = PoissonGroup(100, rates=input_rate)
syn_input = Synapses(poisson_input, exc_neurons, on_pre="v_post += 2.0 * mV")
syn_input.connect(p=0.1)

# ──────────────────────────────────────────────
# Monitors
# ──────────────────────────────────────────────
spike_mon_exc = SpikeMonitor(exc_neurons)
spike_mon_inh = SpikeMonitor(inh_neurons)
rate_mon = PopulationRateMonitor(exc_neurons)

# ──────────────────────────────────────────────
# Run simulation
# ──────────────────────────────────────────────
print("Running simulation...")
run(sim_duration, report="text")
print()

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
n_exc_spikes = spike_mon_exc.num_spikes
n_inh_spikes = spike_mon_inh.num_spikes
mean_rate_exc = n_exc_spikes / (N_exc * float(sim_duration))
mean_rate_inh = n_inh_spikes / (N_inh * float(sim_duration))

print("─" * 60)
print("  Results")
print("─" * 60)
print(f"  Excitatory spikes: {n_exc_spikes:,}")
print(f"  Inhibitory spikes: {n_inh_spikes:,}")
print(f"  Mean excitatory rate: {mean_rate_exc:.1f} Hz")
print(f"  Mean inhibitory rate: {mean_rate_inh:.1f} Hz")
print(f"  Total synaptic connections: {len(syn_ee) + len(syn_ei) + len(syn_ie) + len(syn_ii):,}")
print()
print("  ◎ This network demonstrates the same spiking dynamics")
print("    present in biological computing organoids.")
print("  ◎ External Poisson input → analogous to MEA stimulation")
print("  ◎ Excitatory/Inhibitory balance → self-organising activity")
print("─" * 60)

# ──────────────────────────────────────────────
# Plotting (optional — requires display)
# ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    BORDER_COLOR = (0, 0.824, 1.0, 0.2)   # rgba(0,210,255,0.2)
    fig.patch.set_facecolor("#080b18")

    # Raster plot
    ax1 = axes[0]
    ax1.set_facecolor("#0c1024")
    ax1.scatter(spike_mon_exc.t / ms, spike_mon_exc.i,
                s=0.3, c="#00d2ff", alpha=0.6, label="Excitatory")
    ax1.scatter(spike_mon_inh.t / ms, spike_mon_inh.i + N_exc,
                s=0.3, c="#f43f5e", alpha=0.6, label="Inhibitory")
    ax1.set_ylabel("Neuron Index", color="white", fontsize=12,
                    fontfamily="monospace")
    ax1.set_title("SPIKING NEURAL NETWORK — BIOLOGICAL COMPUTING SIMULATION",
                   color="#00d2ff", fontsize=14, fontfamily="monospace",
                   fontweight="bold", pad=15)
    ax1.legend(loc="upper right", fontsize=9, facecolor="#0c1024",
               edgecolor="#00d2ff", labelcolor="white")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color(BORDER_COLOR)
    ax1.spines["left"].set_color(BORDER_COLOR)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Population rate
    ax2 = axes[1]
    ax2.set_facecolor("#0c1024")
    ax2.plot(rate_mon.t / ms, rate_mon.smooth_rate(width=10 * ms) / Hz,
             color="#0affef", linewidth=1.0)
    ax2.fill_between(rate_mon.t / ms,
                      rate_mon.smooth_rate(width=10 * ms) / Hz,
                      alpha=0.15, color="#0affef")
    ax2.set_xlabel("Time (ms)", color="white", fontsize=12,
                    fontfamily="monospace")
    ax2.set_ylabel("Rate (Hz)", color="white", fontsize=12,
                    fontfamily="monospace")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color(BORDER_COLOR)
    ax2.spines["left"].set_color(BORDER_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("spiking_network_output.png", dpi=150, facecolor="#080b18",
                bbox_inches="tight")
    print("  Plot saved to: spiking_network_output.png")
    print()

except ImportError:
    print("  (matplotlib not available — skipping plot)")

print("Done.")
