"""
GPU-Accelerated ENAQT Dephasing Sweep
======================================
Reimplements Experiment 1c (ENAQT in a model ion channel) using PyTorch
on GPU for massively parallel dephasing rate sweeps.

The original experiment used QuTiP on CPU. This version:
- Solves Lindblad master equation on GPU via vectorized matrix operations
- Runs 500+ dephasing rates in parallel (vs ~50 sequential on CPU)
- Computes transport efficiency P_4(t) as function of dephasing rate gamma

Key result to reproduce: ENAQT peak at gamma ≈ 1145 cm^-1

Reference: experiment_1c_enaqt_ion_channel.py (QuTiP version)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SDO Style ---
plt.rcParams.update({
    "figure.facecolor": "#080b18", "axes.facecolor": "#0c1024",
    "axes.edgecolor": "#1a3a4a", "axes.labelcolor": "white",
    "text.color": "white", "xtick.color": "white", "ytick.color": "white",
    "grid.color": "#1a1a2e", "grid.alpha": 0.4,
    "font.family": "monospace", "font.size": 10, "figure.dpi": 150,
})
SDO_CMAP = LinearSegmentedColormap.from_list(
    "sdo", ["#080b18", "#003654", "#00d2ff", "#0affef", "#ffffff"])


# ============================================================================
# LINDBLAD MASTER EQUATION SOLVER (GPU)
# ============================================================================

def build_chain_hamiltonian(n_sites, coupling, site_energies=None, dtype=torch.complex64):
    """Build donor-bridge-acceptor Hamiltonian for ion channel ENAQT.
    Includes an extra trap state (index n_sites) decoupled from H.
    
    Default site energies: [0, 15, 15, 0] (in units of J).
    Bridge height 15J creates tunnelling barrier; dephasing helps overcome it.
    """
    dim = n_sites + 1  # extra trap state
    H = torch.zeros(dim, dim, dtype=dtype, device=DEVICE)
    # Site energies (diagonal)
    if site_energies is None:
        site_energies = [0.0] + [15.0] * (n_sites - 2) + [0.0]  # bridge height = 15J
    for i in range(n_sites):
        H[i, i] = site_energies[i]
    # Nearest-neighbour hopping (off-diagonal)
    for i in range(n_sites - 1):
        H[i, i+1] = coupling
        H[i+1, i] = coupling
    # trap state (index n_sites) has zero coupling — purely absorbing
    return H


def lindblad_step(rho, H, L_ops, dt):
    """
    Single Euler step of Lindblad master equation:
    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    
    Batched over first dimension for parallel dephasing rates.
    """
    n = rho.shape[-1]
    
    # Commutator: -i[H, rho]
    comm = -1j * (H @ rho - rho @ H)
    
    # Dissipator
    dissipator = torch.zeros_like(rho)
    for L in L_ops:
        Ld = L.conj().transpose(-2, -1)
        LdL = Ld @ L
        dissipator += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
    
    return rho + dt * (comm + dissipator)


def run_enaqt_sweep(n_sites=5, n_gammas=500, coupling=100.0, t_max=2.0, dt=0.001):
    """
    Sweep dephasing rate gamma and compute transport efficiency P_4(t_max).
    
    All gamma values run IN PARALLEL on GPU.
    
    P_4 = <4|rho(t)|4> (population at site 4, the target)
    ENAQT prediction: P_4 peaks at intermediate gamma.
    """
    print("\n" + "=" * 60)
    print(f"ENAQT SWEEP: {n_gammas} dephasing rates, {n_sites}-site chain")
    print("=" * 60)
    
    # Dephasing rates to sweep (cm^-1)
    gammas = torch.linspace(0.01, 500.0, n_gammas, device=DEVICE)  # in units of J
    
    dim = n_sites + 1  # include trap state
    
    # Hamiltonian (same for all gammas, trap state decoupled)
    H = build_chain_hamiltonian(n_sites, coupling, dtype=torch.complex64)
    H = H.unsqueeze(0).expand(n_gammas, -1, -1)  # [n_gammas, dim, dim]
    
    # Initial state: excitation at site 0
    rho = torch.zeros(n_gammas, dim, dim, dtype=torch.complex64, device=DEVICE)
    rho[:, 0, 0] = 1.0
    
    n_steps = int(t_max / dt)
    
    # Track P_4 over time for a few representative gammas
    track_indices = [0, n_gammas//4, n_gammas//2, 3*n_gammas//4, n_gammas-1]
    time_traces = {i: [] for i in track_indices}
    
    t0 = time.time()
    # Trapping rate at target site (irreversible sink)
    trap_rate = 0.05  # κ = 0.05J (weak extraction, ratio matches original κ/J)
    trapped_pop = torch.zeros(n_gammas, device=DEVICE)  # accumulated at sink
    
    for step in range(n_steps):
        # Build dephasing Lindblad operators for each gamma (batched)
        L_ops = []
        for site in range(n_sites):  # dephasing on physical sites only
            L = torch.zeros(n_gammas, dim, dim, dtype=torch.complex64, device=DEVICE)
            L[:, site, site] = torch.sqrt(gammas)
            L_ops.append(L)
        
        # Trapping: |trap><target| — irreversibly moves population to external sink
        L_trap = torch.zeros(n_gammas, dim, dim, dtype=torch.complex64, device=DEVICE)
        L_trap[:, n_sites, n_sites-1] = trap_rate ** 0.5  # |trap><target|
        # This is the key: population goes to state n_sites (trap) and never returns
        L_ops.append(L_trap)
        
        # Record trapped population before step
        p_target = rho[:, n_sites-1, n_sites-1].real
        trapped_pop += p_target * trap_rate * dt
        
        rho = lindblad_step(rho, H, L_ops, dt)
        
        # Track specific gammas
        if step % 10 == 0:
            for idx in track_indices:
                p4 = rho[idx, n_sites-1, n_sites-1].real.item()
                time_traces[idx].append(p4)
        
        if step % (n_steps // 5) == 0:
            elapsed = time.time() - t0
            p4_max = rho[:, n_sites-1, n_sites-1].real.max().item()
            print(f"  step {step:5d}/{n_steps} | P4_max={p4_max:.6f} | {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s ({n_gammas * n_steps / elapsed:.0f} evals/s)")
    
    # Transport efficiency = population accumulated in trap state
    P4_final = rho[:, n_sites, n_sites].real.cpu().numpy()  # trap state population
    gammas_np = gammas.cpu().numpy()
    
    # Find ENAQT peak
    peak_idx = np.argmax(P4_final)
    peak_gamma = gammas_np[peak_idx]
    peak_P4 = P4_final[peak_idx]
    
    print(f"\n  ENAQT peak: gamma = {peak_gamma:.1f} cm^-1, P_4 = {peak_P4:.6f}")
    print(f"  Coherent limit (gamma=0): P_4 = {P4_final[0]:.6f}")
    print(f"  Classical limit (gamma=5000): P_4 = {P4_final[-1]:.6f}")
    print(f"  Enhancement: {peak_P4 / max(P4_final[0], 1e-10):.1f}x over coherent")
    
    return {
        "gammas": gammas_np,
        "P4": P4_final,
        "peak_gamma": peak_gamma,
        "peak_P4": peak_P4,
        "time_traces": {gammas_np[i]: np.array(v) for i, v in time_traces.items()},
        "elapsed": elapsed,
        "n_gammas": n_gammas,
    }


def plot_enaqt_results(data):
    """Generate ENAQT dashboard."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ENAQT GPU SWEEP :: LINDBLAD MASTER EQUATION",
                 fontsize=14, fontweight="bold", color="#00d2ff",
                 fontfamily="monospace", y=0.98)
    
    # --- Panel 1: ENAQT Curve ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data["gammas"], data["P4"], color="#00d2ff", linewidth=1.5)
    ax1.axvline(data["peak_gamma"], color="#f59e0b", linestyle="--", alpha=0.7,
                label=f"Peak: γ={data['peak_gamma']:.0f} cm⁻¹")
    ax1.fill_between(data["gammas"], 0, data["P4"], alpha=0.1, color="#00d2ff")
    ax1.set_xlabel("Dephasing Rate γ (cm⁻¹)")
    ax1.set_ylabel("Transport Efficiency P₄")
    ax1.set_title("ENAQT CURVE", color="#00d2ff", fontsize=11, fontfamily="monospace")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # --- Panel 2: Log-scale ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.semilogx(data["gammas"], data["P4"], color="#0affef", linewidth=1.5)
    ax2.axvline(data["peak_gamma"], color="#f59e0b", linestyle="--", alpha=0.7)
    ax2.set_xlabel("γ (cm⁻¹, log scale)")
    ax2.set_ylabel("P₄")
    ax2.set_title("ENAQT (LOG SCALE)", color="#0affef", fontsize=11, fontfamily="monospace")
    ax2.grid(True, alpha=0.2)
    
    # --- Panel 3: Time traces ---
    ax3 = fig.add_subplot(2, 2, 3)
    colors = ["#f43f5e", "#f59e0b", "#10b981", "#00d2ff", "#0affef"]
    for i, (gamma, trace) in enumerate(data["time_traces"].items()):
        t = np.arange(len(trace)) * 0.01  # dt * 10
        ax3.plot(t, trace, color=colors[i % len(colors)], linewidth=1,
                label=f"γ={gamma:.0f}")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("P₄(t)")
    ax3.set_title("TIME EVOLUTION AT SAMPLE γ", color="#10b981",
                  fontsize=11, fontfamily="monospace")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)
    
    # --- Panel 4: Summary stats ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    summary = (
        f"◉ ENAQT SWEEP RESULTS\n\n"
        f"  Sites:           5 (linear chain)\n"
        f"  Coupling:        100 cm⁻¹\n"
        f"  Dephasing rates: {data['n_gammas']}\n"
        f"  GPU time:        {data['elapsed']:.1f}s\n\n"
        f"  ▸ ENAQT peak:    γ = {data['peak_gamma']:.0f} cm⁻¹\n"
        f"  ▸ Peak P₄:       {data['peak_P4']:.6f}\n"
        f"  ▸ Coherent P₄:   {data['P4'][0]:.6f}\n"
        f"  ▸ Classical P₄:  {data['P4'][-1]:.6f}\n"
        f"  ▸ Enhancement:   {data['peak_P4']/max(data['P4'][0],1e-10):.1f}×\n\n"
        f"  MFT Connection:\n"
        f"  Filtering = entropy reduction at\n"
        f"  optimal noise (ENAQT peak)"
    )
    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=10,
             fontfamily="monospace", color="white", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0c1024",
                       edgecolor="#1a3a4a"))
    ax4.set_title("RESULTS SUMMARY", color="#f59e0b", fontsize=11, fontfamily="monospace")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUTPUT_DIR, "enaqt_gpu_sweep_dashboard.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nDashboard saved: {fname}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENAQT GPU SWEEP — Biological Computing Experiment 1c")
    print("=" * 60)
    
    t0 = time.time()
    data = run_enaqt_sweep(n_sites=4, n_gammas=500, coupling=1.0, t_max=20.0, dt=0.002)
    plot_enaqt_results(data)
    
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
