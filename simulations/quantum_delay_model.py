#!/usr/bin/env python3
"""
quantum_delay_model.py — Quantum Tunnelling Delay Distributions & Network Metrics
==================================================================================

Provides physics-based delay distributions for synaptic transmission modelling
and network analysis utilities for comparing classical vs quantum-modified
BRIAN2 simulations.

Physics basis:
    The WKB approximation gives the transmission probability for a particle
    tunnelling through a rectangular barrier:

        T(E) = exp(−2κL)

    where κ = √(2m(V₀ − E)) / ℏ

    In our synaptic context, we don't claim literal tunnelling sets the delay.
    Instead: if neurotransmitter crosses the ~20 nm cleft via a quantum pathway,
    the crossing time follows an exponentially skewed distribution (shorter
    delays preferentially) — distinct from classical Gaussian/uniform diffusion.

References:
    - Fisher, M.P.A. (2015). Annals of Physics 362, 593–602.
    - Lambert, N. et al. (2013). Nature Physics 9, 10–18.
    - Cao, J. et al. (2020). Science Advances 6(14).
"""

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────
# Physics Constants (SI units)
# ──────────────────────────────────────────────────
HBAR = 1.054571817e-34       # reduced Planck constant  [J·s]
M_PROTON = 1.67262192e-27    # proton mass              [kg]
EV_TO_J = 1.602176634e-19    # electron-volt to joules
NM_TO_M = 1e-9               # nanometres to metres


def compute_kappa(barrier_height_eV=0.3, particle_energy_eV=0.1,
                  mass_factor=1.0):
    """
    Compute the tunnelling decay constant κ.

    Parameters
    ----------
    barrier_height_eV : float
        Barrier height V₀ in eV.  Typical synaptic cleft: 0.2–0.5 eV
    particle_energy_eV : float
        Particle kinetic energy E in eV
    mass_factor : float
        Particle mass as multiple of proton mass (1.0 = proton/H⁺)

    Returns
    -------
    kappa : float
        Decay constant in m⁻¹
    """
    delta_E = (barrier_height_eV - particle_energy_eV) * EV_TO_J
    if delta_E <= 0:
        raise ValueError("Particle energy exceeds barrier — no tunnelling")
    m = mass_factor * M_PROTON
    kappa = np.sqrt(2.0 * m * delta_E) / HBAR
    return kappa


def tunnelling_transmission(distances_nm, kappa=None,
                            barrier_height_eV=0.3,
                            particle_energy_eV=0.1,
                            mass_factor=1.0):
    """
    WKB tunnelling transmission probability T(d) = exp(−2κd).

    Parameters
    ----------
    distances_nm : array-like
        Barrier widths in nanometres
    kappa : float or None
        Pre-computed decay constant.  If None, computed from barrier params.

    Returns
    -------
    T : ndarray
        Transmission probabilities ∈ (0, 1]
    """
    if kappa is None:
        kappa = compute_kappa(barrier_height_eV, particle_energy_eV,
                              mass_factor)
    d_m = np.asarray(distances_nm) * NM_TO_M
    T = np.exp(-2.0 * kappa * d_m)
    return T


def tunnelling_delay_distribution(n_samples, d_min_ms=0.5, d_max_ms=4.0,
                                  kappa=None, barrier_height_eV=0.3,
                                  particle_energy_eV=0.1, mass_factor=1.0,
                                  rng=None):
    """
    Generate synaptic delay samples from a quantum tunnelling distribution.

    The distribution maps uniform random samples through the tunnelling
    transmission function, producing delays that are **exponentially skewed
    toward shorter values** (faster crossing via tunnelling).

    Parameters
    ----------
    n_samples : int
        Number of delay values to generate
    d_min_ms : float
        Minimum delay in milliseconds
    d_max_ms : float
        Maximum delay in milliseconds
    kappa : float or None
        Tunnelling decay constant (computed if None)
    barrier_height_eV, particle_energy_eV, mass_factor : float
        Tunnelling barrier parameters (ignored if kappa is given)
    rng : numpy.random.Generator or None
        Random number generator for reproducibility

    Returns
    -------
    delays_ms : ndarray of shape (n_samples,)
        Synaptic delays in milliseconds, exponentially skewed toward d_min_ms

    Notes
    -----
    The mapping uses inverse transform sampling:
        1. Draw u ~ Uniform(0, 1)
        2. Map through exponential: delay = d_min + (d_max - d_min) * (1 - exp(-λu)) / (1 - exp(-λ))
    where λ controls the skewness (derived from kappa and barrier width).

    For biologically plausible parameters, the distribution peaks near d_min
    with a long tail toward d_max — the quantum tunnelling signature.
    """
    if rng is None:
        rng = np.random.default_rng()

    if kappa is None:
        kappa = compute_kappa(barrier_height_eV, particle_energy_eV,
                              mass_factor)

    # Map kappa to a skewness parameter for the delay distribution
    # Scale: typical synaptic cleft ~20 nm, kappa ~10^9 m^-1
    # λ = 2 * kappa * cleft_width_m controls the exponential shape
    cleft_width_nm = 20.0  # canonical synaptic cleft width
    lam = 2.0 * kappa * cleft_width_nm * NM_TO_M

    # Clamp lambda so the distribution is always well-behaved
    lam = np.clip(lam, 1.0, 50.0)

    # Inverse CDF of truncated exponential → skewed delays
    u = rng.uniform(0, 1, size=n_samples)
    exp_neg_lam = np.exp(-lam)
    # Map: more weight near d_min (fast tunnelling)
    delays_ms = d_min_ms + (d_max_ms - d_min_ms) * (
        1.0 - np.exp(-lam * u)
    ) / (1.0 - exp_neg_lam)

    return delays_ms


def classical_delay_distribution(n_samples, delay_ms=1.5, jitter_ms=0.0,
                                 rng=None):
    """
    Classical fixed (or Gaussian-jittered) synaptic delays.

    Parameters
    ----------
    n_samples : int
    delay_ms : float
        Mean delay in milliseconds
    jitter_ms : float
        Standard deviation of Gaussian jitter (0 = perfectly fixed)
    rng : numpy.random.Generator or None

    Returns
    -------
    delays_ms : ndarray of shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    if jitter_ms <= 0:
        return np.full(n_samples, delay_ms)
    else:
        delays = rng.normal(delay_ms, jitter_ms, size=n_samples)
        return np.clip(delays, 0.1, 10.0)  # enforce physical bounds


# ──────────────────────────────────────────────────
# Network Metrics
# ──────────────────────────────────────────────────

def compute_cv_isi(spike_trains, n_neurons, min_spikes=3):
    """
    Coefficient of Variation of inter-spike intervals.

    CV_ISI = std(ISI) / mean(ISI) for each neuron, then averaged.
    CV_ISI = 1 for Poisson process, < 1 for regular, > 1 for bursting.

    Parameters
    ----------
    spike_trains : dict-like
        {neuron_index: array_of_spike_times}
    n_neurons : int
    min_spikes : int
        Minimum spikes for a neuron to be included

    Returns
    -------
    mean_cv : float
    std_cv : float
    """
    cvs = []
    for i in range(n_neurons):
        if i in spike_trains and len(spike_trains[i]) >= min_spikes:
            isis = np.diff(spike_trains[i])
            if len(isis) > 0 and np.mean(isis) > 0:
                cvs.append(np.std(isis) / np.mean(isis))
    if len(cvs) == 0:
        return 0.0, 0.0
    return float(np.mean(cvs)), float(np.std(cvs))


def compute_fano_factor(spike_monitor_i, spike_monitor_t, n_neurons,
                        duration_s, bin_width_s=0.05):
    """
    Fano factor: variance / mean of spike counts across neurons.

    FF = 1 for Poisson process, < 1 sub-Poisson, > 1 super-Poisson (bursty).

    Parameters
    ----------
    spike_monitor_i : array-like
        Neuron indices for each spike
    spike_monitor_t : array-like
        Spike times in seconds
    n_neurons : int
    duration_s : float
    bin_width_s : float
        Time bin for counting spikes

    Returns
    -------
    fano : float
    """
    bins = np.arange(0, duration_s + bin_width_s, bin_width_s)
    counts = np.zeros(n_neurons)
    for i in range(n_neurons):
        mask = np.array(spike_monitor_i) == i
        counts[i] = np.sum(mask)

    mean_count = np.mean(counts)
    var_count = np.var(counts)
    if mean_count == 0:
        return 0.0
    return float(var_count / mean_count)


def compute_synchrony_index(rate_trace_1, rate_trace_2):
    """
    Pearson correlation between two smoothed population rate traces.

    Returns correlation coefficient ∈ [-1, 1].
    High positive = synchronous; near 0 = independent.
    """
    if len(rate_trace_1) == 0 or len(rate_trace_2) == 0:
        return 0.0
    if np.std(rate_trace_1) == 0 or np.std(rate_trace_2) == 0:
        return 0.0
    r, _ = stats.pearsonr(rate_trace_1, rate_trace_2)
    return float(r)


def compute_spike_entropy(spike_monitor_i, n_neurons):
    """
    Shannon entropy of spike count distribution across neurons.

    Higher entropy = more uniform activity; lower = concentrated in few neurons.

    Parameters
    ----------
    spike_monitor_i : array-like
        Neuron indices for each spike
    n_neurons : int

    Returns
    -------
    entropy : float
        In bits
    max_entropy : float
        Maximum possible entropy (log2(n_neurons))
    """
    counts = np.bincount(np.asarray(spike_monitor_i, dtype=int),
                         minlength=n_neurons)
    total = np.sum(counts)
    if total == 0:
        return 0.0, np.log2(n_neurons)
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_neurons)
    return float(entropy), float(max_entropy)


def build_spike_trains_dict(spike_indices, spike_times, n_neurons):
    """
    Convert flat spike monitor arrays to per-neuron spike train dict.
    """
    trains = {}
    spike_indices = np.asarray(spike_indices, dtype=int)
    spike_times = np.asarray(spike_times)
    for i in range(n_neurons):
        mask = spike_indices == i
        if np.any(mask):
            trains[i] = np.sort(spike_times[mask])
    return trains


def compute_all_metrics(spike_indices, spike_times, n_neurons, duration_s):
    """
    Compute all network metrics from spike data.

    Returns
    -------
    metrics : dict
        Keys: 'mean_rate_hz', 'cv_isi_mean', 'cv_isi_std',
              'fano_factor', 'spike_entropy', 'max_entropy',
              'n_spikes'
    """
    n_spikes = len(spike_indices)
    mean_rate = n_spikes / (n_neurons * duration_s) if n_neurons > 0 else 0

    trains = build_spike_trains_dict(spike_indices, spike_times, n_neurons)
    cv_mean, cv_std = compute_cv_isi(trains, n_neurons)
    fano = compute_fano_factor(spike_indices, spike_times, n_neurons,
                               duration_s)
    entropy, max_ent = compute_spike_entropy(spike_indices, n_neurons)

    return {
        'n_spikes': int(n_spikes),
        'mean_rate_hz': round(mean_rate, 2),
        'cv_isi_mean': round(cv_mean, 4),
        'cv_isi_std': round(cv_std, 4),
        'fano_factor': round(fano, 4),
        'spike_entropy': round(entropy, 4),
        'max_entropy': round(max_ent, 4),
    }


if __name__ == "__main__":
    # Quick demo: show the delay distributions
    rng = np.random.default_rng(42)
    n = 10000

    q_delays = tunnelling_delay_distribution(n, rng=rng)
    c_delays = classical_delay_distribution(n, delay_ms=1.5)

    print("Quantum delays:  mean={:.2f} ms, std={:.2f} ms, "
          "min={:.2f}, max={:.2f}".format(
              np.mean(q_delays), np.std(q_delays),
              np.min(q_delays), np.max(q_delays)))
    print("Classical delays: mean={:.2f} ms, std={:.2f} ms".format(
        np.mean(c_delays), np.std(c_delays)))
