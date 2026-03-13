# Quantum Effects in Biological Computing

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white) ![BRIAN2](https://img.shields.io/badge/BRIAN2-2.x-4B8BBE) ![QuTiP](https://img.shields.io/badge/QuTiP-5.x-E34F26) ![License MIT](https://img.shields.io/badge/License-MIT-10b981)

Computational investigation of whether quantum-mechanical effects—tunnelling delays, nuclear-spin coherence, and environment-assisted quantum transport (ENAQT)—can enhance neural computation. Five experiments trace a quantitative pathway from sub-molecular quantum dynamics to network-level computational capacity.

**Preprint:** [manuscript/manuscript.pdf](manuscript/manuscript.pdf)

---

## Key Finding

When the ENAQT transport efficiency of a model ion channel is used as the synaptic release probability in a reservoir computing network, the network's memory capacity **co-peaks** with the ENAQT curve (Pearson *r* = 0.98). Quantum-enhanced molecular transport directly translates into quantum-enhanced network computation.

---

## Experiments

| # | Title | Tools | Key Result |
|---|-------|-------|------------|
| **1a** | Quantum-modified synaptic delays | BRIAN2 | CV_ISI +44%, Fano factor +89% vs. classical fixed delays |
| **1b** | Posner ³¹P spin dynamics | QuTiP | Coherence survives 346 µs at body temperature (310 K) |
| **1c** | ENAQT in a model ion channel | QuTiP | ENAQT peak at γ ≈ 1145 cm⁻¹, 6.7× over coherent limit |
| **1d** | Reservoir computing: quantum vs. classical noise | NumPy | Quantum (Cauchy) noise preserves MC 5.7× better at high amplitude |
| **1e** | ENAQT-modulated reservoir computing | QuTiP + ESN | MC peak coincides with ENAQT peak across dephasing sweep |
| **2a** | BNN reservoir computing baseline | CL SDK | Baseline MC ≈ 0.003 on Poisson simulator (64-channel BNN) |
| **2b** | ENAQT-gated BNN reservoir | CL SDK | ENAQT gating reproduces stim-scaling on simulated BNN |
| **2c** | Temperature sweep prediction test | CL SDK | MC in noise floor for both ENAQT and classical on simulator — designed for real hardware validation |

---

## Repository Structure

```
.
├── manuscript/
│   ├── manuscript.tex          # LaTeX source (bioRxiv preprint)
│   ├── manuscript.pdf          # Compiled PDF
│   └── references.bib          # BibTeX bibliography
├── simulations/
│   ├── experiment_1a_quantum_delays.py
│   ├── experiment_1b_posner_spins.py
│   ├── experiment_1c_enaqt_ion_channel.py
│   ├── experiment_1d_reservoir_noise.py
│   ├── experiment_1e_enaqt_reservoir.py
│   ├── experiment_2a_bnn_baseline.py
│   ├── experiment_2b_enaqt_bnn.py
│   ├── experiment_2c_temperature_sweep.py
│   ├── quantum_delay_model.py  # WKB tunnelling delay model
│   ├── brian2_spiking_demo.py  # Introductory LIF demo
│   ├── experiment_*_results.png  # Dashboard figures
│   ├── experiment_*_metrics.json # Machine-readable results
│   └── requirements.txt
├── docs/
│   ├── research_report.md      # Literature review & background
│   └── phase2_cl_sdk_plan.md   # Phase 2 experiment plan
├── notes/
│   └── quantum_biology_considerations.md
└── references/
    └── resources.md
```

---

## Quick Start

```bash
cd simulations/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run any experiment (each is self-contained)
python experiment_1a_quantum_delays.py
python experiment_1b_posner_spins.py
python experiment_1c_enaqt_ion_channel.py
python experiment_1d_reservoir_noise.py
python experiment_1e_enaqt_reservoir.py
```

**Dependencies:** Python 3.11+, BRIAN2, QuTiP, NumPy, SciPy, Matplotlib, CL SDK (`pip install cl-sdk`)

---

## Falsifiable Predictions

The model generates three predictions testable on organoid platforms (Cortical Labs, FinalSpark):

1. **Isotope effect** — replacing H₂O with D₂O should shift computational capacity
2. **Temperature sweep** — non-monotonic MC curve between 25–42 °C if ENAQT is operative
3. **Magnetic field** — ~1 mT external field should modulate Posner coherence and network dynamics

---

## Citation

If you use this work, please cite:

```bibtex
@article{Arda2026quantum,
  author  = {Arda, Celal},
  title   = {Quantum Effects in Biological Computing: A Computational
             Investigation of Neural Transport, Coherence, and
             Reservoir Capacity},
  year    = {2026},
  note    = {Preprint}
}
```

---

## License

[MIT License](LICENSE) — see [CITATION.cff](CITATION.cff) for citation metadata.
