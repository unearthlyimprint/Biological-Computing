# Biological Computing — Organoid Intelligence

Research project exploring wetware computing: living neurons grown in petri dishes as biological CPUs, simulation of spiking neural networks, and the feasibility of quantum effects in warm, wet neuronal systems.

## Repository Structure

```
.
├── docs/                    # Research reports & analysis
│   └── research_report.md   # Comprehensive state-of-the-art review
├── simulations/             # Computational models
│   ├── requirements.txt     # Python dependencies (brian2, etc.)
│   └── brian2_spiking_demo.py  # Leaky integrate-and-fire network
├── notes/                   # Working notes & considerations
│   └── quantum_biology_considerations.md
├── references/              # Links, papers, APIs
│   └── resources.md
└── README.md
```

## Key Players (as of March 2026)

| Organisation | Product | Status |
|---|---|---|
| **Cortical Labs** (AU) | CL1 biological computer, biOS, Cortical Cloud | Commercial — shipping since June 2025 |
| **FinalSpark** (CH) | Neuroplatform (16 organoids) | Research access available |
| **Johns Hopkins** (US) | Organoid Intelligence initiative | Academic — learning/memory demonstrated |
| **UC Santa Cruz** (US) | Cart-pole benchmark with organoids | Academic — real-time processing (2026) |

## Quick Start — Simulation

```bash
cd simulations/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python brian2_spiking_demo.py
```

## Topics Covered

- **Wetware Computing** — how neurons on MEA chips compute
- **Programming Interfaces** — Cortical Labs Python SDK, Cortical Cloud, FinalSpark Neuroplatform
- **Spiking Neural Network Simulation** — BRIAN2, NEST, NEURON
- **Quantum Biology** — decoherence in warm/wet, Posner molecules, Orch-OR debate

## License

Private research — all rights reserved.
