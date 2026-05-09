#!/usr/bin/env python3
"""
server.py — Biological Computing Simulator Backend
====================================================

FastAPI server that wraps the 11 quantum biology experiments
with REST endpoints for interactive simulation control.

Endpoints:
  GET  /api/experiments        — list all experiments with metadata
  GET  /api/metrics/{exp_id}   — return pre-computed metrics JSON
  POST /api/run/{exp_id}       — run experiment with custom params
  GET  /api/results/{exp_id}/image — serve result PNG dashboard
  GET  /api/status             — system health check
  GET  /                       — serve frontend

Usage:
    cd interface/
    uvicorn server:app --host 0.0.0.0 --port 8420 --reload
"""

import json
import os
import sys
import time
import subprocess
import platform
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────────

INTERFACE_DIR = Path(__file__).parent
PROJECT_ROOT = INTERFACE_DIR.parent
SIMULATIONS_DIR = PROJECT_ROOT / "simulations"
VENV_PYTHON = SIMULATIONS_DIR / ".venv" / "bin" / "python"

# Ensure simulation modules are importable
sys.path.insert(0, str(SIMULATIONS_DIR))

# ──────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────

app = FastAPI(
    title="Biological Computing Simulator",
    description="Interactive quantum biology simulation interface",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────
# Experiment Registry
# ──────────────────────────────────────────────────

EXPERIMENTS = {
    "1a": {
        "id": "1a",
        "title": "Quantum vs Classical Synaptic Delays",
        "phase": 1,
        "category": "Spiking Network",
        "tools": "Brian2",
        "description": "Compares 1000-neuron LIF networks with fixed vs WKB tunnelling delay distributions. Tests whether quantum-modified delays alter firing rate, regularity, and synchrony.",
        "key_result": "Rate +16%, CV_ISI −8% vs. classical fixed delays",
        "script": "experiment_1a_quantum_delays.py",
        "metrics_file": "experiment_1a_metrics.json",
        "image_file": "experiment_1a_results.png",
        "params": {
            "n_exc": {"default": 800, "min": 50, "max": 2000, "step": 50, "unit": "neurons", "label": "Excitatory neurons", "quick": 100},
            "n_inh": {"default": 200, "min": 10, "max": 500, "step": 10, "unit": "neurons", "label": "Inhibitory neurons", "quick": 25},
            "barrier_height_eV": {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05, "unit": "eV", "label": "Barrier height V₀"},
            "particle_energy_eV": {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "unit": "eV", "label": "Particle energy E"},
            "sim_duration_s": {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "unit": "s", "label": "Simulation duration", "quick": 0.5},
        },
    },
    "1b": {
        "id": "1b",
        "title": "Posner ³¹P Spin Dynamics",
        "phase": 1,
        "category": "Quantum Dynamics",
        "tools": "QuTiP",
        "description": "Simulates 6-spin ³¹P nuclear system in a Posner molecule (Ca₉(PO₄)₆). Tracks singlet-state fidelity and coherence survival at biological temperature (310 K).",
        "key_result": "Coherence survives 346 µs at 310 K",
        "script": "experiment_1b_posner_spins.py",
        "metrics_file": "experiment_1b_metrics.json",
        "image_file": "experiment_1b_results.png",
        "params": {
            "temperature_K": {"default": 310, "min": 1, "max": 500, "step": 1, "unit": "K", "label": "Temperature"},
        },
    },
    "1c": {
        "id": "1c",
        "title": "ENAQT in a Model Ion Channel",
        "phase": 1,
        "category": "Quantum Transport",
        "tools": "QuTiP",
        "description": "Environment-Assisted Quantum Transport through a 4-site tight-binding K⁺ channel selectivity filter. Sweeps dephasing rate to find the ENAQT peak.",
        "key_result": "ENAQT peak at γ ≈ 1145 cm⁻¹, 6.7× over coherent limit",
        "script": "experiment_1c_enaqt_ion_channel.py",
        "metrics_file": "experiment_1c_metrics.json",
        "image_file": "experiment_1c_results.png",
        "params": {
            "hopping_cm": {"default": 100.0, "min": 10.0, "max": 500.0, "step": 10.0, "unit": "cm⁻¹", "label": "Hopping integral J"},
            "bridge_height_cm": {"default": 1500.0, "min": 0, "max": 5000.0, "step": 100.0, "unit": "cm⁻¹", "label": "Bridge barrier height"},
            "sink_rate_cm": {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.5, "unit": "cm⁻¹", "label": "Sink rate κ"},
            "transit_time_ps": {"default": 5.0, "min": 1.0, "max": 50.0, "step": 1.0, "unit": "ps", "label": "Transit time"},
            "n_dephasing": {"default": 120, "min": 20, "max": 200, "step": 10, "unit": "points", "label": "Dephasing sweep points", "quick": 30},
        },
    },
    "1d": {
        "id": "1d",
        "title": "Reservoir Noise Comparison",
        "phase": 1,
        "category": "Reservoir Computing",
        "tools": "NumPy",
        "description": "Echo State Network comparing Gaussian vs Cauchy (quantum-like heavy-tailed) noise. Tests whether fat-tailed noise preserves memory capacity at high amplitudes.",
        "key_result": "Quantum (Cauchy) noise preserves MC 5.7× better at high amplitude",
        "script": "experiment_1d_reservoir_noise.py",
        "metrics_file": "experiment_1d_metrics.json",
        "image_file": "experiment_1d_results.png",
        "params": {
            "n_reservoir": {"default": 200, "min": 50, "max": 500, "step": 50, "unit": "neurons", "label": "Reservoir size"},
            "spectral_radius": {"default": 0.9, "min": 0.1, "max": 1.5, "step": 0.05, "unit": "", "label": "Spectral radius ρ"},
            "n_steps": {"default": 5000, "min": 1000, "max": 20000, "step": 1000, "unit": "steps", "label": "Simulation steps", "quick": 2000},
        },
    },
    "1e": {
        "id": "1e",
        "title": "ENAQT-Modulated Reservoir Computing",
        "phase": 1,
        "category": "Bridge Experiment",
        "tools": "QuTiP + ESN",
        "description": "BRIDGE: Uses P₄ from ENAQT model (Exp 1c) as synaptic release probability in reservoir computer. Tests if MC co-peaks with ENAQT.",
        "key_result": "MC peak coincides with ENAQT peak (Spearman ρ = 0.99)",
        "script": "experiment_1e_enaqt_reservoir.py",
        "metrics_file": "experiment_1e_metrics.json",
        "image_file": "experiment_1e_results.png",
        "params": {
            "n_reservoir": {"default": 200, "min": 50, "max": 500, "step": 50, "unit": "neurons", "label": "Reservoir size"},
            "spectral_radius": {"default": 0.9, "min": 0.1, "max": 1.5, "step": 0.05, "unit": "", "label": "Spectral radius ρ"},
            "n_dephasing_sweep": {"default": 40, "min": 10, "max": 100, "step": 5, "unit": "points", "label": "Dephasing sweep points", "quick": 15},
        },
    },
    "2a": {
        "id": "2a",
        "title": "BNN Reservoir Baseline",
        "phase": 2,
        "category": "Biological Neural Network",
        "tools": "CL SDK",
        "description": "Baseline memory capacity measurement on a simulated 64-channel Biological Neural Network (Cortical Labs platform).",
        "key_result": "Baseline MC ≈ 0.003 on Poisson simulator",
        "script": "experiment_2a_bnn_reservoir.py",
        "metrics_file": "experiment_2a_metrics.json",
        "image_file": "experiment_2a_results.png",
        "params": {},
    },
    "2b": {
        "id": "2b",
        "title": "ENAQT-Gated BNN Reservoir",
        "phase": 2,
        "category": "Biological Neural Network",
        "tools": "CL SDK",
        "description": "ENAQT gating applied to BNN reservoir. Reproduces stimulation-scaling effects using quantum-modulated release probabilities.",
        "key_result": "ENAQT gating reproduces stim-scaling on simulated BNN",
        "script": "experiment_2b_enaqt_bnn.py",
        "metrics_file": "experiment_2b_metrics.json",
        "image_file": "experiment_2b_results.png",
        "params": {},
    },
    "2c": {
        "id": "2c",
        "title": "Temperature Sweep Prediction",
        "phase": 2,
        "category": "Biological Neural Network",
        "tools": "CL SDK",
        "description": "Tests falsifiable prediction: non-monotonic MC curve between 25–42 °C if ENAQT is operative in neural hardware.",
        "key_result": "MC in noise floor — designed for real hardware validation",
        "script": "experiment_2c_temperature_sweep.py",
        "metrics_file": "experiment_2c_metrics.json",
        "image_file": "experiment_2c_results.png",
        "params": {},
    },
    "2d": {
        "id": "2d",
        "title": "Closed-Loop Spike-Triggered ENAQT",
        "phase": 2,
        "category": "Biological Neural Network",
        "tools": "CL SDK",
        "description": "Real-time spike-triggered feedback with ENAQT-modulated stimulation. Tests if closed-loop feedback selectively enhances MC at ENAQT-optimal P_rel.",
        "key_result": "Closed-loop enhances MC only at ENAQT-optimal p_rel (Δ=+0.008)",
        "script": "experiment_2d_closed_loop_enaqt.py",
        "metrics_file": "experiment_2d_metrics.json",
        "image_file": "experiment_2d_results.png",
        "params": {},
    },
    "2e": {
        "id": "2e",
        "title": "Corrected ENAQT-Gated BNN",
        "phase": 2,
        "category": "Biological Neural Network",
        "tools": "CL SDK",
        "description": "Re-run of Exp 2b with physics-corrected donor–bridge–acceptor energy landscape ε=[0,15J,15J,0].",
        "key_result": "ENAQT peak at γ ≈ 11J with corrected landscape",
        "script": "experiment_2e_corrected_enaqt_bnn.py",
        "metrics_file": "experiment_2e_metrics.json",
        "image_file": "experiment_2e_results.png",
        "params": {},
    },
    "3a": {
        "id": "3a",
        "title": "D₂O Isotope Effect on ENAQT",
        "phase": 3,
        "category": "Falsifiable Predictions",
        "tools": "QuTiP",
        "description": "Predicts that replacing H₂O with D₂O reduces ENAQT transport by ~32% due to heavier nuclear mass slowing phonon-assisted transport.",
        "key_result": "D₂O reduces P₄ by 32.2% at 310 K",
        "script": "experiment_3a_isotope_effect.py",
        "metrics_file": "experiment_3a_metrics.json",
        "image_file": "experiment_3a_results.png",
        "params": {
            "temperature_K": {"default": 310, "min": 1, "max": 500, "step": 1, "unit": "K", "label": "Temperature"},
        },
    },
    "3b": {
        "id": "3b",
        "title": "Magnetic Field Modulation",
        "phase": 3,
        "category": "Falsifiable Predictions",
        "tools": "NumPy",
        "description": "Tests whether ~1 mT external magnetic field can modulate Posner ³¹P spin coherence and downstream neural dynamics.",
        "key_result": "Uniform B-field: ω_L/J = 345× but Δτ ≈ 0 (important null)",
        "script": "experiment_3b_magnetic_field.py",
        "metrics_file": "experiment_3b_metrics.json",
        "image_file": "experiment_3b_results.png",
        "params": {},
    },
}


# ──────────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    params: dict = {}
    quick_mode: bool = True


# ──────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────

@app.get("/api/experiments")
async def list_experiments():
    """List all experiments with metadata."""
    result = []
    for exp_id, exp in EXPERIMENTS.items():
        has_metrics = (SIMULATIONS_DIR / exp["metrics_file"]).exists()
        has_image = (SIMULATIONS_DIR / exp["image_file"]).exists()
        result.append({
            **{k: v for k, v in exp.items() if k != "script"},
            "has_metrics": has_metrics,
            "has_image": has_image,
        })
    return result


@app.get("/api/metrics/{exp_id}")
async def get_metrics(exp_id: str):
    """Return pre-computed metrics for an experiment."""
    if exp_id not in EXPERIMENTS:
        raise HTTPException(404, f"Unknown experiment: {exp_id}")

    metrics_path = SIMULATIONS_DIR / EXPERIMENTS[exp_id]["metrics_file"]
    if not metrics_path.exists():
        raise HTTPException(404, f"No pre-computed metrics for experiment {exp_id}")

    with open(metrics_path) as f:
        return json.load(f)


@app.get("/api/results/{exp_id}/image")
async def get_result_image(exp_id: str):
    """Serve the result dashboard PNG."""
    if exp_id not in EXPERIMENTS:
        raise HTTPException(404, f"Unknown experiment: {exp_id}")

    img_path = SIMULATIONS_DIR / EXPERIMENTS[exp_id]["image_file"]
    if not img_path.exists():
        raise HTTPException(404, f"No result image for experiment {exp_id}")

    return FileResponse(img_path, media_type="image/png")


@app.post("/api/run/{exp_id}")
async def run_experiment(exp_id: str, req: RunRequest):
    """
    Run an experiment with custom parameters.
    Returns the metrics JSON upon completion.
    """
    if exp_id not in EXPERIMENTS:
        raise HTTPException(404, f"Unknown experiment: {exp_id}")

    exp = EXPERIMENTS[exp_id]
    script_path = SIMULATIONS_DIR / exp["script"]

    if not script_path.exists():
        raise HTTPException(500, f"Script not found: {exp['script']}")

    # For now, run the existing script and return its metrics
    # The script writes to its own metrics file
    python_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    try:
        t0 = time.time()
        result = subprocess.run(
            [python_exec, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(SIMULATIONS_DIR),
        )
        elapsed = time.time() - t0

        metrics_path = SIMULATIONS_DIR / exp["metrics_file"]
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}

        return {
            "status": "success" if result.returncode == 0 else "error",
            "elapsed_s": round(elapsed, 1),
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "metrics": metrics,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Experiment timed out (10 min limit)")
    except Exception as e:
        raise HTTPException(500, f"Execution error: {str(e)}")


@app.post("/api/enaqt/sweep")
async def enaqt_sweep(req: RunRequest):
    """
    Quick ENAQT dephasing sweep — runs the lightweight QuTiP model
    directly (no subprocess) for fast interactive exploration.
    """
    try:
        import numpy as np
        from experiment_1c_enaqt_ion_channel import compute_transport_efficiency

        params = req.params
        n_points = int(params.get("n_points", 30))
        gamma_min = float(params.get("gamma_min", 0.01))
        gamma_max = float(params.get("gamma_max", 1e5))
        bridge_height = float(params.get("bridge_height_cm", 1500.0))

        gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_points)
        site_energies = np.array([0.0, bridge_height, bridge_height, 0.0])

        results = []
        for gamma in gammas:
            eta, _, _, _ = compute_transport_efficiency(
                gamma, site_energies=site_energies)
            results.append({
                "gamma_cm": round(float(gamma), 4),
                "efficiency": round(float(eta), 6),
            })

        # Find peak
        etas = [r["efficiency"] for r in results]
        peak_idx = int(np.argmax(etas))

        return {
            "status": "success",
            "sweep": results,
            "peak": {
                "gamma_cm": results[peak_idx]["gamma_cm"],
                "efficiency": results[peak_idx]["efficiency"],
                "index": peak_idx,
            },
        }

    except ImportError as e:
        raise HTTPException(500, f"Import error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Sweep error: {str(e)}")


@app.post("/api/reservoir/quick")
async def reservoir_quick(req: RunRequest):
    """
    Quick reservoir computing test — runs a small ESN directly
    for interactive parameter exploration.
    """
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        params = req.params
        n_reservoir = int(params.get("n_reservoir", 100))
        spectral_radius = float(params.get("spectral_radius", 0.9))
        p_release = float(params.get("p_release", 0.5))
        leak_rate = float(params.get("leak_rate", 0.3))
        n_steps = int(params.get("n_steps", 2000))
        seed = int(params.get("seed", 42))

        rng = np.random.default_rng(seed)

        # Create reservoir
        W = rng.standard_normal((n_reservoir, n_reservoir))
        mask = rng.random((n_reservoir, n_reservoir)) < 0.1
        W *= mask
        eigvals = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 0:
            W *= spectral_radius / max_eig

        W_in = rng.uniform(-1.5, 1.5, n_reservoir)
        u = rng.uniform(0.0, 1.0, n_steps)

        # Run ESN
        states = np.zeros((n_steps, n_reservoir))
        x = np.zeros(n_reservoir)
        W_nonzero = W != 0

        for t in range(n_steps):
            if p_release < 1.0:
                release_mask = (rng.random(W.shape) < p_release) & W_nonzero
                W_eff = np.where(release_mask, W, 0.0) / max(p_release, 1e-6)
            else:
                W_eff = W
            pre = W_eff @ x + W_in * u[t]
            x = (1 - leak_rate) * x + leak_rate * np.tanh(pre)
            states[t] = x

        # Memory capacity
        washout = 200
        X = states[washout:]
        u_w = u[washout:]
        max_delay = 20
        mc_per_delay = []

        for k in range(1, max_delay + 1):
            target = u_w[:-k]
            X_k = X[k:]
            min_len = min(len(target), len(X_k))
            if min_len < 20:
                mc_per_delay.append(0.0)
                continue
            target, X_k = target[:min_len], X_k[:min_len]
            n_tr = int(min_len * 0.7)
            model = Ridge(alpha=0.01)
            model.fit(X_k[:n_tr], target[:n_tr])
            r2 = r2_score(target[n_tr:], model.predict(X_k[n_tr:]))
            mc_per_delay.append(max(0.0, round(float(r2), 4)))

        mc_total = sum(mc_per_delay)

        return {
            "status": "success",
            "mc_total": round(mc_total, 4),
            "mc_per_delay": mc_per_delay,
            "mean_activation": round(float(np.mean(np.abs(X))), 4),
            "params": {
                "n_reservoir": n_reservoir,
                "spectral_radius": spectral_radius,
                "p_release": p_release,
                "n_steps": n_steps,
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Reservoir error: {str(e)}")


@app.get("/api/status")
async def system_status():
    """System health check."""
    status = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "node": platform.node(),
        "packages": {},
        "simulations_dir": str(SIMULATIONS_DIR),
        "experiments_count": len(EXPERIMENTS),
        "metrics_available": 0,
    }

    # Check installed packages
    for pkg in ["brian2", "qutip", "numpy", "scipy", "matplotlib", "sklearn"]:
        try:
            mod = __import__(pkg)
            status["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            status["packages"][pkg] = "NOT INSTALLED"

    # Count available metrics
    for exp in EXPERIMENTS.values():
        if (SIMULATIONS_DIR / exp["metrics_file"]).exists():
            status["metrics_available"] += 1

    return status


# ──────────────────────────────────────────────────
# Static File Serving
# ──────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    """Return empty 204 to suppress browser favicon 404 noise."""
    from fastapi.responses import Response
    return Response(status_code=204)


# Mount static files for CSS/JS
app.mount("/static", StaticFiles(directory=str(INTERFACE_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8420, reload=True)
