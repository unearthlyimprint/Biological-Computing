# Biological Computing Simulator — Interface

Interactive web interface for simulating and exploring the 11 quantum biology experiments from the _Quantum Effects in Biological Computing_ research programme.

## Quick Start

```bash
# Activate the simulations venv (has Brian2, QuTiP, etc.)
cd simulations/ && source .venv/bin/activate

# Install the web server (one-time)
pip install fastapi uvicorn[standard]

# Start the interface
cd ../interface/
uvicorn server:app --host 0.0.0.0 --port 8420

# Or with hot-reload for development:
python server.py
```

Open **http://localhost:8420** in a browser.

## Architecture

```
interface/
├── server.py      # FastAPI backend — wraps experiments with REST API
├── index.html     # Single-page application shell
├── style.css      # Scientific Data Observatory design system
├── app.js         # Frontend logic (experiment selection, controls, charts)
└── README.md      # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/experiments` | GET | List all 12 experiments with metadata |
| `/api/metrics/{id}` | GET | Pre-computed metrics JSON |
| `/api/results/{id}/image` | GET | Dashboard PNG |
| `/api/run/{id}` | POST | Execute experiment with custom params |
| `/api/enaqt/sweep` | POST | Quick ENAQT dephasing sweep (direct QuTiP) |
| `/api/reservoir/quick` | POST | Quick ESN memory capacity test |
| `/api/status` | GET | System health check |

## Dependencies

The interface uses the same Python venv as the simulations (`simulations/.venv/`), plus:
- `fastapi` — async web framework
- `uvicorn` — ASGI server

No Node.js or frontend build step required. Pure vanilla HTML/CSS/JS.
