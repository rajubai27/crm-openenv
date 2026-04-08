"""
api/server.py — FastAPI REST server for CRM Lead Scoring OpenEnv.

Endpoints:
  POST /reset          → initialize lead episode
  POST /step           → execute action
  GET  /state/{session_id} → get current state
  GET  /actions        → list valid actions
  GET  /health         → health check
  GET  /               → welcome
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from environment.environment import CRMEnvironment

# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="🤖 CRM Lead Scoring OpenEnv API",
    description="""
## AI-Powered CRM Lead Scoring Environment

An **OpenEnv-compatible** Reinforcement Learning environment for CRM lead management.

### Quick Start
1. **POST /reset** — initialize a lead session (choose `easy`, `medium`, or `hard`)
2. **POST /step**  — take an action and receive scored observation + reward
3. **GET  /state/{session_id}** — inspect current session state

### Task Levels
| Task   | Features |
|--------|----------|
| `easy`   | Base lead state + scoring |
| `medium` | + Scam detection |
| `hard`   | + Next best action + Risk flags + Recommended schedule |

### Valid Actions
`call_lead` · `send_email` · `schedule_meeting` · `ignore_lead` · `mark_as_hot` · `mark_as_cold`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
env = CRMEnvironment()


# ── Request / Response models ─────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field(
        default="medium",
        description="Task difficulty level",
        pattern="^(easy|medium|hard)$",
        examples=["easy", "medium", "hard"],
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic lead generation",
        examples=[42],
    )


class StepRequest(BaseModel):
    session_id: str = Field(
        description="Session ID returned by /reset",
        examples=["3f7a9c12-..."],
    )
    action: str = Field(
        description="CRM action to execute",
        examples=["call_lead", "send_email", "schedule_meeting"],
    )
    additional_input: Optional[str] = Field(
        default="",
        description="Free-text from the lead interaction (email, call notes, etc.)",
        examples=["We are ready to sign the contract this week. Budget approved."],
    )


# ── Exception handler ─────────────────────────────────────────

@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name":    "CRM Lead Scoring OpenEnv API",
        "version": "1.0.0",
        "docs":    "/docs",
        "endpoints": {
            "reset":   "POST /reset",
            "step":    "POST /step",
            "state":   "GET  /state/{session_id}",
            "actions": "GET  /actions",
            "health":  "GET  /health",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "active_sessions": len(env._sessions),
    }


@app.get("/actions", tags=["Info"])
def list_actions():
    """List all valid CRM actions."""
    return {
        "actions": [
            {"name": "call_lead",        "description": "Call the lead directly",             "score_delta": +0.12},
            {"name": "send_email",       "description": "Send a follow-up email",             "score_delta": +0.05},
            {"name": "schedule_meeting", "description": "Schedule a discovery/demo meeting",  "score_delta": +0.18},
            {"name": "ignore_lead",      "description": "Take no action this step",           "score_delta": -0.05},
            {"name": "mark_as_hot",      "description": "Flag lead as high-priority hot lead","score_delta": +0.10},
            {"name": "mark_as_cold",     "description": "Downgrade lead to cold/dropped",     "score_delta": -0.30},
        ]
    }


@app.post(
    "/reset",
    tags=["OpenEnv"],
    summary="Initialize a new lead episode",
    response_description="New lead session with initial observation",
)
def reset(req: ResetRequest) -> Dict[str, Any]:
    """
    **Reset** — Start a new CRM lead session.

    - Generates a synthetic lead with realistic initial state
    - Returns `session_id` for use in subsequent `/step` calls
    - Task level controls which extra fields are included in the response
    """
    return env.reset(task=req.task, seed=req.seed)


@app.post(
    "/step",
    tags=["OpenEnv"],
    summary="Execute a CRM action",
    response_description="Updated lead state, reward, and AI analysis",
)
def step(req: StepRequest) -> Dict[str, Any]:
    """
    **Step** — Execute one CRM action and advance the environment.

    Returns:
    - **observation** — updated lead state (score, stage, priority…)
    - **reward** — per-step shaped reward (-1 to +1)
    - **reward_score** — cumulative episode reward
    - **analysis** — NLP analysis of `additional_input`
    - **scam detection** — (medium & hard tasks)
    - **next best action** — (hard task only)
    """
    return env.step(
        session_id=req.session_id,
        action=req.action,
        additional_input=req.additional_input or "",
    )


@app.get(
    "/state/{session_id}",
    tags=["OpenEnv"],
    summary="Get current session state",
)
def state(session_id: str) -> Dict[str, Any]:
    """
    **State** — Retrieve the current state of a session without advancing it.
    """
    return env.state(session_id)