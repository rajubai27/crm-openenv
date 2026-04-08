"""
models/schemas.py — All Pydantic request/response models.
Typed, validated, OpenEnv-compliant.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Shared sub-models
# ──────────────────────────────────────────────────────────────

class ObservationModel(BaseModel):
    lead_id:           str
    score:             float = Field(ge=0.0, le=1.0)
    priority:          str   # low | medium | high | urgent
    stage:             str   # new | contacted | engaged | hot | converted | dropped
    interaction_count: int
    history_summary:   str


class AnalysisModel(BaseModel):
    key_points: List[str]
    intent:     str       # Low | Medium | High
    entities:   List[str]


class NextBestAction(BaseModel):
    action:               str
    action_reason:        str
    recommended_schedule: str


# ──────────────────────────────────────────────────────────────
# RESET
# ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    seed: Optional[int] = None


class ResetResponseEasy(BaseModel):
    session_id:   str
    lead_id:      str
    task:         str
    observation:  ObservationModel
    reward:       float = 0.0
    message:      str


class ResetResponseMedium(ResetResponseEasy):
    scam_detected: bool
    scam_reason:   str


class ResetResponseHard(ResetResponseMedium):
    next_best_action:     str
    recommended_schedule: str
    risk_flag:            str


# ──────────────────────────────────────────────────────────────
# STEP
# ──────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    session_id:       str
    action:           str
    additional_input: Optional[str] = ""


class StepResponseEasy(BaseModel):
    observation:      ObservationModel
    lead_score:       float
    reward:           float
    reward_score:     float
    step:             int
    done:             bool
    message:          str
    # analysis block
    input_summary:    str
    task_complexity:  str
    complexity_reason: str
    analysis:         AnalysisModel
    prediction:       str
    confidence_score: float
    status:           str


class StepResponseMedium(StepResponseEasy):
    scam_detected:  bool
    scam_keywords:  List[str]
    scam_reason:    str


class StepResponseHard(StepResponseMedium):
    next_best_action:     str
    action_reason:        str
    recommended_schedule: str
    risk_flag:            str


# ──────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────

class StateResponse(BaseModel):
    session_id:        str
    task:              str
    step:              int
    observation:       ObservationModel
    reward_score:      float
    done:              bool