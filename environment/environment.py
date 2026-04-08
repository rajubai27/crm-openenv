"""
environment/environment.py
Core CRM Lead Scoring OpenEnv Environment.
Implements: reset(), step(), state()
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from environment.nlp_analyzer import NLPAnalyzer
from environment.scam_detector import ScamDetector
from environment.reward_engine import RewardEngine, RewardState

# ── Constants ─────────────────────────────────────────────────

VALID_ACTIONS = {
    "call_lead", "send_email", "schedule_meeting",
    "ignore_lead", "mark_as_hot", "mark_as_cold",
}

STAGE_ORDER = ["new", "contacted", "engaged", "hot", "converted", "dropped"]

PRIORITY_MAP = {
    (0.0,  0.30): "low",
    (0.30, 0.55): "medium",
    (0.55, 0.80): "high",
    (0.80, 1.01): "urgent",
}

ACTION_STAGE_ADVANCE = {
    "call_lead":       {"new": "contacted", "contacted": "engaged"},
    "schedule_meeting":{"contacted": "engaged", "engaged": "hot", "hot": "converted"},
    "send_email":      {"new": "contacted"},
    "mark_as_hot":     {"engaged": "hot"},
    "mark_as_cold":    {"new": "dropped", "contacted": "dropped", "engaged": "dropped"},
    "ignore_lead":     {},
}

SCORE_DELTAS = {
    "call_lead":        +0.12,
    "send_email":       +0.05,
    "schedule_meeting": +0.18,
    "mark_as_hot":      +0.10,
    "mark_as_cold":     -0.30,
    "ignore_lead":      -0.05,
}

NEXT_BEST_ACTION_MAP = {
    "Low":    ("send_email",       "Nurturing approach needed — lead is not yet ready."),
    "Medium": ("call_lead",        "Moderate engagement — a personal call can move them forward."),
    "High":   ("schedule_meeting", "Strong buying signal — lock in a meeting immediately."),
}

RISK_FLAGS = {
    "new":       "🟡 New lead — qualification needed.",
    "contacted": "🟡 Early stage — monitor engagement carefully.",
    "engaged":   "🟢 Positive trajectory — keep momentum.",
    "hot":       "🔥 High-value lead — prioritize immediately.",
    "converted": "✅ Converted — ensure smooth onboarding.",
    "dropped":   "🔴 Lead dropped — consider re-engagement campaign.",
}

# Seed interaction history snippets
HISTORY_SEEDS = {
    "easy":   "Lead signed up for newsletter. Visited pricing page.",
    "medium": "Attended webinar, downloaded case study. Replied to cold email.",
    "hard":   "Multiple demo requests, discussed budget & enterprise pricing. "
              "Referred two colleagues. Asked for contract draft.",
}

INITIAL_SCORE_RANGES = {
    "easy":   (0.05, 0.20),
    "medium": (0.20, 0.40),
    "hard":   (0.50, 0.70),
}


# ── Session dataclass ─────────────────────────────────────────

@dataclass
class LeadSession:
    session_id:   str
    lead_id:      str
    task:         str
    score:        float
    stage:        str
    priority:     str
    interaction_count: int
    history_summary:   str
    step_count:   int = 0
    done:         bool = False
    reward_state: RewardState = field(default_factory=RewardState)


# ── Environment ────────────────────────────────────────────────

class CRMEnvironment:
    """
    OpenEnv-compliant CRM Lead Scoring Environment.
    Stateful: sessions stored in-memory dict keyed by session_id.
    """

    def __init__(self):
        self._sessions: Dict[str, LeadSession] = {}
        self._nlp      = NLPAnalyzer()
        self._scam     = ScamDetector()
        self._reward   = RewardEngine()

    # ── reset() ───────────────────────────────────────────────

    def reset(self, task: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)

        task = task.lower()
        lo, hi = INITIAL_SCORE_RANGES.get(task, (0.10, 0.40))
        score    = round(random.uniform(lo, hi), 4)
        priority = self._score_to_priority(score)
        history  = HISTORY_SEEDS.get(task, "No interactions yet.")

        session = LeadSession(
            session_id=str(uuid.uuid4()),
            lead_id=f"LEAD-{uuid.uuid4().hex[:8].upper()}",
            task=task,
            score=score,
            stage="new",
            priority=priority,
            interaction_count=0,
            history_summary=history,
        )
        self._sessions[session.session_id] = session

        obs = self._build_obs(session)
        base = {
            "session_id":  session.session_id,
            "lead_id":     session.lead_id,
            "task":        task,
            "observation": obs,
            "reward":      0.0,
            "message":     f"New lead initialized for task '{task}'. Session ready.",
        }

        if task in ("medium", "hard"):
            scam_result = self._scam.detect(history)
            base["scam_detected"] = scam_result.scam_detected
            base["scam_reason"]   = scam_result.scam_reason

        if task == "hard":
            nba_action, nba_reason = NEXT_BEST_ACTION_MAP["Low"]
            base["next_best_action"]     = nba_action
            base["recommended_schedule"] = "Within 48 hours"
            base["risk_flag"]            = RISK_FLAGS.get("new", "Unknown")

        return base

    # ── step() ────────────────────────────────────────────────

    def step(self, session_id: str, action: str, additional_input: str = "") -> Dict[str, Any]:
        session = self._get_session(session_id)

        if session.done:
            raise RuntimeError("Episode is done. Call reset() to start a new session.")

        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid: {sorted(VALID_ACTIONS)}")

        session.step_count       += 1
        session.interaction_count += 1

        # NLP analysis of additional_input
        nlp = self._nlp.analyze(additional_input)

        # Scam detection (always run internally; exposed based on task)
        scam_result = self._scam.detect(additional_input, session.history_summary)

        # Update score
        delta    = SCORE_DELTAS.get(action, 0.0)
        intent_multiplier = {"High": 1.3, "Medium": 1.0, "Low": 0.7}.get(nlp.intent, 1.0)
        session.score = round(
            max(0.0, min(1.0, session.score + delta * intent_multiplier)), 4
        )

        # Advance stage
        old_stage   = session.stage
        new_stage   = ACTION_STAGE_ADVANCE.get(action, {}).get(old_stage, old_stage)
        if scam_result.scam_detected:
            new_stage = "dropped"
        session.stage    = new_stage
        session.priority = self._score_to_priority(session.score)

        # Update history
        session.history_summary = (
            f"{session.history_summary} | Step {session.step_count}: "
            f"Action='{action}', Intent={nlp.intent}"
        )

        # Reward
        reward, reward_msg = self._reward.calculate(
            action=action,
            stage=new_stage,
            intent=nlp.intent,
            scam_detected=scam_result.scam_detected,
            reward_state=session.reward_state,
        )

        # Done check
        session.done = new_stage in ("converted", "dropped") or session.step_count >= 20

        # Prediction
        confidence, prediction = self._predict(session)

        # Next best action (for HARD)
        intent_key = nlp.intent
        nba_action, nba_reason = NEXT_BEST_ACTION_MAP.get(intent_key, ("send_email", "Continue nurturing."))
        recommended_schedule   = self._recommend_schedule(intent_key, new_stage)

        obs = self._build_obs(session)

        base: Dict[str, Any] = {
            "observation":      obs,
            "lead_score":       session.score,
            "reward":           reward,
            "reward_score":     session.reward_state.cumulative,
            "step":             session.step_count,
            "done":             session.done,
            "message":          reward_msg,
            # analysis block
            "input_summary":    nlp.input_summary,
            "task_complexity":  nlp.task_complexity,
            "complexity_reason": nlp.complexity_reason,
            "analysis": {
                "key_points": nlp.key_points,
                "intent":     nlp.intent,
                "entities":   nlp.entities,
            },
            "prediction":       prediction,
            "confidence_score": confidence,
            "status":           "converted" if session.done and new_stage == "converted"
                                else ("dropped" if session.done else "active"),
        }

        if session.task in ("medium", "hard"):
            base["scam_detected"] = scam_result.scam_detected
            base["scam_keywords"] = scam_result.scam_keywords
            base["scam_reason"]   = scam_result.scam_reason

        if session.task == "hard":
            base["next_best_action"]     = nba_action
            base["action_reason"]        = nba_reason
            base["recommended_schedule"] = recommended_schedule
            base["risk_flag"]            = RISK_FLAGS.get(new_stage, "Unknown")

        return base

    # ── state() ───────────────────────────────────────────────

    def state(self, session_id: str) -> Dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id":   session.session_id,
            "task":         session.task,
            "step":         session.step_count,
            "observation":  self._build_obs(session),
            "reward_score": session.reward_state.cumulative,
            "done":         session.done,
        }

    # ── Helpers ───────────────────────────────────────────────

    def _get_session(self, session_id: str) -> LeadSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")
        return session

    def _build_obs(self, session: LeadSession) -> Dict[str, Any]:
        return {
            "lead_id":           session.lead_id,
            "score":             session.score,
            "priority":          session.priority,
            "stage":             session.stage,
            "interaction_count": session.interaction_count,
            "history_summary":   session.history_summary[-300:],  # cap length
        }

    def _score_to_priority(self, score: float) -> str:
        for (lo, hi), label in PRIORITY_MAP.items():
            if lo <= score < hi:
                return label
        return "low"

    def _predict(self, session: LeadSession) -> Tuple[float, str]:
        score = session.score
        if score >= 0.80:
            return 0.92, "High probability of conversion. Recommend immediate close action."
        if score >= 0.55:
            return 0.70, "Good engagement signals. Nurture toward closing stage."
        if score >= 0.30:
            return 0.45, "Moderate interest detected. Further qualification needed."
        return 0.20, "Low engagement. Lead requires re-activation or disqualification."

    def _recommend_schedule(self, intent: str, stage: str) -> str:
        if intent == "High" or stage in ("hot", "converted"):
            return "Today or within 24 hours"
        if intent == "Medium" or stage == "engaged":
            return "Within 48–72 hours"
        return "Within 5 business days"