"""
environment/reward_engine.py
Shaped reward function: range -1.0 to +1.0.
Provides per-step reward and cumulative reward_score.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Stage → optimal action mapping
STAGE_OPTIMAL_ACTIONS: Dict[str, str] = {
    "new":       "send_email",
    "contacted": "call_lead",
    "engaged":   "schedule_meeting",
    "hot":       "schedule_meeting",
    "converted": "mark_as_hot",
    "dropped":   "send_email",
}

# Reward values
REWARDS = {
    "correct_action":  +0.50,
    "great_action":    +1.00,   # converts or marks hot when hot
    "neutral_action":  +0.10,
    "wrong_action":    -0.50,
    "ignore_active":   -0.30,   # ignoring an active (engaged/hot) lead
    "scam_penalty":    -1.00,
    "spam_repeat":     -0.30,
    "conversion":      +1.00,   # bonus for conversion
}


@dataclass
class RewardState:
    cumulative: float = 0.0
    action_counts: Dict[str, int] = field(default_factory=dict)

    def add(self, r: float) -> None:
        self.cumulative = round(self.cumulative + r, 4)
        self.cumulative = max(-10.0, min(10.0, self.cumulative))

    def count_action(self, action: str) -> int:
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return self.action_counts[action]


class RewardEngine:
    """
    Calculates per-step reward given action, stage, and context.
    """

    def calculate(
        self,
        action:        str,
        stage:         str,
        intent:        str,
        scam_detected: bool,
        reward_state:  RewardState,
    ) -> Tuple[float, str]:
        """
        Returns (reward: float, message: str).
        Also updates reward_state.cumulative.
        """

        # 1. Scam detected → immediate penalty
        if scam_detected:
            r = REWARDS["scam_penalty"]
            reward_state.add(r)
            return r, "Scam/spam detected — maximum penalty applied."

        # 2. Spam (repeat same action too many times)
        repeat_count = reward_state.count_action(action)
        if repeat_count > 3:
            r = REWARDS["spam_repeat"]
            reward_state.add(r)
            return r, f"Repeated action '{action}' {repeat_count}× — diminishing returns."

        # 3. Conversion bonus
        if stage == "converted":
            r = REWARDS["conversion"]
            reward_state.add(r)
            return r, "🎉 Lead converted — maximum reward!"

        # 4. Check against optimal action
        optimal = STAGE_OPTIMAL_ACTIONS.get(stage, "send_email")

        if action == "ignore_lead" and stage in ("engaged", "hot"):
            r = REWARDS["ignore_active"]
            reward_state.add(r)
            return r, f"Ignoring an active '{stage}' lead is a missed opportunity."

        if action == "mark_as_hot" and stage == "hot":
            r = REWARDS["great_action"]
            reward_state.add(r)
            return r, "Correctly identified and acted on a hot lead."

        if action == optimal:
            # Scale by intent
            base = REWARDS["correct_action"]
            multiplier = {"High": 1.0, "Medium": 0.8, "Low": 0.6}.get(intent, 0.7)
            r = round(base * multiplier, 4)
            reward_state.add(r)
            return r, f"Optimal action '{action}' for stage '{stage}' — intent: {intent}."

        if action in ("send_email", "call_lead", "schedule_meeting"):
            r = REWARDS["neutral_action"]
            reward_state.add(r)
            return r, f"Reasonable action '{action}' but not optimal for stage '{stage}'."

        if action in ("mark_as_cold", "ignore_lead"):
            r = REWARDS["wrong_action"]
            reward_state.add(r)
            return r, f"Action '{action}' is counterproductive for stage '{stage}'."

        # Default neutral
        r = REWARDS["neutral_action"]
        reward_state.add(r)
        return r, f"Action '{action}' processed with neutral reward."