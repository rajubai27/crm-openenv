"""
environment/scam_detector.py
Detects suspicious / fraudulent lead interactions using keyword patterns
and heuristic scoring. Used for MEDIUM and HARD tasks.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple

# Primary scam keyword list (spec-required)
PRIMARY_SCAM_KEYWORDS = [
    "urgent", "free", "money", "lottery", "win", "prize",
    "congratulations", "winner", "claim now", "click here",
    "wire transfer", "bitcoin", "crypto payment", "inheritance",
    "nigerian prince", "verify your account", "suspended",
    "limited time offer", "act now", "100% guaranteed",
    "no risk", "double your", "make money fast", "earn from home",
    "get rich", "million dollar", "secret formula",
]

# Secondary suspicious patterns (raise risk score but not auto-flag)
SECONDARY_SUSPICIOUS = [
    "western union", "money order", "untraceable", "offshore",
    "tax free", "no questions asked", "confidential deal",
    "advance fee", "processing fee", "bank details",
]


@dataclass
class ScamResult:
    scam_detected:  bool
    scam_keywords:  List[str]
    scam_reason:    str
    risk_score:     float   # 0.0–1.0


class ScamDetector:
    """
    Identifies fake or fraudulent leads.
    Returns a ScamResult with detailed reasoning.
    """

    def detect(self, text: str, lead_history: str = "") -> ScamResult:
        combined = (text + " " + lead_history).lower()

        primary_hits   = self._find_hits(combined, PRIMARY_SCAM_KEYWORDS)
        secondary_hits = self._find_hits(combined, SECONDARY_SUSPICIOUS)

        risk_score = min(
            len(primary_hits) * 0.25 + len(secondary_hits) * 0.10,
            1.0,
        )

        scam_detected = risk_score >= 0.25 or len(primary_hits) >= 1

        reason = self._build_reason(primary_hits, secondary_hits, scam_detected)
        all_hits = list(dict.fromkeys(primary_hits + secondary_hits))  # deduplicated

        return ScamResult(
            scam_detected=scam_detected,
            scam_keywords=all_hits,
            scam_reason=reason,
            risk_score=round(risk_score, 3),
        )

    # ── Private ────────────────────────────────────────────────

    def _find_hits(self, text: str, keyword_list: List[str]) -> List[str]:
        return [kw for kw in keyword_list if kw in text]

    def _build_reason(
        self,
        primary: List[str],
        secondary: List[str],
        detected: bool,
    ) -> str:
        if not detected:
            return "No suspicious patterns detected. Lead appears legitimate."

        parts = []
        if primary:
            kws = ", ".join(f"'{k}'" for k in primary[:5])
            parts.append(f"High-risk keywords found: {kws}")
        if secondary:
            kws = ", ".join(f"'{k}'" for k in secondary[:3])
            parts.append(f"Secondary suspicious terms: {kws}")

        return (
            "⚠️ Potential scam/spam detected. "
            + " | ".join(parts)
            + ". Recommend disqualifying or flagging this lead."
        )