"""
environment/nlp_analyzer.py
Lightweight NLP analyzer: extracts key_points, intent, entities,
input_summary, and task_complexity from free-text additional_input.
No external ML deps — pure keyword + heuristic approach.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Intent signal dictionaries ─────────────────────────────────

HIGH_INTENT_SIGNALS = [
    "buy", "purchase", "contract", "sign", "deal", "close", "ready",
    "confirm", "approve", "budget approved", "let's go", "finalize",
    "schedule meeting", "demo", "proposal", "pricing", "enterprise",
    "urgent", "asap", "immediately", "today", "this week",
]

MEDIUM_INTENT_SIGNALS = [
    "interested", "maybe", "consider", "thinking", "follow up",
    "more info", "question", "how much", "when", "trial", "test",
    "evaluate", "compare", "options", "discuss", "call me",
]

LOW_INTENT_SIGNALS = [
    "just looking", "not sure", "later", "not now", "no budget",
    "not interested", "unsubscribe", "remove", "stop", "cancel",
    "browsing", "newsletter", "info only",
]

# Entity patterns
ENTITY_PATTERNS = {
    "email":    r"\b[\w.+-]+@[\w-]+\.\w{2,}\b",
    "phone":    r"\b(\+?\d[\d\s\-().]{7,}\d)\b",
    "company":  r"\b([A-Z][a-z]+ (Inc|Ltd|LLC|Corp|Co|Group|Technologies|Solutions)\.?)\b",
    "amount":   r"\$[\d,]+(\.\d{2})?|\b\d[\d,]*\s?(USD|dollars?|k|M)\b",
    "date":     r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
                r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
                r"(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2})\b",
}

COMPLEXITY_RULES = {
    "easy":   ["help", "info", "details", "general", "basic", "what is"],
    "medium": ["compare", "evaluate", "options", "pricing", "trial", "demo"],
    "hard":   ["contract", "enterprise", "multi-year", "customization",
               "integration", "migration", "full deployment"],
}


@dataclass
class NLPResult:
    input_summary:     str
    key_points:        List[str]
    intent:            str        # Low | Medium | High
    entities:          List[str]
    task_complexity:   str        # easy | medium | hard
    complexity_reason: str


class NLPAnalyzer:
    """Analyze free-text CRM interaction input."""

    def analyze(self, text: str) -> NLPResult:
        if not text or not text.strip():
            text = "No additional input provided."

        clean = text.strip()
        lower = clean.lower()

        intent, intent_score  = self._classify_intent(lower)
        key_points             = self._extract_key_points(clean)
        entities               = self._extract_entities(clean)
        task_complexity, reason = self._classify_complexity(lower)
        summary                = self._summarize(clean, intent, key_points)

        return NLPResult(
            input_summary=summary,
            key_points=key_points,
            intent=intent,
            entities=entities,
            task_complexity=task_complexity,
            complexity_reason=reason,
        )

    # ── Private ──────────────────────────────────────────────────

    def _classify_intent(self, text: str) -> Tuple[str, float]:
        high_hits   = sum(1 for s in HIGH_INTENT_SIGNALS   if s in text)
        medium_hits = sum(1 for s in MEDIUM_INTENT_SIGNALS if s in text)
        low_hits    = sum(1 for s in LOW_INTENT_SIGNALS    if s in text)

        if high_hits > medium_hits and high_hits > low_hits:
            return "High", min(0.5 + high_hits * 0.1, 1.0)
        if medium_hits > low_hits:
            return "Medium", min(0.3 + medium_hits * 0.08, 0.8)
        if low_hits > 0:
            return "Low", max(0.1, 0.4 - low_hits * 0.05)
        # Neutral default — short or empty input
        return "Low", 0.20

    def _extract_key_points(self, text: str) -> List[str]:
        """Split into sentences, return up to 4 meaningful ones."""
        sentences = re.split(r"[.!?;\n]+", text)
        points = []
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                points.append(s[:120])  # cap length
            if len(points) >= 4:
                break
        return points if points else ["No specific key points identified."]

    def _extract_entities(self, text: str) -> List[str]:
        found = []
        for label, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                val = m if isinstance(m, str) else m[0]
                found.append(f"{label}: {val.strip()}")
        return found[:6] if found else ["No structured entities detected."]

    def _classify_complexity(self, text: str) -> Tuple[str, str]:
        for level in ["hard", "medium", "easy"]:
            for kw in COMPLEXITY_RULES[level]:
                if kw in text:
                    return level, f"Keyword '{kw}' signals {level}-complexity interaction."
        return "easy", "No strong complexity signals detected; classified as easy."

    def _summarize(self, text: str, intent: str, key_points: List[str]) -> str:
        word_count = len(text.split())
        first_point = key_points[0] if key_points else "general inquiry"
        return (
            f"Lead provided a {word_count}-word message. "
            f"Intent classified as {intent}. "
            f"Primary topic: \"{first_point[:80]}\"."
        )