"""
environment/__init__.py

Public API for CRM Lead Scoring OpenEnv environment package.
"""

# ✅ CORRECT IMPORTS (relative imports)
from .environment import CRMEnvironment
from .nlp_analyzer import NLPAnalyzer
from .scam_detector import ScamDetector
from .reward_engine import RewardEngine, RewardState


__version__ = "1.0.0"
__author__ = "CRM OpenEnv Team"


__all__ = [
    "CRMEnvironment",
    "NLPAnalyzer",
    "ScamDetector",
    "RewardEngine",
    "RewardState",
]


def create_env() -> CRMEnvironment:
    return CRMEnvironment()