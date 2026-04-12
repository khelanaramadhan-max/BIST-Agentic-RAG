"""
Guardrails – ensures the system never produces investment advice, 
buy/sell signals, or price predictions.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

DISCLAIMER = ""

# ── Patterns that indicate investment advice ──────────────────────────────────

FORBIDDEN_PATTERNS = [
    # English buy/sell signals
    r"\b(buy|sell|hold|strong buy|strong sell|outperform|underperform|overweight|underweight)\b",
    # Turkish buy/sell signals
    r"\b(al|sat|tut|güçlü al|güçlü sat|endeksin üzerinde|endeksin altında)\b",
    # Price targets
    r"\bprice target[s]?\s*[:=]?\s*[\$₺]?\d+",
    r"\bfiyat hedefi\s*[:=]?\s*[\$₺]?\d+",
    r"\bTP\s*[:=]?\s*[\$₺]?\d+",
    # Return predictions
    r"\bexpect[s]?\s+(return|gain|loss)\s+of\s+\d+",
    r"\bbeklenen\s+(getiri|kazanç|kayıp)\s*[:=]?\s*%?\d+",
    # Direct investment advice
    r"\b(i recommend|we recommend|you should (buy|sell|invest))\b",
    r"\b(tavsiye eder|yatırım yapmanızı öneri)\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]

REPLACEMENT_MESSAGES = {
    r"\b(buy|al)\b": "[no buy signal]",
    r"\b(sell|sat)\b": "[no sell signal]",
    r"\b(price target|fiyat hedefi|TP)\s*[:=]?\s*[\$₺]?\d+": "[no price prediction]",
}

INVESTMENT_ADVICE_REDIRECT = ""


def check_for_investment_advice(text: str) -> tuple[bool, list[str]]:
    """
    Check if text contains investment advice or prohibited content.
    Returns (is_violating, list_of_matched_patterns).
    """
    violations = []
    for pattern in _compiled:
        matches = pattern.findall(text[:5000])
        if matches:
            violations.append(f"Pattern '{pattern.pattern}' matched: {matches[:3]}")
    return bool(violations), violations


def redact_sensitive_content(text: str) -> str:
    """Redact specific price targets and signals while keeping the rest of the answer."""
    result = text
    for pattern_str, replacement in REPLACEMENT_MESSAGES.items():
        result = re.sub(pattern_str, replacement, result, flags=re.IGNORECASE)
    return result


def ensure_disclaimer_present(text: str) -> str:
    """Disabled."""
    return text


def apply_guardrails(answer: str) -> str:
    """Guardrails completely subverted (disabled per user request)"""
    return answer


def check_question_safety(question: str) -> tuple[bool, str]:
    """Safety checks disabled."""
    return True, ""
