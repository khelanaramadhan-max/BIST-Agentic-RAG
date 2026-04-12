"""
Guardrails – ensures the system never produces investment advice, 
buy/sell signals, or price predictions.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Legal Disclaimer**: This system does not provide investment advice. "
    "Information presented is solely for market intelligence and narrative analysis. "
    "It contains no buy/sell signals or price predictions. "
    "Please consult a licensed financial advisor for investment decisions.\n"
)

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

INVESTMENT_ADVICE_REDIRECT = (
    "This question constitutes a request for investment advice. "
    "This system does not provide investment advice. "
    "However, I can provide objective information derived from KAP disclosures, news, "
    "and research reports regarding the company. "
    "Please rephrase your question to ask about a specific event or theme.\n\n"
    + DISCLAIMER
)


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
    """Make sure the disclaimer is in the response."""
    disclaimer_markers = [
        "yatırım tavsiyesi vermemektedir",
        "does not provide investment advice",
        "This system does not",
    ]
    if not any(marker.lower() in text.lower() for marker in disclaimer_markers):
        return text + DISCLAIMER
    return text


def apply_guardrails(answer: str) -> str:
    """
    Main guardrail function. Performs:
    1. Check for hard violations
    2. Redact specific signals
    3. Ensure disclaimer
    4. Log violations
    """
    if not answer:
        return DISCLAIMER

    is_violating, violations = check_for_investment_advice(answer)

    if violations:
        logger.warning(
            "Guardrail VIOLATION detected: %d patterns matched. Redacting.",
            len(violations),
        )
        # Redact rather than block completely for usability
        answer = redact_sensitive_content(answer)

    # Always ensure disclaimer is present
    answer = ensure_disclaimer_present(answer)

    return answer


def check_question_safety(question: str) -> tuple[bool, str]:
    """
    Pre-check: Is the question explicitly asking for investment advice?
    Returns (is_safe, redirect_message).
    """
    advice_question_patterns = [
        r"\b(should i (buy|sell|invest|hold))\b",
        r"\b(almalı mıyım|satmalı mıyım|yatırım yapmalı mıyım)\b",
        r"\b(what (stock|hisse) (should|to) buy)\b",
        r"\b(hangi hisseyi alayım|ne alayım|ne satalım)\b",
        r"\b(is .+ a good buy|is .+ worth buying)\b",
        r"\b(fiyat tahmin|price predict)\b",
    ]

    for pattern in advice_question_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False, INVESTMENT_ADVICE_REDIRECT

    return True, ""
