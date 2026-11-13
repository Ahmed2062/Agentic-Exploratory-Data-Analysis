"""verification.py -- deterministic checks on LLM-written text.

1. Numeric grounding: every number the LLM writes must match (within
   rounding) a number that a tool actually computed.
2. Causal language: flags causal claims that correlation/regression
   evidence cannot support, while ignoring negated forms ("does not cause").
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

NUMBER_RE = re.compile(r"(?<![\w.])([<>≤≥]\s*)?(-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?)(\s*%)?")

CAUSAL_RE = re.compile(
    r"\b(causes?|caused by|causing|due to|because of|leads? to|led to|results? in|"
    r"drives?|driven by|driving)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(not|no|n't|never|cannot|without)\b[^.]{0,25}$", re.IGNORECASE)


# Row positions are identifiers, not measurements: counting them as evidence
# would let any integer below the row count pass the grounding check.
SKIP_KEYS = ("indices", "index")


def collect_evidence_numbers(*sources: Any) -> list[float]:
    """Recursively gather every numeric value from tool outputs."""
    found: list[float] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, (int, float)):
            if math.isfinite(float(obj)):
                found.append(float(obj))
        elif isinstance(obj, str):
            # Tools also state numbers in text ("modified Z-score above 3.5").
            # Tool text is trusted evidence; only LLM text is under test.
            for m in NUMBER_RE.finditer(obj):
                try:
                    found.append(float(m.group(2).replace(",", "")))
                except ValueError:
                    pass
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if any(skip in str(k) for skip in SKIP_KEYS):
                    continue
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    for s in sources:
        walk(s)
    return found


def _decimals(token: str) -> int:
    token = token.lower().split("e")[0]
    return len(token.split(".")[1]) if "." in token else 0


def _is_trivial(value: float, token: str) -> bool:
    """Small integers ("top 3") and years are not claims about the data."""
    if "." not in token and "e" not in token.lower():
        if 0 <= abs(value) <= 10:
            return True
        if 1900 <= value <= 2100:
            return True
    return False


def _matches(value: float, token: str, evidence: Iterable[float]) -> bool:
    tol_round = 0.5 * 10 ** (-_decimals(token))
    for ev in evidence:
        for candidate in (ev, ev * 100):  # allow fraction <-> percent
            tol = max(tol_round, 0.005 * abs(candidate))
            if abs(value - candidate) <= tol:
                return True
    return False


def ungrounded_numbers(text: str, evidence: list[float]) -> list[str]:
    bad = []
    for m in NUMBER_RE.finditer(text):
        comparator, token = m.group(1), m.group(2)
        if comparator:  # "p < 0.05" is a threshold, not a computed claim
            continue
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if _is_trivial(value, token):
            continue
        if not _matches(value, token, evidence):
            bad.append(token)
    return bad


def causal_phrases(text: str) -> list[str]:
    hits = []
    for m in CAUSAL_RE.finditer(text):
        prefix = text[max(0, m.start() - 40) : m.start()]
        if NEGATION_RE.search(prefix):
            continue
        hits.append(m.group(0))
    return hits
