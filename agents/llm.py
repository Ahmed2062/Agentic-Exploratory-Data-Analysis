"""llm.py -- single point of contact with local Ollama.

The LLM is used ONLY for planning suggestions, interpretation, and prose
-- never to compute a number. Every caller passes a deterministic
fallback, so the pipeline degrades instead of crashing when Ollama is
down, slow, disabled, or returns malformed output.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "180"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
LLM_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")  # keep the model loaded between runs

_health_cache: dict[str, float | bool] = {"checked_at": 0.0, "ok": False}


def ollama_available(ttl_s: float = 30.0) -> bool:
    """Cheap reachability check, cached so we don't ping on every call."""
    now = time.time()
    if now - float(_health_cache["checked_at"]) < ttl_s:
        return bool(_health_cache["ok"])
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=1.5) as resp:
            ok = resp.status == 200
    except Exception:  # noqa: BLE001
        ok = False
    _health_cache.update(checked_at=now, ok=ok)
    return ok


def llm_enabled(state_flag: bool | None = True) -> bool:
    if os.getenv("LLM_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return bool(state_flag) and ollama_available()


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False, max_tokens: int | None = None) -> str:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        format="json" if json_mode else "",  # this langchain-ollama version rejects None
        num_predict=max_tokens or LLM_MAX_TOKENS,  # on CPU, output length dominates latency
        keep_alive=LLM_KEEP_ALIVE,
        client_kwargs={"timeout": LLM_TIMEOUT_S},
    )
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.content


def safe_json_call(system_prompt: str, user_prompt: str, fallback: dict, enabled: bool = True) -> dict:
    if not llm_enabled(enabled):
        return fallback
    try:
        raw = call_llm(system_prompt, user_prompt, json_mode=True)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM JSON call failed, using fallback: %s", exc)
        return fallback


def safe_text_call(system_prompt: str, user_prompt: str, fallback: str, enabled: bool = True) -> str:
    if not llm_enabled(enabled):
        return fallback
    try:
        text = call_llm(system_prompt, user_prompt).strip()
        return text or fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM text call failed, using fallback: %s", exc)
        return fallback
