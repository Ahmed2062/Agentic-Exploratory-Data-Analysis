"""Exercises the real ChatOllama client construction in both modes, with only
the network call stubbed. Regression test: text mode passed format=None, which
this langchain-ollama version rejects, so every text call silently fell back."""
from types import SimpleNamespace

from langchain_ollama import ChatOllama

from agents import llm


def test_call_llm_builds_valid_client_in_text_and_json_mode(monkeypatch):
    seen = []

    def fake_invoke(self, messages, *args, **kwargs):
        seen.append(self.format)
        return SimpleNamespace(content="ok")

    monkeypatch.setattr(ChatOllama, "invoke", fake_invoke)
    assert llm.call_llm("sys", "user", json_mode=False) == "ok"
    assert llm.call_llm("sys", "user", json_mode=True) == "ok"
    assert seen == ["", "json"]


def test_safe_text_call_uses_the_model_when_available(monkeypatch):
    monkeypatch.setattr(ChatOllama, "invoke", lambda self, m, *a, **k: SimpleNamespace(content="A real summary."))
    monkeypatch.setattr(llm, "llm_enabled", lambda flag=True: True)
    assert llm.safe_text_call("sys", "user", fallback="TEMPLATE") == "A real summary."
