"""
Small LLM wrapper used by the Planner and Critic agents.

Two backends, picked from .env:
  1. NVIDIA Brev (or any OpenAI-compatible server: vLLM, NIM, Ollama...)
       LLM_BASE_URL=http://<brev-ip-or-tunnel>:8000/v1
       LLM_MODEL=Qwen/Qwen2.5-14B-Instruct      # whatever the server is serving
       LLM_API_KEY=...                           # only if you started vLLM with --api-key
  2. Claude, used when LLM_BASE_URL is not set
       ANTHROPIC_API_KEY=...
       CLAUDE_MODEL=claude-sonnet-5-5
With neither set, llm_available() is False and the agents use their rule-based fallback.

Either way we force the model to answer by calling a "tool" whose input schema is
the exact JSON shape we want back, so there is no free text to parse.
"""

import json
import os
import re

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MODEL = CLAUDE_MODEL  # older name, still imported by the orchestrator
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))


def backend() -> str:
    if LLM_BASE_URL:
        return "brev"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return "none"


def llm_available() -> bool:
    return backend() != "none"


def call_structured(system: str, user: str, tool: dict, max_tokens: int = 1500) -> dict:
    """Ask the model a question and get back a dict matching tool['input_schema']."""
    if backend() == "brev":
        return _call_openai_compatible(system, user, tool, max_tokens)
    return _call_claude(system, user, tool, max_tokens)


# ---------------------------------------------------------------- Brev / vLLM

def _call_openai_compatible(system: str, user: str, tool: dict, max_tokens: int) -> dict:
    import httpx

    model = LLM_MODEL or _first_served_model()
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["input_schema"],
            },
        }],
        # Named tool choice: vLLM constrains the output to the schema.
        "tool_choice": {"type": "function", "function": {"name": tool["name"]}},
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    r = httpx.post(f"{LLM_BASE_URL}/chat/completions", json=body, headers=headers, timeout=LLM_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM server returned {r.status_code}: {r.text[:300]}")

    message = r.json()["choices"][0]["message"]
    for call in message.get("tool_calls") or []:
        args = call["function"]["arguments"]
        return json.loads(args) if isinstance(args, str) else args
    # Some servers ignore tool_choice and put the JSON in the text instead.
    return _json_from_text(message.get("content") or "")


def _first_served_model() -> str:
    import httpx

    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    r = httpx.get(f"{LLM_BASE_URL}/models", headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def _json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Model did not return a structured answer: {text[:200]!r}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------- Claude

def _call_claude(system: str, user: str, tool: dict, max_tokens: int) -> dict:
    import anthropic  # imported here so agents without an LLM don't need it

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not return a structured answer")
