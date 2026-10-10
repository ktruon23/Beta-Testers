"""
Small wrapper around the Claude API used by the Critic and Coach agents.

We force Claude to answer by calling a "tool" whose input schema is the exact
JSON shape we want back. That's the most reliable way to get structured
output: no parsing free text, no "here's your JSON!" preambles.
"""

import os

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def call_structured(system: str, user: str, tool: dict, max_tokens: int = 1500) -> dict:
    """Ask Claude a question and get back a dict matching tool['input_schema']."""
    import anthropic  # imported here so agents without an LLM don't need it

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
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
