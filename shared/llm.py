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

def run_agent(
    system: str,
    user: str,
    tools: list[dict],
    execute,
    finish_tool: str,
    max_turns: int = 10,
    max_tokens: int = 2000,
    verbose: bool = True,
) -> dict:
    """Tool-use loop: Claude picks tools, we run them, until it calls `finish_tool`.

    execute(name, args) -> str   runs one tool and returns its result as text.
    The input of the finish_tool call is returned (it is the agent's final answer).
    """
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice={"type": "any"},  # must call a tool, but WHICH one is Claude's call
        )
        messages.append({"role": "assistant", "content": response.content})

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"[agent turn {turn}] {block.name}({block.input})")
            if block.name == finish_tool:
                return block.input  # Claude decided it is done

            result = {"type": "tool_result", "tool_use_id": block.id}
            try:
                result["content"] = execute(block.name, block.input)
            except Exception as e:  # report the error to Claude so it can recover
                result["content"] = f"Error: {type(e).__name__}: {e}"
                result["is_error"] = True
            results.append(result)

        if not results:
            raise RuntimeError(f"Turn {turn}: no tool call (stop_reason={response.stop_reason})")
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")
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
