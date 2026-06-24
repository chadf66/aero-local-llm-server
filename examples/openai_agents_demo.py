"""Run the OpenAI Agents SDK against a local aero model.

Proves aero works as the backend for a real agent framework, not just raw SDK
calls: the Agents SDK plans, calls the tool, and produces a final answer — all
against your local Metal-served model.

Setup:
    aero pull NousResearch/Hermes-2-Pro-Mistral-7B-GGUF Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf
    printf 'from = "Hermes-2-Pro-Mistral-7B.Q4_K_M"\\ntools = true\\n' \\
        > ~/.aero/models/hermes-tools.toml
    aero serve                        # in another terminal
    pip install openai-agents         # or: pip install -e ".[examples]"
    python examples/openai_agents_demo.py
"""

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

# Point the Agents SDK at the local server; disable tracing (it would call OpenAI).
client = AsyncOpenAI(base_url="http://127.0.0.1:8317/v1", api_key="not-needed")
set_tracing_disabled(True)
model = OpenAIChatCompletionsModel(model="hermes-tools", openai_client=client)


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"{city}: 72°F and sunny"


agent = Agent(
    name="Weather assistant",
    instructions="You answer weather questions. Use the get_weather tool when asked.",
    model=model,
    tools=[get_weather],
)


def main() -> None:
    result = Runner.run_sync(agent, "What's the weather in San Francisco?")
    print(result.final_output)


if __name__ == "__main__":
    main()
