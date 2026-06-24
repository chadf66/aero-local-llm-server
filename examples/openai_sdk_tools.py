"""A full tool-calling loop against aero using the OpenAI Python SDK.

Proves aero is a drop-in tool/agent endpoint: the model asks to call a tool, we
execute it locally, hand the result back, and the model writes the final answer.

Setup:
    aero pull NousResearch/Hermes-2-Pro-Mistral-7B-GGUF Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf
    printf 'from = "Hermes-2-Pro-Mistral-7B.Q4_K_M"\\ntools = true\\n' \\
        > ~/.aero/models/hermes-tools.toml
    aero serve            # in another terminal
    pip install openai    # or: pip install -e ".[examples]"
    python examples/openai_sdk_tools.py
"""

import json

from openai import OpenAI

MODEL = "hermes-tools"
client = OpenAI(base_url="http://127.0.0.1:8317/v1", api_key="not-needed")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> str:
    """A fake tool — pretend this hits a weather API."""
    return json.dumps({"city": city, "temp_f": 72, "conditions": "sunny"})


def main() -> None:
    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]

    # 1) The model decides to call the tool.
    first = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    msg = first.choices[0].message
    print("finish_reason:", first.choices[0].finish_reason)

    if not msg.tool_calls:
        print("Model answered without a tool call:", msg.content)
        return

    messages.append(msg)  # the assistant turn with tool_calls

    # 2) Execute each requested tool and append the results.
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        print(f"-> calling {call.function.name}({args})")
        result = get_weather(**args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # 3) The model writes the final answer using the tool result.
    final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print("\nfinal answer:", final.choices[0].message.content)


if __name__ == "__main__":
    main()
