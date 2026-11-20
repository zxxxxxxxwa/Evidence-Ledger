from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    from .openai_messages_client import Message, build_messages, chat_completion
except ImportError:
    from openai_messages_client import Message, build_messages, chat_completion


BASE_URL = "https://run.v3.cm/v1"
MODEL = "qwen3.7-flash"


def read_text(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --messages-file, --prompt, --prompt-file, or stdin.")


def read_messages(path: str) -> list[Message]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--messages-file must contain an OpenAI-format messages array.")
    return data


def resolve_api_key() -> str:
    key = os.environ.get("V_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set V_API_KEY.")
    return key


def resolve_base_url() -> str:
    return os.environ.get("V_API_BASE_URL") or BASE_URL


def chat(
    messages: Sequence[Message],
    temperature: float = 0.3,
    max_tokens: int = 24000,
    timeout: int = 180,
) -> str:
    return chat_completion(
        base_url=resolve_base_url(),
        api_key=resolve_api_key(),
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_from_prompt(
    prompt: str,
    system: str = "You are a careful research assistant.",
    temperature: float = 0.3,
    max_tokens: int = 24000,
    timeout: int = 180,
) -> str:
    return chat(
        build_messages(prompt=prompt, system=system),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Call qwen3.7-flash with messages.")
    parser.add_argument("--messages-file", help="JSON file containing an OpenAI-format messages array.")
    parser.add_argument("--prompt", help="User prompt text. Compatibility shortcut for one user message.")
    parser.add_argument("--prompt-file", help="UTF-8 text file containing the user prompt.")
    parser.add_argument("--system", default="You are a careful research assistant.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", help="Optional output file.")
    args = parser.parse_args()

    if args.messages_file:
        messages = read_messages(args.messages_file)
    else:
        prompt = args.prompt if args.prompt is not None else read_text(args.prompt_file)
        messages = build_messages(prompt=prompt, system=args.system)

    result = chat(
        messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if args.output:
        Path(args.output).write_text(result.rstrip() + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
