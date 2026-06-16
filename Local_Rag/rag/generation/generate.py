"""
Ollama streaming chat client.
"""
import json
import sys
from pathlib import Path
from typing import Callable, Iterator

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, GEN_MODEL, GEN_TEMPERATURE, GEN_TOP_P, GEN_NUM_CTX
from generation.prompt import build_messages


def stream_response(
    query: str,
    chunks: list[dict],
    on_token: Callable[[str], None] | None = None,
) -> str:
    """
    Stream an answer from Ollama.
    Calls on_token(token_str) for each token as it arrives.
    Returns the full generated text.
    """
    messages = build_messages(query, chunks)

    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "num_ctx": GEN_NUM_CTX,
        },
    }

    url = f"{OLLAMA_BASE_URL}/api/chat"
    full_text = []

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            token = obj.get("message", {}).get("content", "")
            if token:
                full_text.append(token)
                if on_token:
                    on_token(token)

            if obj.get("done"):
                break

    return "".join(full_text)
