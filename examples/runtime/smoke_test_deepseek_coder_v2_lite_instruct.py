#!/usr/bin/env python3
"""Smoke test for DeepSeek-Coder-V2-Lite-Instruct on a running SGLang server.

1. Start:
     ./examples/runtime/serve_deepseek_coder_v2_lite_instruct.sh
2. Run:
     python -u examples/runtime/smoke_test_deepseek_coder_v2_lite_instruct.py
   (-u = unbuffered stdout; use if you do not see lines until the script exits)

   DEBUG=1  — print full JSON response to stderr
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000").rstrip("/")
MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _out(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def _err(s: str) -> None:
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a minimal Python function `def add(a, b):` that returns a+b. "
                    "Reply with only the function, no explanation."
                ),
            }
        ],
        "max_tokens": 128,
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        _err(f"HTTP {e.code} from {BASE_URL}/v1/chat/completions")
        _err(body)
        return 1
    except urllib.error.URLError as e:
        _err(f"Request failed: {e}")
        _err(
            f"Is the server running at {BASE_URL}? "
            "Start with: ./examples/runtime/serve_deepseek_coder_v2_lite_instruct.sh"
        )
        return 1

    try:
        data = json.loads(raw.decode())
    except json.JSONDecodeError as e:
        _err(f"Invalid JSON: {e}")
        _err(raw.decode("utf-8", errors="replace")[:2000])
        return 1

    if DEBUG:
        _err(json.dumps(data, indent=2, ensure_ascii=False))

    choices = data.get("choices") or []
    if not choices:
        _err("No choices in response:")
        _err(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
        return 1

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    reasoning = msg.get("reasoning_content")
    finish = choices[0].get("finish_reason")

    _out("--- model reply ---")
    if reasoning:
        _out(reasoning.strip())
    if content:
        _out(content.strip())
    if not (reasoning or content):
        _out("(empty content and reasoning_content)")
        _out(f"finish_reason={finish!r}")
        _err("Full first choice (debug):")
        _err(json.dumps(choices[0], indent=2, ensure_ascii=False))

    usage = data.get("usage")
    if usage:
        _out(f"--- usage: {usage} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
