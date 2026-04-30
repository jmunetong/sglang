"""
DeepSeek-V4-Flash: simple text Q&A on XPU (OpenAI /v1), same shape as
``test_gemma_4_e2b.py``.

Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

  - XPU test runs when Intel XPU is available.

Run from test/srt::

  python3 -m unittest xpu.jmuneton.test_deepseek_v4.TestDeepSeekV4XPU.test_simple_code_qa

Logs written under ``test_run_logs/deepseek_v4/v<N>/`` where N is the next
free version slot (picked by scanning existing ``test_log_v*.log`` files in
the parent directory). The runner script also emits a top-level
``test_log_v<N>.log`` with the full console output.
  - ``server.stdout.log`` / ``server.stderr.log`` — streamed server output,
    written incrementally so they survive crashes / timeouts during setUp.
  - ``response.txt``                              — the OpenAI response body,
    written from a ``finally`` so it records whatever came back, even on
    assertion failure.
  - ``run.log``                                   — high-level per-test
    timeline (launch args, timings, pass/fail).

The version number comes from ``SGLANG_DSV4_RUN_VERSION`` when set (the
runner script sets this); otherwise the test computes it itself.

The summary file ``deepseek_v4_comparison.txt`` in this directory is still
appended for parity with the Gemma-4 test.

Server is started with ``sglang.launch_server`` (``--model-impl sglang``).

Notes on flags:
  - ``--correctness-test`` is a ``sglang.bench_one_batch`` flag, not a valid
    server CLI flag. Since we run the server and exercise it via the OpenAI
    API (matching the rest of test/srt/xpu), correctness is validated by
    asserting on the response content instead of passing --correctness-test.
"""

from __future__ import annotations

import json
import os
import re
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

import openai

from sglang.srt.utils.common import is_xpu
from sglang.test.test_utils import CustomTestCase
from sglang.test.vlm_utils import (
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash"

_HERE = Path(__file__).resolve().parent
COMPARISON_LOG_PATH = _HERE / "deepseek_v4_comparison.txt"
TEST_RUN_LOGS_ROOT = _HERE / "test_run_logs" / "deepseek_v4"
LAUNCH_TIMEOUT = 1800

# Populated in setUpModule so every artifact from a single invocation of this
# test module lands under one timestamped directory.
_RUN_DIR: Path | None = None
_RUN_LOG_PATH: Path | None = None
_SERVER_STDOUT_PATH: Path | None = None
_SERVER_STDERR_PATH: Path | None = None


def _server_subprocess_env() -> dict:
    # Inherit the parent env so the server subprocess sees PATH /
    # CONDA_PREFIX / PYTHONPATH / ONEAPI_ROOT. Without this, popen passes a
    # 4-key dict as env= and python3 resolves to /usr/bin/python3, which
    # does not have sglang installed in your conda env.
    env = os.environ.copy()
    env.update(
        {
            "TORCHDYNAMO_VERBOSE": "0",
            "TORCHINDUCTOR_VERBOSE": "0",
            "TORCH_COMPILE_DEBUG": "0",
            "TORCH_SHOW_CPP_STACKTRACES": "0",
        }
    )
    return env


def _prettify_spm_style_text(s: str) -> str:
    """Turn SentencePiece-style space/newline markers in API strings into normal text."""
    if not s:
        return s
    return s.replace("Ċ", "\n").replace("Ġ", " ")


def _run_log(msg: str) -> None:
    """Append a timestamped line to run.log. Best-effort; never raises."""
    if _RUN_LOG_PATH is None:
        return
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n"
    try:
        with _RUN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _pick_run_version() -> int:
    """Pick the next free v<N> slot based on existing test_log_v*.log files.

    The runner script writes ``test_log_v<N>.log`` in TEST_RUN_LOGS_ROOT and
    exports ``SGLANG_DSV4_RUN_VERSION=<N>``. If the env var is set we trust
    it; otherwise compute N here so direct ``python3 -m unittest`` also
    produces a correctly-named slot.
    """
    env_v = os.environ.get("SGLANG_DSV4_RUN_VERSION")
    if env_v and env_v.isdigit():
        return int(env_v)

    TEST_RUN_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    highest = 0
    for p in TEST_RUN_LOGS_ROOT.glob("test_log_v*.log"):
        stem = p.stem  # test_log_vN
        try:
            highest = max(highest, int(stem[len("test_log_v") :]))
        except ValueError:
            continue
    for p in TEST_RUN_LOGS_ROOT.glob("v*"):
        if p.is_dir() and p.name.startswith("v"):
            try:
                highest = max(highest, int(p.name[1:]))
            except ValueError:
                continue
    return highest + 1


def setUpModule():
    global _RUN_DIR, _RUN_LOG_PATH, _SERVER_STDOUT_PATH, _SERVER_STDERR_PATH

    version = _pick_run_version()
    _RUN_DIR = TEST_RUN_LOGS_ROOT / f"v{version}"
    _RUN_DIR.mkdir(parents=True, exist_ok=True)

    _RUN_LOG_PATH = _RUN_DIR / "run.log"
    _SERVER_STDOUT_PATH = _RUN_DIR / "server.stdout.log"
    _SERVER_STDERR_PATH = _RUN_DIR / "server.stderr.log"

    COMPARISON_LOG_PATH.write_text(
        "DeepSeek-V4-Flash — device comparison log\n"
        f"Model: {MODEL}\n"
        f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Artifacts: {_RUN_DIR}\n"
        f"{'=' * 80}\n\n",
        encoding="utf-8",
    )
    _run_log(f"setUpModule: artifacts dir = {_RUN_DIR}")


def _append_comparison_log(
    *,
    title: str,
    device_cli: str,
    extra_server_notes: str,
    user_prompt: str,
    response,
) -> None:
    msg = response.choices[0].message
    content = _prettify_spm_style_text(msg.content or "")
    reasoning = _prettify_spm_style_text(getattr(msg, "reasoning_content", None) or "")
    usage = response.usage
    block = (
        f"\n{'#' * 80}\n"
        f"{title}\n"
        f"Server device flag: {device_cli}\n"
        f"{extra_server_notes}\n"
        f"{'#' * 80}\n"
        f"--- user prompt ---\n{user_prompt}\n"
        f"--- assistant message.content ---\n{content}\n"
        f"--- assistant message.reasoning_content (if any) ---\n{reasoning}\n"
        f"--- usage ---\n"
        f"  prompt_tokens: {getattr(usage, 'prompt_tokens', None)}\n"
        f"  completion_tokens: {getattr(usage, 'completion_tokens', None)}\n"
        f"  total_tokens: {getattr(usage, 'total_tokens', None)}\n"
        f"{'=' * 80}\n"
    )
    with COMPARISON_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(block)


def _write_response_txt(
    *,
    test_name: str,
    user_prompt: str,
    response,
    error: BaseException | None,
) -> None:
    """Always-on dump of the raw response (or the error) to response.txt.

    Called from a ``finally`` so the file is produced whether the request
    succeeded, raised, or the assertions on it failed.
    """
    if _RUN_DIR is None:
        return
    path = _RUN_DIR / f"{test_name}.response.txt"
    try:
        lines: list[str] = []
        lines.append(f"Test: {test_name}")
        lines.append(f"UTC:  {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        lines.append("--- user prompt ---")
        lines.append(user_prompt)
        lines.append("")
        if response is not None:
            try:
                msg = response.choices[0].message
                content = _prettify_spm_style_text(msg.content or "")
                reasoning = _prettify_spm_style_text(
                    getattr(msg, "reasoning_content", None) or ""
                )
                lines.append("--- assistant message.content ---")
                lines.append(content)
                lines.append("")
                lines.append("--- assistant message.reasoning_content (if any) ---")
                lines.append(reasoning)
                lines.append("")
                if response.usage is not None:
                    lines.append("--- usage ---")
                    lines.append(
                        f"  prompt_tokens: {getattr(response.usage, 'prompt_tokens', None)}"
                    )
                    lines.append(
                        f"  completion_tokens: {getattr(response.usage, 'completion_tokens', None)}"
                    )
                    lines.append(
                        f"  total_tokens: {getattr(response.usage, 'total_tokens', None)}"
                    )
                    lines.append("")
                try:
                    raw = (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else None
                    )
                    if raw is not None:
                        lines.append("--- raw response (json) ---")
                        lines.append(json.dumps(raw, indent=2, default=str))
                        lines.append("")
                except Exception as dump_err:
                    lines.append(f"[raw dump failed: {dump_err!r}]")
            except Exception as parse_err:
                lines.append(f"[response parse failed: {parse_err!r}]")
                lines.append(repr(response))
        else:
            lines.append("--- no response captured ---")
        if error is not None:
            lines.append("")
            lines.append("--- error ---")
            lines.append(
                "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        _run_log(f"{test_name}: wrote response artifact -> {path.name}")
    except Exception as write_err:
        _run_log(f"{test_name}: failed to write response artifact: {write_err!r}")


# Flags requested for DeepSeek-V4-Flash on XPU (tp=8, compressed attention).
# Rationale:
#   --device xpu:                    required to land on Intel XPU.
#   --trust-remote-code:             DeepSeek-V4 uses a remote model_type.
#   --tp 8:                          8-way tensor parallel.
#   --disable-radix-cache:           radix cache off per user request.
#   --attention-backend compressed:  DeepSeek-V4 compressed MLA backend.
#   --page-size 128:                 matches requested KV page size.
#   --disable-shared-experts-fusion: avoid shared-experts fusion path.
#   --disable-cuda-graph:            no CUDA-graph capture on XPU.
#   --tool-call-parser deepseekv4:   enable V4 tool-call parser.
#   --reasoning-parser deepseek-v4:  surface <think> as reasoning_content.
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--trust-remote-code",
    "--tp=8",
    "--disable-radix-cache",
    "--attention-backend",
    "compressed",
    "--page-size",
    "128",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
    "--model-impl",
    "sglang",
]

_SIMPLE_CODE_PROMPT = (
    "Write a minimal Python function `def add(a, b):` that returns a+b. "
    "Reply with only the function, give a brief explanation. "
    "Finish with asking me How can I help you today?"
)


def _simple_text_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SIMPLE_CODE_PROMPT},
            ],
        }
    ]


def _compact_code_text(s: str) -> str:
    t = s.replace("Ġ", " ").replace("Ċ", "\n")
    return re.sub(r"\s+", "", t.lower())


def _assert_code_reply(response):
    assert response.choices[0].message.role == "assistant"
    msg = response.choices[0].message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    combined = f"{text} {reasoning}".strip()
    assert len(combined) > 0
    lower = combined.lower()
    assert (
        "def" in lower and "add" in lower
    ), f"expected a Python `def add` in reply, got: {combined!r}"
    assert "return" in lower, f"expected `return` in reply, got: {combined!r}"
    compact = _compact_code_text(combined)
    assert (
        "a+b" in compact
    ), f"expected `a+b` (allowing spaces) in reply, got: {combined!r}"
    assert response.usage is not None
    assert response.usage.completion_tokens > 0


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDeepSeekV4XPU(CustomTestCase):
    _server_stdout = None
    _server_stderr = None

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        os.environ["SGLANG_USE_SGL_XPU"] = "1"

        assert _SERVER_STDOUT_PATH is not None and _SERVER_STDERR_PATH is not None
        # Line-buffered so log contents are flushed during launch — crucial
        # if the server crashes before health check passes.
        cls._server_stdout = _SERVER_STDOUT_PATH.open(
            "a", encoding="utf-8", buffering=1
        )
        cls._server_stderr = _SERVER_STDERR_PATH.open(
            "a", encoding="utf-8", buffering=1
        )

        _run_log(f"setUpClass: launching {MODEL} with args={XPU_SERVER_ARGS}")
        _run_log(f"setUpClass: server stdout -> {_SERVER_STDOUT_PATH}")
        _run_log(f"setUpClass: server stderr -> {_SERVER_STDERR_PATH}")

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=LAUNCH_TIMEOUT,
                api_key=cls.api_key,
                other_args=list(XPU_SERVER_ARGS),
                device="cuda",
                env=_server_subprocess_env(),
                return_stdout_stderr=(cls._server_stdout, cls._server_stderr),
            )
        except BaseException as e:
            _run_log(f"setUpClass: server launch failed: {e!r}")
            # Close sinks so whatever was streamed is flushed to disk.
            for fh in (cls._server_stdout, cls._server_stderr):
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            raise
        cls.base_url += "/v1"
        _run_log("setUpClass: server healthy")

    @classmethod
    def tearDownClass(cls):
        try:
            kill_process_tree(cls.process.pid)
            _run_log("tearDownClass: server process tree killed")
        finally:
            for fh in (cls._server_stdout, cls._server_stderr):
                try:
                    if fh is not None:
                        fh.flush()
                        fh.close()
                except Exception:
                    pass

    def test_simple_code_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = None
        error: BaseException | None = None
        _run_log("test_simple_code_qa: sending request")
        try:
            response = client.chat.completions.create(
                model="default",
                messages=_simple_text_messages(),
                temperature=0,
                max_tokens=256,
            )
            _assert_code_reply(response)
            _append_comparison_log(
                title="OUTPUT FROM --device XPU (DeepSeek-V4-Flash)",
                device_cli="--device xpu",
                extra_server_notes=(
                    "SGLANG_USE_SGL_XPU=1; tp=8; attention-backend=compressed; "
                    "page-size=128; radix/cuda-graph/shared-experts-fusion disabled."
                ),
                user_prompt=_SIMPLE_CODE_PROMPT,
                response=response,
            )
            _run_log("test_simple_code_qa: PASS")
        except BaseException as e:
            error = e
            _run_log(f"test_simple_code_qa: FAIL: {e!r}")
            raise
        finally:
            _write_response_txt(
                test_name="test_simple_code_qa",
                user_prompt=_SIMPLE_CODE_PROMPT,
                response=response,
                error=error,
            )


if __name__ == "__main__":
    unittest.main()
