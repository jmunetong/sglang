"""
Qwen3-14B-FP8: text Q&A (Paris weather) on XPU vs CPU (OpenAI /v1), same prompt style as
``test_deepseek_vl2_small.py``.

Model card: https://huggingface.co/Qwen/Qwen3-14B-FP8

  - XPU test runs when Intel XPU is available.
  - CPU test is opt-in: SGLANG_TEST_QWEN3_14B_FP8_CPU=1

Run from test/srt::

  python3 -m unittest xpu.test_qwen3_14b_fp8.TestQwen314BFP8XPU.test_simple_text_qa
  SGLANG_TEST_QWEN3_14B_FP8_CPU=1 python3 -m unittest xpu.test_qwen3_14b_fp8.TestQwen314BFP8CPU.test_simple_text_qa

Appends to ``qwen3_14b_fp8_comparison.txt`` in this directory.

Tensor dumps (all layers) go under ``debug_tensor_dump_output/qwen3_14b_fp8/``.

Troubleshooting (CPU): exit code -9 is often OOM during load; check ``dmesg``.
Do not set TORCH_LOGS="" for the server child (PyTorch import can break); we only
override dynamo/inductor verbosity below.
"""

from __future__ import annotations

import os
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

MODEL = "Qwen/Qwen3-14B-FP8"

COMPARISON_LOG_PATH = Path(__file__).resolve().parent / "qwen3_14b_fp8_comparison.txt"
DEBUG_TENSOR_DUMP_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "debug_tensor_dump_output" / "qwen3_14b_fp8"
)
LAUNCH_TIMEOUT = 900


def _server_subprocess_env() -> dict:
    return {
        "TORCHDYNAMO_VERBOSE": "0",
        "TORCHINDUCTOR_VERBOSE": "0",
        "TORCH_COMPILE_DEBUG": "0",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    }


def setUpModule():
    DEBUG_TENSOR_DUMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _pt in DEBUG_TENSOR_DUMP_OUTPUT_DIR.rglob("*.pt"):
        _pt.unlink(missing_ok=True)
    COMPARISON_LOG_PATH.write_text(
        "Qwen3-14B-FP8 — device comparison log\n"
        f"Model: {MODEL}\n"
        f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"(Sections appended per test; unittest class order: CPU before XPU.)\n"
        f"{'=' * 80}\n\n",
        encoding="utf-8",
    )


def _append_comparison_log(
    *,
    title: str,
    device_cli: str,
    extra_server_notes: str,
    user_prompt: str,
    response,
) -> None:
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
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


XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "triton",
    "--trust-remote-code",
    "--tp=4",
    "--disable-overlap-schedule",
    "--model-impl",
    "sglang",
    "--debug-tensor-dump-output-folder",
    str(DEBUG_TENSOR_DUMP_OUTPUT_DIR.resolve()),
]

CPU_SERVER_ARGS = [
    "--device",
    "cpu",
    "--trust-remote-code",
    "--model-impl",
    "sglang",
    "--context-length",
    "4096",
    "--debug-tensor-dump-output-folder",
    str(DEBUG_TENSOR_DUMP_OUTPUT_DIR.resolve()),
]

_SIMPLE_QA_PROMPT = (
    "Describe the typical weather in Paris. Describe weather depending on seasons"
)


def _simple_text_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SIMPLE_QA_PROMPT},
            ],
        }
    ]


def _assert_nonempty_qa_response(response):
    assert response.choices[0].message.role == "assistant"
    msg = response.choices[0].message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    combined = f"{text} {reasoning}".strip()
    assert len(combined) > 0
    lower = combined.lower()
    weather_hints = (
        "paris",
        "weather",
        "climate",
        "rain",
        "temperature",
        "mild",
        "temperate",
        "summer",
        "winter",
        "humid",
        "sunny",
        "cloud",
    )
    assert any(
        h in lower for h in weather_hints
    ), f"expected Paris/weather-related terms in answer, got: {combined!r}"
    assert response.usage is not None
    assert response.usage.completion_tokens > 0


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestQwen314BFP8XPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        os.environ["SGLANG_USE_SGL_XPU"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            api_key=cls.api_key,
            other_args=list(XPU_SERVER_ARGS),
            device="cuda",
            env=_server_subprocess_env(),
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_simple_text_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=96,
        )
        _assert_nonempty_qa_response(response)
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (Qwen3-14B-FP8)",
            device_cli="--device xpu",
            extra_server_notes="SGLANG_USE_SGL_XPU=1; see XPU_SERVER_ARGS in test source.",
            user_prompt=_SIMPLE_QA_PROMPT,
            response=response,
        )


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_QWEN3_14B_FP8_CPU") == "1",
    "Set SGLANG_TEST_QWEN3_14B_FP8_CPU=1 to run the CPU variant (slow, high host RAM).",
)
class TestQwen314BFP8CPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            api_key=cls.api_key,
            other_args=list(CPU_SERVER_ARGS),
            device="cpu",
            env=_server_subprocess_env(),
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_simple_text_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=96,
        )
        _assert_nonempty_qa_response(response)
        _append_comparison_log(
            title="OUTPUT FROM --device CPU (Qwen3-14B-FP8)",
            device_cli="--device cpu",
            extra_server_notes="See CPU_SERVER_ARGS in test source.",
            user_prompt=_SIMPLE_QA_PROMPT,
            response=response,
        )


if __name__ == "__main__":
    unittest.main()
