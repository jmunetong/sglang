"""
DeepSeek-Coder-V2-Lite-Instruct: simple code Q&A on XPU vs CPU (OpenAI /v1).

Model card: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

  - XPU test runs when Intel XPU is available.
  - CPU test is opt-in: SGLANG_TEST_DSCODER_V2_LITE_CPU=1

Run from test/srt::

  python3 -m unittest xpu.test_deepseek_coder_v2_lite_instruct.TestDeepSeekCoderV2LiteInstructXPU.test_simple_code_qa
  SGLANG_TEST_DSCODER_V2_LITE_CPU=1 python3 -m unittest xpu.test_deepseek_coder_v2_lite_instruct.TestDeepSeekCoderV2LiteInstructCPU.test_simple_code_qa

Appends to ``deepseek_coder_v2_lite_instruct_comparison.txt`` in this directory.

Tensor dumps (all layers) go under ``debug_tensor_dump_output/deepseek_coder_v2_lite_instruct/``.
"""

from __future__ import annotations

import os
import re
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

MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

COMPARISON_LOG_PATH = (
    Path(__file__).resolve().parent / "deepseek_coder_v2_lite_instruct_comparison.txt"
)
DEBUG_TENSOR_DUMP_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "debug_tensor_dump_output"
    / "deepseek_coder_v2_lite_instruct"
)
LAUNCH_TIMEOUT = 900


def _server_subprocess_env() -> dict:
    return {
        "TORCHDYNAMO_VERBOSE": "0",
        "TORCHINDUCTOR_VERBOSE": "0",
        "TORCH_COMPILE_DEBUG": "0",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
    }


def _prettify_spm_style_text(s: str) -> str:
    """Turn SentencePiece-style space/newline markers in API strings into normal text.

    Some stacks surface raw decoded pieces where ``Ġ`` is a space and ``Ċ`` is a newline
    (GPT-2 / Llama-style), so logs look garbled. This does not run a full tokenizer detokenize.
    """
    if not s:
        return s
    return s.replace("Ċ", "\n").replace("Ġ", " ")


def setUpModule():
    DEBUG_TENSOR_DUMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _pt in DEBUG_TENSOR_DUMP_OUTPUT_DIR.rglob("*.pt"):
        _pt.unlink(missing_ok=True)
    COMPARISON_LOG_PATH.write_text(
        "DeepSeek-Coder-V2-Lite-Instruct — device comparison log\n"
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


XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--tp=4",
    "--trust-remote-code",
    "--disable-overlap-schedule",
    "--page-size",
    "32",
    "--attention-backend",
    "triton",
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

_SIMPLE_CODE_PROMPT = (
    "Write a minimal Python function `def add(a, b):` that returns a+b. "
    "Reply with only the function, give a brief explanation. Finish with asking me How can I help you today?"
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
    """Collapse whitespace; normalize SentencePiece-style Ġ/Ċ in some decoders."""
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
class TestDeepSeekCoderV2LiteInstructXPU(CustomTestCase):
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

    def test_simple_code_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=96,
        )
        _assert_code_reply(response)
        _append_comparison_log(
            title="OUTPUT FROM --device XPU (DeepSeek-Coder-V2-Lite-Instruct)",
            device_cli="--device xpu",
            extra_server_notes="SGLANG_USE_SGL_XPU=1; see XPU_SERVER_ARGS in test source.",
            user_prompt=_SIMPLE_CODE_PROMPT,
            response=response,
        )


@unittest.skipUnless(
    os.environ.get("SGLANG_TEST_DSCODER_V2_LITE_CPU") == "1",
    "Set SGLANG_TEST_DSCODER_V2_LITE_CPU=1 to run the CPU variant (slow, high host RAM).",
)
class TestDeepSeekCoderV2LiteInstructCPU(CustomTestCase):
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

    def test_simple_code_qa(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=_simple_text_messages(),
            temperature=0,
            max_tokens=96,
        )
        _assert_code_reply(response)
        _append_comparison_log(
            title="OUTPUT FROM --device CPU (DeepSeek-Coder-V2-Lite-Instruct)",
            device_cli="--device cpu",
            extra_server_notes="See CPU_SERVER_ARGS in test source.",
            user_prompt=_SIMPLE_CODE_PROMPT,
            response=response,
        )


if __name__ == "__main__":
    unittest.main()
