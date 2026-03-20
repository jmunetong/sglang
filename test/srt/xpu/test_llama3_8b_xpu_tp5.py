"""
Llama 3 8B on Intel XPU: bfloat16, tensor parallel (default TP from ``SGLANG_TEST_TP_SIZE``).

Run (from repo root, with XPU-enabled SGLang / pyproject_xpu.toml):

    cd test/srt && python3 -m unittest xpu.test_llama3_8b_xpu_tp5.TestLlama38BXPUTP5.test_bench_one_batch
    cd test/srt && python3 -m unittest xpu.test_llama3_8b_xpu_tp5.TestLlama38BXPUQuestionAnswer.test_question_answer

The Q&A test starts ``sglang serve`` and checks completions via HTTP ``/generate`` (not ``bench_one_batch``).

Flash-attention call logging (``sgl_kernel.flash_attn`` — see ``sgl-kernel/.../flash_attn.py``):

    By default this test sets ``SGL_FLASH_ATTN_LOG_FILE`` (and ``SGL_FLASH_ATTN_LOG=1`` if unset) on
    child processes so lines from ``flash_attn_with_kvcache`` / ``flash_attn_varlen_func`` append to
    ``test/srt/xpu/flash_attn.log`` (same directory as this module; not ``$XDG_CACHE_HOME``).
    Each line starts with a header like
    ``iteration=<n>: layer=<L>: decode|prefill|idle: flash_attn_*`` where ``iteration`` advances
    once per full model forward (separate counters for decode vs prefill), and ``layer`` is the
    attention layer index (0 .. num_layers-1). Override the path with ``SGL_FLASH_ATTN_LOG_FILE``.

    Override or disable:

    - ``SGL_FLASH_ATTN_LOG_FILE=/path/to/flash_attn.log`` — explicit log path (wins; passed through).
    - ``SGL_FLASH_ATTN_LOG=0`` — do not enable default file logging from this test.
    - ``SGLANG_SKIP_FLASH_ATTN_FILE_LOG=1`` — same as above, test-specific opt-out.

    Subprocesses use ``sys.executable`` (same interpreter as the test) so they load the same
    ``sglang`` / ``sgl_kernel`` as your venv or editable install—not a different ``python3`` or
    ``sglang`` on PATH. With tensor parallel, workers use the ``spawn`` start method; the runtime
    primes flash-attn file logging in the parent before spawning so the log path exists under Docker
    and CI (see ``maybe_init_sgl_flash_attn_file_logging`` in ``sglang.srt.utils``).

    When file logging is enabled, after a successful run the test prints a warning if
    ``test/srt/xpu/flash_attn.log`` is missing or empty. Set ``SGLANG_XPU_STRICT_FLASH_ATTN_LOG=1``
    to turn that into a hard failure. Quick check::

        ls -l test/srt/xpu/flash_attn.log && tail test/srt/xpu/flash_attn.log

Model implementation:
    Both ``bench_one_batch`` and the Q&A server pass ``--model-impl sglang`` so weights
    load into SGLang's native Llama module (``sglang/srt/models/llama.py``), not the
    Transformers fallback. Without this, ``--model-impl auto`` can still pick Transformers
    when no native implementation is registered for the checkpoint architecture.

Architecture note (tensor parallel vs Llama 3 8B):
    SGLang's Llama implementation requires ``num_attention_heads % tp_size == 0`` and
    shards MLP by ``intermediate_size // tp_size``. Meta Llama 3 / 3.1 8B models use
    32 Q heads and ``intermediate_size`` 14336; choose ``tp_size`` that divides both
    (e.g. 1, 2, 4, or 8). TP=5 is invalid for the stock checkpoint.
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import requests

from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
)

# Hugging Face id for Llama 3 8B Instruct (override with env for local paths).
MODEL_PATH = os.environ.get(
    "SGLANG_LLAMA3_8B_MODEL",
    "meta-llama/Meta-Llama-3-8B",
)

TP_SIZE = int(os.environ.get("SGLANG_TEST_TP_SIZE", "4"))

# Dedicated port for the Q&A server test (avoid clashing with ``DEFAULT_URL_FOR_TEST``).
_QA_PORT = int(
    os.environ.get(
        "SGLANG_XPU_QA_PORT",
        str(DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1950),
    )
)
QA_BASE_URL = f"http://127.0.0.1:{_QA_PORT}"

# Flash-attn file log default: ``test/srt/xpu/flash_attn.log`` (not XDG cache; see module docstring).
FLASH_ATTN_DEFAULT_LOG_FILE = str(Path(__file__).resolve().parent / "flash_attn.log")


def _ensure_flash_attn_default_log_dir() -> None:
    """Ensure the log directory exists (``test/srt/xpu/``) so ``FileHandler`` can open the file."""
    Path(FLASH_ATTN_DEFAULT_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


# Short completion prompts; answers checked case-insensitively.
QA_CASES = (
    {
        "name": "capital_france",
        "prompt": "The capital of France is",
        "expect_substrings": ("Paris", "paris"),
    },
    {
        "name": "two_plus_two",
        "prompt": "The result of 2 plus 2 is",
        "expect_substrings": ("4", "four"),
    },
)


def _flash_attn_log_path_for_child(env: dict) -> str:
    """Effective flash-attn log path for the child env (explicit ``SGL_FLASH_ATTN_LOG_FILE`` or test default)."""
    explicit = (env.get("SGL_FLASH_ATTN_LOG_FILE") or "").strip()
    if explicit:
        return explicit
    return FLASH_ATTN_DEFAULT_LOG_FILE


def _flash_attn_file_logging_active(env: dict) -> bool:
    if (env.get("SGL_FLASH_ATTN_LOG_FILE") or "").strip():
        return True
    return (env.get("SGL_FLASH_ATTN_LOG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _bench_subprocess_env() -> dict:
    """Child env with flash-attn file logging enabled unless user opted out (see module docstring)."""
    env = os.environ.copy()
    skip = (env.get("SGLANG_SKIP_FLASH_ATTN_FILE_LOG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    log_off = (env.get("SGL_FLASH_ATTN_LOG") or "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )
    if skip or log_off:
        return env
    if not (env.get("SGL_FLASH_ATTN_LOG_FILE") or "").strip():
        _ensure_flash_attn_default_log_dir()
        env["SGL_FLASH_ATTN_LOG_FILE"] = FLASH_ATTN_DEFAULT_LOG_FILE
        if not (env.get("SGL_FLASH_ATTN_LOG") or "").strip():
            env["SGL_FLASH_ATTN_LOG"] = "1"
    elif not (env.get("SGL_FLASH_ATTN_LOG") or "").strip():
        env["SGL_FLASH_ATTN_LOG"] = "1"
    return env


def _verify_flash_attn_log(test: unittest.TestCase, env: dict) -> None:
    """If file logging is on, warn or (strict) fail when the flash-attn log is missing or empty."""
    if not _flash_attn_file_logging_active(env):
        return
    log_path = Path(_flash_attn_log_path_for_child(env))
    strict = (
        os.environ.get("SGLANG_XPU_STRICT_FLASH_ATTN_LOG") or ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    ok = log_path.is_file() and log_path.stat().st_size > 0
    if ok:
        return
    msg = (
        f"Flash-attn log missing or empty at {log_path}. "
        f"This test runs bench/server with sys.executable={sys.executable!r}; "
        "confirm editable sgl-kernel is installed for that interpreter. "
        "Opt out: SGLANG_SKIP_FLASH_ATTN_FILE_LOG=1. "
        "Strict check: SGLANG_XPU_STRICT_FLASH_ATTN_LOG=1."
    )
    if strict:
        test.fail(msg)
    print(f"WARNING: {msg}", flush=True)


def _bench_command(model_path: str, tp: int) -> list:
    return [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--batch-size",
        "1",
        "--input-len",
        "128",
        "--output-len",
        "8",
        "--device",
        "xpu",
        "--tp",
        str(tp),
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--disable-radix",
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        "0.3",
        "--attention-backend",
        "intel_xpu",
        "--page-size",
        "128",
        "--model-impl",
        "sglang",
    ]


def _xpu_server_args() -> list:
    """Arguments aligned with the offline XPU bench test (``sglang serve``)."""
    return [
        "--device",
        "xpu",
        "--tp",
        str(TP_SIZE),
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        "0.3",
        "--attention-backend",
        "intel_xpu",
        "--page-size",
        "128",
        # Match ``_bench_command``: force SGLang's Llama stack, not Transformers fallback.
        "--model-impl",
        "sglang",
    ]


def _generate_completion(base_url: str, prompt: str, max_new_tokens: int = 24) -> str:
    r = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return body.get("text", "")


class TestLlama38BXPUTP5(CustomTestCase):
    def test_bench_one_batch(self):
        cmd = _bench_command(MODEL_PATH, TP_SIZE)
        child_env = _bench_subprocess_env()
        log_hint = _flash_attn_log_path_for_child(child_env)
        print("Running:", " ".join(cmd), flush=True)
        if _flash_attn_file_logging_active(child_env):
            print(
                f"sgl_kernel.flash_attn file log (if FA path is used): {log_hint}",
                flush=True,
            )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        try:
            out, _ = proc.communicate()
        finally:
            kill_process_tree(proc.pid)

        print(out, flush=True)
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"bench_one_batch failed (see output above). cmd={' '.join(cmd)}",
        )

        pattern = r"Decode\..*?throughput:\s*(?P<throughput>\d+\.\d+)"
        match = re.search(pattern, out, re.DOTALL)
        if match:
            decode_tput = float(match.group("throughput"))
            print(f"decode_throughput={decode_tput}", flush=True)
            if is_in_ci():
                self.assertGreater(decode_tput, 0.0)

        _verify_flash_attn_log(self, child_env)


class TestLlama38BXPUQuestionAnswer(CustomTestCase):
    """Question-style prompts verified against decoded completions (HTTP server, not batch bench)."""

    _process = None
    _child_env_for_fa_log = None

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as e:
            raise unittest.SkipTest(f"torch not available: {e}") from e
        if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU not available")

        cls.base_url = QA_BASE_URL
        child_env = _bench_subprocess_env()
        cls._child_env_for_fa_log = child_env
        if _flash_attn_file_logging_active(child_env):
            print(
                f"sgl_kernel.flash_attn file log (if FA path is used): "
                f"{_flash_attn_log_path_for_child(child_env)}",
                flush=True,
            )

        cls._process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="xpu",
            other_args=_xpu_server_args(),
            env=child_env,
            python_executable=sys.executable,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._process is not None:
            kill_process_tree(cls._process.pid)
            cls._process = None

    def test_question_answer(self):
        for case in QA_CASES:
            with self.subTest(case=case["name"]):
                completion = _generate_completion(
                    self.base_url, case["prompt"], max_new_tokens=32
                )
                lower = completion.lower()
                found = any(s.lower() in lower for s in case["expect_substrings"])
                self.assertTrue(
                    found,
                    msg=(
                        f"Expected one of {case['expect_substrings']!r} in completion "
                        f"for prompt {case['prompt']!r}; got: {completion!r}"
                    ),
                )

        _verify_flash_attn_log(self, self._child_env_for_fa_log)


if __name__ == "__main__":
    unittest.main()
