"""
DeepSeek-V4-Flash bench_serving reproducer on XPU.

Purpose
-------
Exercise the exact code path the team reports as their current blocker:

  "We're hitting a watchdog timer crash when running bench_serving after
   server launch at full model depth. It works with fewer layers, suggesting
   a timeout during initialization or first-batch execution on XPU. It shows
   error as a 'bad file descriptor at the server'."

This test launches ``sglang.launch_server`` with the same DSv4 flag set used
by ``test_deepseek_v4.py`` (tp=8, compressed attention, page-size=128,
disable-radix / cuda-graph / shared-experts-fusion, deepseek-v4 tool +
reasoning parsers) and then drives it through ``sglang.bench_serving``'s
``run_benchmark`` with a small random workload.

Unlike ``test_deepseek_v4.TestDeepSeekV4XPU`` which validates correctness on
a single OpenAI chat request, this test intentionally *only* checks that
the benchmark completes. We want to isolate whether the crash happens at
init, warmup, or the first real batch — the artifacts under
``test_run_logs/deepseek_v4/v<N>/`` record exactly that, even when the test
errors out.

Usage (from test/srt)::

  python3 -m unittest -v \
      xpu.jmuneton.test_deepseek_v4_bench_serving.TestDeepSeekV4BenchServing.test_bench_serving_random

Or via the runner script::

  bash test/srt/xpu/jmuneton/run_deepseek_v4_bench_serving_test.sh

Per-run artifacts (shared schema with test_deepseek_v4.py):
  - server.stdout.log / server.stderr.log — streamed server output.
  - run.log                                — high-level timeline.
  - bench_serving_result.json              — parsed benchmark summary
                                             (or the error that prevented
                                             a summary from existing).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sglang.bench_serving import run_benchmark
from sglang.srt.utils.common import is_xpu
from sglang.test.test_utils import CustomTestCase, get_benchmark_args
from sglang.test.vlm_utils import (
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)

MODEL = "deepseek-ai/DeepSeek-V4-Flash"

_HERE = Path(__file__).resolve().parent
TEST_RUN_LOGS_ROOT = _HERE / "test_run_logs" / "deepseek_v4"
LAUNCH_TIMEOUT = 1800  # weights + TP init on XPU is slow.

# bench_serving workload — intentionally tiny so any hang isn't masked by
# sheer volume. If the team's failure happens "at full model depth during
# first-batch execution", one prompt is enough to trigger it.
NUM_PROMPTS = 4
RANDOM_INPUT_LEN = 256
RANDOM_OUTPUT_LEN = 64
REQUEST_RATE = float("inf")  # all requests go out immediately.

# Populated in setUpModule so every artifact from a single invocation lives
# under one v<N>/ subdirectory (shared with test_deepseek_v4.py scheme).
_RUN_DIR: Path | None = None
_RUN_LOG_PATH: Path | None = None
_SERVER_STDOUT_PATH: Path | None = None
_SERVER_STDERR_PATH: Path | None = None


def _server_subprocess_env() -> dict:
    # Inherit the parent env so PATH / CONDA_PREFIX / PYTHONPATH /
    # ONEAPI_ROOT flow into the server subprocess (same rationale as the
    # sibling test).
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
    """Pick v<N> based on existing test_log_v*.log / v*/ entries. Matches the
    scheme used by test_deepseek_v4.py so both tests' artifacts can share a
    directory when run back-to-back."""
    env_v = os.environ.get("SGLANG_DSV4_RUN_VERSION")
    if env_v and env_v.isdigit():
        return int(env_v)
    TEST_RUN_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    highest = 0
    for p in TEST_RUN_LOGS_ROOT.glob("test_log_v*.log"):
        try:
            highest = max(highest, int(p.stem[len("test_log_v") :]))
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
    _run_log(
        f"setUpModule: bench_serving variant; artifacts dir = {_RUN_DIR}, "
        f"num_prompts={NUM_PROMPTS}, random_input_len={RANDOM_INPUT_LEN}, "
        f"random_output_len={RANDOM_OUTPUT_LEN}"
    )


# Same flag set as the sibling test — the team's blocker is specifically
# tied to this configuration "at full model depth".
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


def _dump_bench_result(result: dict | None, error: BaseException | None) -> None:
    """Always-on dump of the bench_serving outcome.

    Called from a finally so we record the run whether it returned a metrics
    dict, returned None (the bench early-exited), or raised. The team's
    reported failure mode surfaces here as either an exception traceback or
    a partial result with fewer completed requests than requested.
    """
    if _RUN_DIR is None:
        return
    path = _RUN_DIR / "bench_serving_result.json"
    payload: dict = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_prompts_requested": NUM_PROMPTS,
        "random_input_len": RANDOM_INPUT_LEN,
        "random_output_len": RANDOM_OUTPUT_LEN,
    }
    if result is not None:
        # bench_serving returns a large dict including per-request detail —
        # keep only the summary fields to stay under a sensible artifact
        # size while still showing whether the run made forward progress.
        keep_keys = (
            "completed",
            "duration",
            "total_input_tokens",
            "total_output_tokens",
            "request_throughput",
            "input_throughput",
            "output_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p99_itl_ms",
            "mean_e2e_latency_ms",
            "median_e2e_latency_ms",
            "p99_e2e_latency_ms",
        )
        payload["result"] = {k: result.get(k) for k in keep_keys if k in result}
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        _run_log(f"bench_serving: wrote result -> {path.name}")
    except Exception as write_err:
        _run_log(f"bench_serving: failed to write result: {write_err!r}")


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDeepSeekV4BenchServing(CustomTestCase):
    _server_stdout = None
    _server_stderr = None

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        os.environ["SGLANG_USE_SGL_XPU"] = "1"

        assert _SERVER_STDOUT_PATH is not None and _SERVER_STDERR_PATH is not None
        # Line-buffered so that if the server hangs / is killed by the
        # watchdog mid-init, whatever it had printed is already on disk.
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
            for fh in (cls._server_stdout, cls._server_stderr):
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            raise
        _run_log("setUpClass: server healthy; ready for bench_serving")

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

    def test_bench_serving_random(self):
        """Drive sglang.bench_serving with a tiny random workload.

        Assertion is intentionally minimal — we only require that the
        benchmark made forward progress (completed > 0). The interesting
        signal for the team is *how* it fails: the bench_serving_result.json
        file shows whether completion was zero (hang/crash before first
        batch), partial (watchdog killed mid-run), or full (no repro).
        """
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            num_prompts=NUM_PROMPTS,
            random_input_len=RANDOM_INPUT_LEN,
            random_output_len=RANDOM_OUTPUT_LEN,
            request_rate=REQUEST_RATE,
            disable_stream=False,
            disable_ignore_eos=False,
            seed=0,
            device="xpu",
        )

        _run_log(
            "test_bench_serving_random: invoking run_benchmark "
            f"(num_prompts={NUM_PROMPTS}, in_len={RANDOM_INPUT_LEN}, "
            f"out_len={RANDOM_OUTPUT_LEN})"
        )

        result: dict | None = None
        error: BaseException | None = None
        try:
            # run_benchmark is sync and expects to be driven in an event
            # loop when the caller wants async; here we call it directly on
            # the main thread inside asyncio.run() to mirror
            # run_bench_serving's pattern and so bench_serving's own
            # asyncio.gather over requests has a fresh loop.
            async def _drive():
                return await asyncio.to_thread(run_benchmark, copy.deepcopy(args))

            result = asyncio.run(_drive())
            _run_log(
                "test_bench_serving_random: run_benchmark returned; "
                f"completed={result.get('completed') if result else None}"
            )
        except BaseException as e:
            error = e
            _run_log(f"test_bench_serving_random: EXCEPTION {e!r}")
            raise
        finally:
            _dump_bench_result(result, error)

        # Minimal forward-progress check — tighter assertions would hide
        # the very failure mode we're trying to catalogue.
        self.assertIsNotNone(result)
        self.assertGreater(
            result.get("completed", 0),
            0,
            "bench_serving reported zero completed requests — likely hang "
            "during init or first-batch execution; see server.stderr.log",
        )


if __name__ == "__main__":
    unittest.main()
