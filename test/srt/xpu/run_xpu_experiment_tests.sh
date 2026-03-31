#!/usr/bin/env bash
# Run the three XPU model experiment unittests (DeepSeek-VL2, Qwen3-14B-FP8,
# DeepSeek-Coder-V2-Lite) with one log file per test.
#
# Usage (from anywhere):
#   ./run_xpu_experiment_tests.sh
#   bash /path/to/sglang/test/srt/xpu/run_xpu_experiment_tests.sh
#
# Requires: Intel XPU (tests skip otherwise), sglang deps, models on disk/HF cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${SCRIPT_DIR}/test_run_logs"
mkdir -p "${LOG_DIR}"

run_one() {
  local name="$1"
  local unittest_target="$2"
  local log="${LOG_DIR}/${name}.log"
  echo "=== ${name} ===" | tee "${log}"
  echo "Log: ${log}" | tee -a "${log}"
  echo "Started (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${log}"
  (
    cd "${SRT_DIR}"
    python3 -m unittest -v "${unittest_target}"
  ) 2>&1 | tee -a "${log}"
  local st="${PIPESTATUS[0]}"
  echo "Exit code: ${st}" | tee -a "${log}"
  return "${st}"
}

FAILED=0

run_one "deepseek_vl2_small" \
  "xpu.test_deepseek_vl2_small.TestDeepSeekVL2SmallXPU.test_simple_text_qa" \
  || FAILED=1

run_one "qwen3_14b_fp8" \
  "xpu.test_qwen3_14b_fp8.TestQwen314BFP8XPU.test_simple_text_qa" \
  || FAILED=1

run_one "deepseek_coder_v2_lite_instruct" \
  "xpu.test_deepseek_coder_v2_lite_instruct.TestDeepSeekCoderV2LiteInstructXPU.test_simple_code_qa" \
  || FAILED=1

if [[ "${FAILED}" -ne 0 ]]; then
  echo "One or more tests failed (see logs under ${LOG_DIR})." >&2
  exit 1
fi

echo "All tests passed. Logs: ${LOG_DIR}/*.log"
