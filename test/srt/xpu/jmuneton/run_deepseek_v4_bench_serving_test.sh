#!/usr/bin/env bash
# Run the DeepSeek-V4-Flash bench_serving test (watchdog/fd blocker repro)
# and mirror console output to a version-numbered log alongside the
# per-run artifacts the test itself writes
# (server.stdout.log, server.stderr.log, run.log, bench_serving_result.json).
#
# Naming: each run picks the next free vN slot based on the existing
# test_log_v*.log files in LOG_DIR, matching the scheme used by
# run_deepseek_v4_test.sh. Python uses the same N via
# SGLANG_DSV4_RUN_VERSION for its v<N>/ artifact subdir.
#
# Usage (from anywhere):
#   bash test/srt/xpu/jmuneton/run_deepseek_v4_bench_serving_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_ROOT="${REPO_ROOT}/test/srt"
LOG_DIR="${SCRIPT_DIR}/test_run_logs/deepseek_v4"
mkdir -p "${LOG_DIR}"

next_version=1
shopt -s nullglob
for f in "${LOG_DIR}"/test_log_v*.log; do
  base="$(basename "${f}")"
  n="${base#test_log_v}"
  n="${n%.log}"
  if [[ "${n}" =~ ^[0-9]+$ ]] && (( n >= next_version )); then
    next_version=$(( n + 1 ))
  fi
done
shopt -u nullglob

CONSOLE_LOG="${LOG_DIR}/test_log_v${next_version}.log"
mkdir -p "${LOG_DIR}/v${next_version}"

export SGLANG_USE_SGL_XPU=1
export SGLANG_DSV4_RUN_VERSION="${next_version}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] deepseek_v4 bench_serving test run v${next_version}" | tee "${CONSOLE_LOG}"

cd "${TEST_ROOT}"
python3 -m unittest -v \
  xpu.jmuneton.test_deepseek_v4_bench_serving.TestDeepSeekV4BenchServing.test_bench_serving_random \
  2>&1 | tee -a "${CONSOLE_LOG}"
