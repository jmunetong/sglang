#!/usr/bin/env bash
# Run the DeepSeek-V4-Flash XPU test and mirror console output to a
# timestamped log alongside the per-run artifacts the test itself writes
# (server.stdout.log, server.stderr.log, run.log, *.response.txt).
#
# Usage (from anywhere):
#   bash test/srt/xpu/jmuneton/run_deepseek_v4_test.sh

set -euo pipefail

# Resolve repo root from this script's location: test/srt/xpu/jmuneton/<script>.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_ROOT="${REPO_ROOT}/test/srt"
LOG_DIR="${SCRIPT_DIR}/test_run_logs/deepseek_v4"
mkdir -p "${LOG_DIR}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CONSOLE_LOG="${LOG_DIR}/console.${STAMP}.log"

export SGLANG_USE_SGL_XPU=1

cd "${TEST_ROOT}"
python3 -m unittest -v xpu.jmuneton.test_deepseek_v4.TestDeepSeekV4XPU.test_simple_code_qa \
  2>&1 | tee -a "${CONSOLE_LOG}"
