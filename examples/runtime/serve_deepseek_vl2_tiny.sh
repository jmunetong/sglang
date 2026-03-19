#!/usr/bin/env bash
# Lightweight local test server for DeepSeek-VL2-Tiny (vision-language).
#
# Prerequisites (from repo root, after installing sglang):
#   pip install torchvision   # required for DeepSeek-VL2 image preprocessing
#
# Usage:
#   ./examples/runtime/serve_deepseek_vl2_tiny.sh
#   PORT=8080 ./examples/runtime/serve_deepseek_vl2_tiny.sh --tp 1
#
# Then test with OpenAI-compatible API, e.g.:
#   python examples/runtime/smoke_test_deepseek_vl2_tiny.py
#
# Or use the CLI (recommended):
#   sglang serve --model-path deepseek-ai/deepseek-vl2-tiny --port 30000

set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-vl2-tiny}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

cd "$(dirname "$0")/../.." || exit 1

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --device xpu \
  --tp 2 \
  --attention-backend intel_xpu \
  --trust-remote-code \
  --disable-overlap-schedule  \
  --page-size 32 \
  "$@"
