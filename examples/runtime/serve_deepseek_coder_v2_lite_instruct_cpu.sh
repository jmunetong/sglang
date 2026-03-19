#!/usr/bin/env bash
# Local test server for DeepSeek-Coder-V2-Lite-Instruct (text / code MoE) on CPU.
#
# Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
#
# Usage:
#   ./examples/runtime/serve_deepseek_coder_v2_lite_instruct_cpu.sh
#   PORT=8080 ./examples/runtime/serve_deepseek_coder_v2_lite_instruct_cpu.sh
#
# Smoke test:
#   python examples/runtime/smoke_test_deepseek_coder_v2_lite_instruct.py
#
# CLI equivalent:
#   sglang serve --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --device cpu --port 30000
#
# Device: CPU only (TP=1). For GPU/XPU use serve_deepseek_coder_v2_lite_instruct.sh instead.

# set +euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
DEVICE="${DEVICE:-cpu}"
TP="${TP:-1}"

cd "$(dirname "$0")/../.." || exit 1

common=(
  --model-path "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --tp "$TP"
  --trust-remote-code
  --disable-overlap-schedule
  --page-size 32
  --attention-backend triton
)

exec python3 -m sglang.launch_server "${common[@]}" "$@"
