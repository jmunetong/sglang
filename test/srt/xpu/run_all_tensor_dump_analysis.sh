#!/usr/bin/env bash
# Run analyze_debug_tensor_dumps.py for every model known to the analyzer.
# Extra arguments are passed through to each run, e.g.:
#   ./run_all_tensor_dump_analysis.sh --max-passes 1

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZE="${SCRIPT_DIR}/analyze_debug_tensor_dumps.py"

MODELS=(
  deepseek_vl2_small
  qwen3_14b_fp8
  deepseek_coder_v2_lite_instruct
)

failed=0
for model in "${MODELS[@]}"; do
  echo "================================================================================"
  echo "Tensor dump analysis: ${model}"
  echo "================================================================================"
  if ! python3 "${ANALYZE}" "${model}" "$@"; then
    echo "FAILED: ${model}" >&2
    failed=$((failed + 1))
  fi
  echo ""
done

if [[ "${failed}" -gt 0 ]]; then
  echo "${failed} model run(s) failed." >&2
  exit 1
fi
exit 0
