#!/usr/bin/env python3
"""
Scan SGLang debug tensor dumps (Pass*.pt) and report numeric health for hidden states:
value ranges, NaN/Inf counts, and tallies of very small / very large / subnormal values.

Default output is **per transformer layer**, with submodules grouped by kind:
attention (``self_attn`` / ``attn``), RMSNorm / LayerNorm, FFN (``mlp`` / ``feed_forward``),
embeddings, and other. Use ``--group-by-module`` for the legacy view (grouped by inner path).

Dumps are produced with --debug-tensor-dump-output-folder (see tensor_dump_forward_hook.py).

Unless ``--no-log`` is set, the full report is also written to
``<dump-root>/<model>/tensor_dump_analysis.log`` (override with ``--log-file``).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import torch

# Keys look like "model.layers.12.mlp.down_proj" or
# "language_model.model.layers.0.self_attn.q_proj".
LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")

MODELS = (
    "deepseek_vl2_small",
    "qwen3_14b_fp8",
    "deepseek_coder_v2_lite_instruct",
)

# Print order within each layer; human-readable headings.
COMPONENT_KIND_HEADINGS: Tuple[Tuple[str, str], ...] = (
    ("embedding", "EMBEDDING"),
    ("rmsnorm", "RMSNORM / LAYERNORM"),
    ("attention", "ATTENTION"),
    ("ffn", "FFN (MLP)"),
    ("other", "OTHER"),
)


def layer_component_kind(inner: str) -> str:
    """
    Classify the inner module path (after ``.layers.<i>.``) for per-layer reporting.

    Attention is checked before norm so ``self_attn.q_norm`` stays under attention.
    """
    il = inner.lower()
    if "embed" in il:
        return "embedding"
    if any(
        p in il
        for p in (
            "self_attn",
            "cross_attn",
            "encoder_attn",
            "masked_attention",
        )
    ):
        return "attention"
    if il.startswith("attn.") or ".attn." in il:
        return "attention"
    if "mlp" in il or "feed_forward" in il or il.startswith("ffn") or ".ffn." in il:
        return "ffn"
    if any(
        p in il
        for p in (
            "rms_norm",
            "layernorm",
            "layer_norm",
            "input_layernorm",
            "post_attention_layernorm",
        )
    ):
        return "rmsnorm"
    if il.endswith("q_norm") or il.endswith("k_norm"):
        return "attention"
    if "_norm" in il or il.endswith(".norm") or ".norm." in il:
        return "rmsnorm"
    return "other"


def _script_default_dump_root() -> Path:
    return Path(__file__).resolve().parent / "debug_tensor_dump_output"


def default_analysis_log_path(model_dir: Path) -> Path:
    """One log per model, next to that model's dump tree under --dump-root."""
    return model_dir / "tensor_dump_analysis.log"


class _TeeStdout:
    """Mirror stdout writes to a log file (terminal output unchanged)."""

    def __init__(self, primary: Any, log_file: Any) -> None:
        self._primary = primary
        self._log = log_file

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._log.write(s)
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()


def inner_module_and_layer(key: str) -> Tuple[Optional[int], str]:
    """
    Split a dump key into (layer_index, inner_module_path).

    ``model.layers.3.self_attn.qkv_proj`` -> (3, "self_attn.qkv_proj")
    Keys without ``.layers.<n>.`` -> (None, full key), one group per distinct op.
    """
    m = LAYER_INDEX_RE.search(key)
    if not m:
        return None, key
    inner = key[m.end() :].lstrip(".")
    if not inner:
        inner = "(layer_output)"
    return int(m.group(1)), inner


def iter_leaf_tensors(obj: Any) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from iter_leaf_tensors(x)


def is_float_tensor(t: torch.Tensor) -> bool:
    return t.dtype.is_floating_point or t.dtype.is_complex


@dataclass
class TensorScan:
    numel: int = 0
    numel_finite: int = 0
    nan: int = 0
    pos_inf: int = 0
    neg_inf: int = 0
    min_val: float = math.inf
    max_val: float = -math.inf
    sum_val: float = 0.0
    sum_abs: float = 0.0
    count_small: int = 0
    count_large: int = 0
    count_subnormal: int = 0

    def merge(self, other: TensorScan) -> None:
        had_finite = self.numel_finite > 0
        self.numel += other.numel
        self.numel_finite += other.numel_finite
        self.nan += other.nan
        self.pos_inf += other.pos_inf
        self.neg_inf += other.neg_inf
        if other.numel_finite > 0:
            if not had_finite:
                self.min_val = other.min_val
                self.max_val = other.max_val
            else:
                self.min_val = min(self.min_val, other.min_val)
                self.max_val = max(self.max_val, other.max_val)
        self.sum_val += other.sum_val
        self.sum_abs += other.sum_abs
        self.count_small += other.count_small
        self.count_large += other.count_large
        self.count_subnormal += other.count_subnormal


def scan_tensor(
    t: torch.Tensor,
    small_thresh: float,
    large_thresh: float,
) -> Optional[TensorScan]:
    if t.numel() == 0 or not is_float_tensor(t):
        return None
    if t.dtype.is_complex:
        t = torch.view_as_real(t).reshape(-1).to(torch.float32)
    else:
        t = t.detach().reshape(-1)
        if t.dtype != torch.float32:
            t = t.float()

    out = TensorScan()
    out.numel = t.numel()
    out.nan = int(torch.isnan(t).sum().item())
    out.pos_inf = int(torch.isposinf(t).sum().item())
    out.neg_inf = int(torch.isneginf(t).sum().item())
    finite = torch.isfinite(t)
    if not bool(finite.any().item()):
        return out

    tf = t[finite]
    out.numel_finite = tf.numel()
    out.min_val = float(tf.min().item())
    out.max_val = float(tf.max().item())
    out.sum_val = float(tf.sum().item())
    out.sum_abs = float(tf.abs().sum().item())
    abs_t = tf.abs()
    out.count_small = int(((abs_t > 0) & (abs_t < small_thresh)).sum().item())
    out.count_large = int((abs_t > large_thresh).sum().item())
    finfo = torch.finfo(torch.float32)
    smallest_normal = float(finfo.smallest_normal)
    out.count_subnormal = int(((tf != 0) & (abs_t < smallest_normal)).sum().item())
    return out


def pick_rank_directory(model_dir: Path, rank_dir: Optional[str]) -> Path:
    if rank_dir:
        p = model_dir / rank_dir
        if not p.is_dir():
            raise FileNotFoundError(f"Not a directory: {p}")
        return p
    subs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    if not subs:
        raise FileNotFoundError(f"No rank subdirectories under {model_dir}")

    def pass_count(d: Path) -> int:
        return len(list(d.glob("Pass*.pt")))

    with_passes = [p for p in subs if pass_count(p) > 0]
    pool = with_passes if with_passes else subs
    tp0 = [p for p in pool if p.name.startswith("TP0_PP0_")]
    candidates = tp0 if tp0 else pool
    # Prefer the rank dir with the most dump files (often the latest successful run).
    return max(candidates, key=lambda p: (pass_count(p), p.stat().st_mtime))


def pass_files(rank_dir: Path, max_passes: Optional[int]) -> List[Path]:
    files = sorted(rank_dir.glob("Pass*.pt"))
    if max_passes is not None:
        files = files[:max_passes]
    return files


@dataclass
class ModuleAggregate:
    """Stats for one inner module at one layer index (or non-layer)."""

    stats: TensorScan = field(default_factory=TensorScan)
    tensor_keys: int = 0
    pass_files: int = 0


def _analyze_files_fixed(
    files: Iterable[Path],
    small_thresh: float,
    large_thresh: float,
    full_dump_keys: bool,
) -> Tuple[
    DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
    DefaultDict[Tuple[str, Optional[int]], Dict[str, TensorScan]],
]:
    """
    Build per-inner-module -> per-layer aggregates. Optionally keep full dump keys
    under (inner, layer_id) for --full-dump-keys.
    """
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]] = defaultdict(dict)
    full_key_maps: DefaultDict[Tuple[str, Optional[int]], Dict[str, TensorScan]] = (
        defaultdict(dict)
    )

    for pt_path in files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            continue
        seen: Set[Tuple[str, Optional[int]]] = set()
        for key, value in data.items():
            layer_id, inner = inner_module_and_layer(key)
            tensors = list(iter_leaf_tensors(value))
            if not tensors:
                continue
            combined = TensorScan()
            for t in tensors:
                s = scan_tensor(t, small_thresh, large_thresh)
                if s is None:
                    continue
                combined.merge(s)
            if combined.numel == 0:
                continue
            slot = by_inner[inner].get(layer_id)
            if slot is None:
                slot = ModuleAggregate()
                by_inner[inner][layer_id] = slot
            slot.stats.merge(combined)
            slot.tensor_keys += 1
            seen.add((inner, layer_id))
            if full_dump_keys:
                fk = full_key_maps[(inner, layer_id)]
                if key not in fk:
                    fk[key] = TensorScan()
                fk[key].merge(combined)
        for inner, lid in seen:
            by_inner[inner][lid].pass_files += 1

    return by_inner, full_key_maps


def _rollup_by_layer(
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
) -> Dict[Optional[int], ModuleAggregate]:
    """Merge all inner modules into one aggregate per layer index (legacy view)."""
    out: Dict[Optional[int], ModuleAggregate] = defaultdict(ModuleAggregate)
    for _inner, layer_map in by_inner.items():
        for layer_id, agg in layer_map.items():
            slot = out[layer_id]
            slot.stats.merge(agg.stats)
            slot.tensor_keys += agg.tensor_keys
            slot.pass_files = max(slot.pass_files, agg.pass_files)
    # pass_files should be max across inners per layer (same passes touch all); max is OK
    return dict(out)


def group_by_layer_and_kind(
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
) -> Tuple[
    Dict[int, Dict[str, List[Tuple[str, ModuleAggregate]]]],
    Dict[str, List[Tuple[str, ModuleAggregate]]],
]:
    """
    Split aggregates into per-layer, per-component-kind buckets.

    Returns (layer_id -> kind -> [(inner_module, agg), ...], non_layer_kind -> [...]).
    """
    layers: DefaultDict[int, DefaultDict[str, List[Tuple[str, ModuleAggregate]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    non_layer: DefaultDict[str, List[Tuple[str, ModuleAggregate]]] = defaultdict(list)

    for inner, layer_map in by_inner.items():
        kind = layer_component_kind(inner)
        for lid, agg in layer_map.items():
            if lid is None:
                non_layer[kind].append((inner, agg))
            else:
                layers[lid][kind].append((inner, agg))

    layers_out: Dict[int, Dict[str, List[Tuple[str, ModuleAggregate]]]] = {}
    for lid, km in layers.items():
        layers_out[lid] = {k: sorted(v, key=lambda x: x[0]) for k, v in km.items()}
    non_layer_out = {k: sorted(v, key=lambda x: x[0]) for k, v in non_layer.items()}
    return layers_out, non_layer_out


def stability_verdict(
    s: TensorScan,
    small_thresh: float,
    large_thresh: float,
) -> Tuple[str, str]:
    """
    Short label for hidden-state health: UNSTABLE (NaN/Inf), WARN (many extremes), OK.
    """
    if s.nan or s.pos_inf or s.neg_inf:
        parts: List[str] = []
        if s.nan:
            parts.append(f"NaN={s.nan}")
        if s.pos_inf or s.neg_inf:
            parts.append(f"+Inf={s.pos_inf} -Inf={s.neg_inf}")
        return "UNSTABLE", f" ({', '.join(parts)})"
    denom = s.numel_finite if s.numel_finite else s.numel
    if not denom:
        return "OK", ""
    warn: List[str] = []
    pct_large = 100.0 * s.count_large / denom
    pct_small = 100.0 * s.count_small / denom
    pct_sub = 100.0 * s.count_subnormal / denom
    if pct_large > 1.0:
        warn.append(f"|x|>{large_thresh:g}: {pct_large:.2f}%")
    if pct_small > 5.0:
        warn.append(f"tiny (0, {small_thresh:g}): {pct_small:.2f}%")
    if pct_sub > 0.1:
        warn.append(f"subnormal: {pct_sub:.4f}%")
    if warn:
        return "WARN", f" ({'; '.join(warn)})"
    return "OK", ""


def print_stability_summary(
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
    small_thresh: float,
    large_thresh: float,
) -> None:
    """
    Final section: list transformer layers (and non-layer ops) with UNSTABLE stats (NaN/Inf).
    """
    unstable_by_layer: DefaultDict[int, List[str]] = defaultdict(list)
    unstable_non_layer: List[str] = []

    for inner, layer_map in by_inner.items():
        for lid, agg in layer_map.items():
            if agg.stats.numel == 0:
                continue
            verdict, detail = stability_verdict(agg.stats, small_thresh, large_thresh)
            if verdict != "UNSTABLE":
                continue
            line = f"{inner} — UNSTABLE{detail}"
            if lid is None:
                unstable_non_layer.append(line)
            else:
                unstable_by_layer[lid].append(line)

    print()
    print("=== STABILITY SUMMARY ===")
    if unstable_by_layer or unstable_non_layer:
        if unstable_by_layer:
            print(
                "Transformer layers with UNSTABLE numeric output (NaN / Inf in at least "
                "one submodule):"
            )
            for lid in sorted(unstable_by_layer.keys()):
                print(f"  layer {lid}:")
                for entry in sorted(unstable_by_layer[lid]):
                    print(f"    - {entry}")
        else:
            print(
                "All transformer layers were stable (no NaN or Inf in any dumped "
                "layer submodule)."
            )
        if unstable_non_layer:
            print("Non-layer modules with UNSTABLE numeric output:")
            for entry in sorted(unstable_non_layer):
                print(f"  - {entry}")
    else:
        print(
            "All transformer layers were stable (no NaN or Inf in any dumped submodule)."
        )
    print()


def format_module_aggregate_lines(
    label: str,
    agg: ModuleAggregate,
    small_thresh: float,
    large_thresh: float,
    indent: str = "  ",
) -> List[str]:
    """Human-readable lines for one ModuleAggregate (no section header)."""
    s = agg.stats
    lines: List[str] = []
    lines.append(
        f"{indent}{label}  passes={agg.pass_files}  outputs={agg.tensor_keys}  "
        f"elements={s.numel} (finite={s.numel_finite})"
    )
    if s.numel == 0:
        lines.append(f"{indent}  (no floating-point tensors)")
        return lines

    bad = []
    if s.nan:
        bad.append(f"NaN={s.nan}")
    if s.pos_inf or s.neg_inf:
        bad.append(f"+Inf={s.pos_inf} -Inf={s.neg_inf}")
    lines.append(f"{indent}  issues: {', '.join(bad) if bad else 'none'}")
    sv, sd = stability_verdict(s, small_thresh, large_thresh)
    lines.append(f"{indent}  stability: {sv}{sd}")

    if s.numel_finite > 0:
        mean = s.sum_val / s.numel_finite
        mean_abs = s.sum_abs / s.numel_finite
        range_s = f"[{s.min_val:.6g}, {s.max_val:.6g}]"
    else:
        mean = float("nan")
        mean_abs = float("nan")
        range_s = "n/a (no finite values)"
    lines.append(
        f"{indent}  range: {range_s}  mean: {mean:.6g}  mean|x|: {mean_abs:.6g}"
    )
    denom = s.numel_finite if s.numel_finite else s.numel
    pct_small = 100.0 * s.count_small / denom if denom else 0.0
    pct_large = 100.0 * s.count_large / denom if denom else 0.0
    pct_sub = 100.0 * s.count_subnormal / denom if denom else 0.0
    lines.append(
        f"{indent}  |x| in (0, {small_thresh:g}): {s.count_small} ({pct_small:.4f}%)"
    )
    lines.append(
        f"{indent}  |x| > {large_thresh:g}: {s.count_large} ({pct_large:.4f}%)"
    )
    lines.append(
        f"{indent}  subnormal (fp32, 0 < |x| < smallest_normal): "
        f"{s.count_subnormal} ({pct_sub:.6f}%)"
    )
    return lines


def print_inner_module_inventory(
    by_inner: Dict[str, Dict[Optional[int], ModuleAggregate]],
) -> None:
    """One line per inner path: layer index range and count (quick coverage check)."""
    print("Inner module paths in dump (check for layernorm / self_attn / mlp / …):")
    print(f"{'inner_module':<48} {'layers':<24} count")
    for inner in sorted(by_inner.keys()):
        lids = [i for i in by_inner[inner] if i is not None]
        if not lids:
            layer_s = "non-layer"
            n = 1 if None in by_inner[inner] else 0
        else:
            lo, hi = min(lids), max(lids)
            layer_s = f"{lo}..{hi}" if lo != hi else str(lo)
            n = len(lids)
        print(f"{inner:<48} {layer_s:<24} {n}")
    print()


def _print_full_dump_keys_for_slot(
    full_key_maps: DefaultDict[Tuple[str, Optional[int]], Dict[str, TensorScan]],
    inner: str,
    lid: Optional[int],
    indent: str = "      ",
) -> None:
    fk = full_key_maps.get((inner, lid), {})
    for dkey in sorted(fk):
        ts = fk[dkey]
        mean = ts.sum_val / ts.numel_finite if ts.numel_finite else float("nan")
        flags: List[str] = []
        if ts.nan:
            flags.append(f"NaN={ts.nan}")
        if ts.pos_inf or ts.neg_inf:
            flags.append(f"Inf +{ts.pos_inf}/-{ts.neg_inf}")
        if ts.count_large:
            flags.append(f"large={ts.count_large}")
        if ts.count_small:
            flags.append(f"tiny={ts.count_small}")
        flag_s = f" [{'; '.join(flags)}]" if flags else ""
        print(
            f"{indent}dump_key {dkey}: [{ts.min_val:.6g}, {ts.max_val:.6g}] "
            f"mean={mean:.6g} n={ts.numel}{flag_s}"
        )


def print_per_layer_grouped_report(
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
    full_key_maps: DefaultDict[Tuple[str, Optional[int]], Dict[str, TensorScan]],
    small_thresh: float,
    large_thresh: float,
    full_dump_keys: bool,
) -> None:
    """Default report: each layer, submodules grouped by attention / norm / FFN / …"""
    layers, non_layer = group_by_layer_and_kind(by_inner)

    for lid in sorted(layers.keys()):
        print(f"=== Transformer layer {lid} ===")
        by_kind = layers[lid]
        any_block = False
        for kind, heading in COMPONENT_KIND_HEADINGS:
            items = by_kind.get(kind, [])
            if not items:
                continue
            any_block = True
            print(f"--- {heading} ---")
            for inner, agg in items:
                for line in format_module_aggregate_lines(
                    inner,
                    agg,
                    small_thresh,
                    large_thresh,
                    indent="  ",
                ):
                    print(line)
                if full_dump_keys:
                    _print_full_dump_keys_for_slot(
                        full_key_maps, inner, lid, indent="      "
                    )
        if not any_block:
            print("  (no tensors in dump for this layer)")
        print()

    if non_layer:
        print("=== Non-layer modules (embeddings, lm_head, vision stack, etc.) ===")
        for kind, heading in COMPONENT_KIND_HEADINGS:
            items = non_layer.get(kind, [])
            if not items:
                continue
            print(f"--- {heading} ---")
            for inner, agg in items:
                for line in format_module_aggregate_lines(
                    inner,
                    agg,
                    small_thresh,
                    large_thresh,
                    indent="  ",
                ):
                    print(line)
                if full_dump_keys:
                    _print_full_dump_keys_for_slot(
                        full_key_maps, inner, None, indent="      "
                    )
            print()


def print_group_by_module_report(
    by_inner: DefaultDict[str, Dict[Optional[int], ModuleAggregate]],
    full_key_maps: DefaultDict[Tuple[str, Optional[int]], Dict[str, TensorScan]],
    small_thresh: float,
    large_thresh: float,
    full_dump_keys: bool,
    sort_layer_ids_fn: Callable[[List[Optional[int]]], List[Optional[int]]],
) -> None:
    """Legacy layout: one section per inner module path."""
    for inner in sorted(by_inner.keys()):
        layer_map = by_inner[inner]
        kind = layer_component_kind(inner)
        print(f"=== {inner}  [{kind}] ===")
        for lid in sort_layer_ids_fn(list(layer_map.keys())):
            agg = layer_map[lid]
            layer_label = "non-layer" if lid is None else f"layer {lid}"
            for line in format_module_aggregate_lines(
                layer_label,
                agg,
                small_thresh,
                large_thresh,
                indent="  ",
            ):
                print(line)
            if full_dump_keys:
                _print_full_dump_keys_for_slot(full_key_maps, inner, lid)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SGLang debug tensor dumps for NaNs, Infs, value ranges, and "
        "magnitude issues in hidden states — by transformer layer (attention / norm / FFN) "
        "or optionally grouped by module path."
    )
    parser.add_argument(
        "model",
        choices=MODELS,
        help="Which model subdirectory under the dump root to analyze.",
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=None,
        help="Parent folder containing <model>/TP*_PP*_Rank*_pid*/Pass*.pt "
        f"(default: {_script_default_dump_root()})",
    )
    parser.add_argument(
        "--rank-dir",
        default=None,
        help="Exact subdirectory name under the model folder (default: prefer TP0_PP0_*).",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=None,
        help="Only read the first N Pass*.pt files (sorted by name). Default: all.",
    )
    parser.add_argument(
        "--small-threshold",
        type=float,
        default=1e-7,
        help="Flag values with 0 < |x| < this threshold (underflow / denorm risk).",
    )
    parser.add_argument(
        "--large-threshold",
        type=float,
        default=1e4,
        help="Flag values with |x| > this threshold (possible blow-up).",
    )
    parser.add_argument(
        "--full-dump-keys",
        action="store_true",
        help="Under each (inner module, layer), list every full dump key (e.g. "
        "language_model.model.layers.0.self_attn.q_proj) with stats.",
    )
    parser.add_argument(
        "--layer-rollup",
        action="store_true",
        help="After the per-module report, print one merged summary per layer index "
        "(all inner modules combined).",
    )
    parser.add_argument(
        "--list-inner-modules",
        action="store_true",
        help="Print a compact table of inner module paths and layer index coverage, "
        "then exit (no full numeric report). Use this to verify attention, norms, FFN, etc. are present.",
    )
    parser.add_argument(
        "--group-by-module",
        action="store_true",
        help="Group output by inner module path (legacy layout) instead of by layer with "
        "attention / RMSNorm / FFN sections.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write the same report to this file (UTF-8). Default: "
        "<dump-root>/<model>/tensor_dump_analysis.log",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a log file (stdout only).",
    )
    args = parser.parse_args()

    root = args.dump_root or _script_default_dump_root()
    model_dir = root / args.model
    if not model_dir.is_dir():
        print(f"Model dump directory not found: {model_dir}", file=sys.stderr)
        return 1

    try:
        rank_dir = pick_rank_directory(model_dir, args.rank_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    files = pass_files(rank_dir, args.max_passes)
    if not files:
        print(f"No Pass*.pt under {rank_dir}", file=sys.stderr)
        return 1

    log_path: Optional[Path] = None
    log_fp = None
    old_stdout = sys.stdout
    if not args.no_log:
        log_path = args.log_file or default_analysis_log_path(model_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_fp = open(log_path, "w", encoding="utf-8")
        except OSError as e:
            print(f"Could not open log file {log_path}: {e}", file=sys.stderr)
            return 1
        sys.stdout = _TeeStdout(old_stdout, log_fp)

    def _close_log() -> None:
        nonlocal log_fp
        if log_fp is not None:
            sys.stdout = old_stdout
            log_fp.close()
            log_fp = None
            print(f"Analysis log written to: {log_path}", file=old_stdout)

    try:
        print(
            f"# tensor_dump_analysis  model={args.model}  "
            f"utc={datetime.now(timezone.utc).isoformat()}"
        )
        print(f"Model: {args.model}")
        print(f"Rank directory: {rank_dir}")
        print(f"Pass files: {len(files)} (from {files[0].name} … {files[-1].name})")
        print(
            f"Thresholds: small=|x| in (0, {args.small_threshold:g}), "
            f"large=|x| > {args.large_threshold:g}"
        )
        if log_path is not None:
            print(f"Log file: {log_path}")
        print()

        by_inner, full_key_maps = _analyze_files_fixed(
            files,
            args.small_threshold,
            args.large_threshold,
            args.full_dump_keys,
        )

        if args.list_inner_modules:
            print_inner_module_inventory(dict(by_inner))
        else:

            def sort_layer_ids(ids: List[Optional[int]]) -> List[Optional[int]]:
                ints = sorted(i for i in ids if i is not None)
                rest = [i for i in ids if i is None]
                return ints + rest  # type: ignore[return-value]

            if args.group_by_module:
                print_group_by_module_report(
                    by_inner,
                    full_key_maps,
                    args.small_threshold,
                    args.large_threshold,
                    args.full_dump_keys,
                    sort_layer_ids,
                )
            else:
                print_per_layer_grouped_report(
                    by_inner,
                    full_key_maps,
                    args.small_threshold,
                    args.large_threshold,
                    args.full_dump_keys,
                )

            if args.layer_rollup:
                rolled = _rollup_by_layer(by_inner)
                print("=== LAYER ROLLUP (all inner modules merged) ===")
                for lid in sort_layer_ids(list(rolled.keys())):
                    agg = rolled[lid]
                    label = "non-layer" if lid is None else f"layer {lid}"
                    for line in format_module_aggregate_lines(
                        label,
                        agg,
                        args.small_threshold,
                        args.large_threshold,
                        indent="",
                    ):
                        print(line)
                    print()

            print_stability_summary(
                by_inner,
                args.small_threshold,
                args.large_threshold,
            )
    finally:
        _close_log()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
