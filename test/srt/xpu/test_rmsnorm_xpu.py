"""
RMSNorm on Intel XPU vs CPU reference (sglang.srt.layers.layernorm.RMSNorm).

Requires:
  - PyTorch with XPU (Intel Extension for PyTorch)
  - sgl-kernel-xpu (sgl_kernel with XPU rmsnorm / fused_add_rmsnorm ops)

Run from repo (cwd must be test/srt for CI):
  cd test/srt && python3 xpu/test_rmsnorm_xpu.py
"""

from __future__ import annotations

import sys
from typing import Literal, Tuple

Layout = Literal["contiguous", "strided_batch", "strided_hidden"]


def _cpu_xpu_diff_stats(
    xpu_tensor, cpu_reference, label: str
) -> tuple[float, float, float]:
    """Return (max_abs_diff, mean_abs_diff, max_relative_diff) in float32 on CPU."""

    a = xpu_tensor.detach().float().cpu().flatten()
    b = cpu_reference.detach().float().cpu().flatten()
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rel = diff / (b.abs() + 1e-5)
    max_rel = float(rel.max().item())
    print(
        f"  [{label}] max|xpu-cpu_ref|={max_abs:.4e}  mean_abs={mean_abs:.4e}  max_rel={max_rel:.4e}"
    )
    return max_abs, mean_abs, max_rel


def _make_2d(
    layout: Layout,
    batch: int,
    hidden: int,
    device,
    dtype,
    seed: int,
):
    """Build [batch, hidden] with requested memory layout (uses manual_seed(seed))."""
    import torch

    torch.manual_seed(seed)
    if layout == "contiguous":
        return torch.randn(batch, hidden, device=device, dtype=dtype)
    if layout == "strided_batch":
        t = torch.randn(batch * 2, hidden, device=device, dtype=dtype)
        x = t[::2, :]
        assert x.shape == (batch, hidden) and not x.is_contiguous(), layout
        return x
    if layout == "strided_hidden":
        t = torch.randn(batch, hidden * 2, device=device, dtype=dtype)
        x = t[:, ::2]
        assert x.shape == (batch, hidden) and not x.is_contiguous(), layout
        return x
    raise ValueError(layout)


def main() -> int:
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("ERROR: torch.xpu is not available. Run on an XPU machine.")
        return 1

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.utils import cpu_has_amx_support

    device = torch.device("xpu")
    batch, hidden = 8, 256
    eps = 1e-6

    layouts: Tuple[Layout, ...] = (
        "contiguous",
        "strided_batch",
        "strided_hidden",
    )

    if RMSNorm(hidden_size=hidden, eps=eps)._forward_method.__name__ != "forward_xpu":
        print(
            "WARNING: RMSNorm is not using forward_xpu. "
            "Import sglang on a host where torch.xpu.is_available() is True."
        )

    def assert_close(
        a: torch.Tensor, b: torch.Tensor, msg: str, *, rtol: float, atol: float
    ) -> None:
        a32 = a.detach().float().cpu()
        b32 = b.detach().float().cpu()
        max_diff = (a32 - b32).abs().max().item()
        ok = torch.allclose(a32, b32, rtol=rtol, atol=atol)
        if not ok:
            print(f"FAIL {msg}: max_abs_diff={max_diff:.6g}")
            raise AssertionError(msg)
        print(f"OK   {msg} (max_abs_diff={max_diff:.6g})")

    dtype_configs: Tuple[Tuple[str, torch.dtype, float, float], ...] = (
        ("bfloat16", torch.bfloat16, 2e-2, 2e-2),
        ("float16", torch.float16, 3e-2, 3e-2),
    )

    max_abs_overall = 0.0

    for dtype_idx, (dtype_label, dtype, rtol, atol) in enumerate(dtype_configs):
        print(
            f"\n{'=' * 60}\n"
            f"dtype={dtype_label} (XPU vs CPU forward_native; layouts contiguous + strided)\n"
            f"{'=' * 60}"
        )

        norm = RMSNorm(hidden_size=hidden, eps=eps).to(device=device, dtype=dtype)
        norm_cpu = RMSNorm(hidden_size=hidden, eps=eps)
        norm_cpu.weight.data.copy_(norm.weight.data.cpu())

        base_seed = 10_000 + dtype_idx * 50_000

        for layout in layouts:
            h = abs(hash(layout)) % 1000
            print(
                f"\n--- dtype={dtype_label} layout={layout} "
                f"(contiguous={layout == 'contiguous'}) ---"
            )

            x = _make_2d(layout, batch, hidden, device, dtype, base_seed + h)
            out_xpu = norm(x.clone())
            x_cpu = x.cpu().clone()
            out_ref = norm_cpu.forward_native(x_cpu, None, None)
            ma, _, _ = _cpu_xpu_diff_stats(
                out_xpu, out_ref, f"{dtype_label}/{layout} / no residual"
            )
            max_abs_overall = max(max_abs_overall, ma)
            assert_close(
                out_xpu,
                out_ref,
                f"{dtype_label} rmsnorm no residual [{layout}]",
                rtol=rtol,
                atol=atol,
            )

            if layout == "contiguous" and cpu_has_amx_support():
                try:
                    w_cpu = norm_cpu.weight.data.cpu().contiguous()
                    out_amx = torch.ops.sgl_kernel.rmsnorm_cpu(
                        x_cpu.contiguous().clone(), w_cpu, eps
                    )
                    _cpu_xpu_diff_stats(
                        out_xpu,
                        out_amx,
                        f"{dtype_label} / vs CPU rmsnorm_cpu (AMX)",
                    )
                    if not torch.allclose(
                        out_xpu.detach().float().cpu(),
                        out_amx.float(),
                        rtol=rtol,
                        atol=atol,
                    ):
                        print(
                            "WARN: XPU vs CPU rmsnorm_cpu (AMX) exceeds tolerance "
                            f"for {dtype_label}."
                        )
                except (AttributeError, RuntimeError) as e:
                    print(f"  (skip CPU AMX compare: {e})")

            x = _make_2d(layout, batch, hidden, device, dtype, base_seed + h + 100)
            res = _make_2d(layout, batch, hidden, device, dtype, base_seed + h + 200)
            x_in, r_in = x.clone(), res.clone()
            out_xpu, res_out_xpu = norm(x_in, r_in)
            out_ref, res_out_ref = norm_cpu.forward_native(
                x.cpu().clone(), res.cpu().clone(), None
            )
            ma1, _, _ = _cpu_xpu_diff_stats(
                out_xpu, out_ref, f"{dtype_label}/{layout} / fused norm"
            )
            ma2, _, _ = _cpu_xpu_diff_stats(
                res_out_xpu, res_out_ref, f"{dtype_label}/{layout} / fused res"
            )
            max_abs_overall = max(max_abs_overall, ma1, ma2)
            assert_close(
                out_xpu,
                out_ref,
                f"{dtype_label} fused norm out [{layout}]",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                res_out_xpu,
                res_out_ref,
                f"{dtype_label} fused residual [{layout}]",
                rtol=rtol,
                atol=atol,
            )

            if layout == "contiguous" and cpu_has_amx_support():
                try:
                    xc = x.cpu().clone().contiguous()
                    rc = res.cpu().clone().contiguous()
                    torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                        xc, rc, norm_cpu.weight.data.cpu().contiguous(), eps
                    )
                    _cpu_xpu_diff_stats(
                        out_xpu,
                        xc,
                        f"{dtype_label} fused / vs CPU fused_add_rmsnorm_cpu",
                    )
                    _cpu_xpu_diff_stats(
                        res_out_xpu,
                        rc,
                        f"{dtype_label} fused res / vs CPU",
                    )
                except (AttributeError, RuntimeError) as e:
                    print(f"  (skip CPU fused AMX compare: {e})")

    if max_abs_overall > 0.5:
        print(f"FAIL sanity: max_abs XPU vs CPU ref too large: {max_abs_overall}")
        return 1

    print(
        "\nAll RMSNorm XPU checks passed "
        "(bfloat16 + float16, contiguous + non-contiguous)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
