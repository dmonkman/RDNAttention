"""Accuracy of the lossy attention backends against an fp64 CPU reference.
tests/benchmark.py measures speed; this measures error.

    python tests/accuracy.py
"""
import sys
from math import sqrt
from pathlib import Path
from typing import Optional

RDNATTENTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RDNATTENTION_ROOT))

try:
    import torch
except ImportError:
    print("This script needs a ROCm PyTorch install - not one of rdnattention's own "
          "dependencies. Run it from an environment that has one.")
    sys.exit(1)

import rdnattention

NUM_HEADS = 8
HEAD_DIMS = [64, 128]
SEQ_LENS = [1024, 4096]

CAPTURED_QKV_DIR = Path(__file__).parent / "data" / "captured_qkv"

INT8_HEAD_DIMS = rdnattention.SUPPORTED_HEAD_DIMS_INT8


def rel_rms(got: "torch.Tensor", ref: "torch.Tensor") -> float:
    got64 = got.detach().to("cpu", dtype=torch.float64)
    diff = got64 - ref
    denom = ref.pow(2).sum().sqrt()
    return (diff.pow(2).sum().sqrt() / denom).item() if denom > 0 else diff.pow(2).sum().sqrt().item()


def dense_attention_reference(q: "torch.Tensor", k: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    # Per (batch, head): a batched fp64 score tensor at capture shapes is gigabytes.
    scale = 1.0 / sqrt(q.shape[-1])
    out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            s = (q[i, j] @ k[i, j].transpose(-1, -2)) * scale
            out[i, j] = torch.softmax(s, dim=-1) @ v[i, j]
    return out


def quantize_symmetric(x: "torch.Tensor"):
    scale = x.abs().max().item() / 127.0
    if scale == 0.0:
        scale = 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def make_random_qkv(batch, heads, seq_len, head_dim):
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    return q, k, v


def make_structured_qkv(batch, heads, seq_len, head_dim):
    pos = torch.linspace(0, 8 * 3.14159, seq_len, dtype=torch.float64)
    freqs = torch.arange(1, head_dim // 2 + 1, dtype=torch.float64)
    phase = pos[:, None] * freqs[None, :] / head_dim
    base = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    base = base / base.norm(dim=-1, keepdim=True)
    base = base.view(1, 1, seq_len, head_dim).expand(batch, heads, seq_len, head_dim)
    noise_scale = 0.3
    q = base + noise_scale * torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    k = base + noise_scale * torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float64)
    return q, k, v


def load_captured_qkv():
    if not CAPTURED_QKV_DIR.is_dir():
        print(f"\n[CAPTURED regime] {CAPTURED_QKV_DIR} does not exist - skipping.")
        return []
    files = sorted(CAPTURED_QKV_DIR.rglob("*.pt"))
    if not files:
        print(f"\n[CAPTURED regime] {CAPTURED_QKV_DIR} exists but has no .pt files - skipping.")
        return []
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        # The capture hook slices K/V along seq; the kernels need contiguous.
        q = d["q"].double().contiguous()
        k = d["k"].double().contiguous()
        v = d["v"].double().contiguous()
        rel = f.relative_to(CAPTURED_QKV_DIR).parent
        name = f.stem if rel == Path(".") else f"{rel.as_posix()}/{f.stem}"
        out.append((name, q, k, v))
    return out


def eval_fp16(q64, k64, v64, ref) -> float:
    q, k, v = (t.to("cuda", dtype=torch.float16) for t in (q64, k64, v64))
    got = rdnattention.flash_attn(q, k, v, is_causal=False)
    return rel_rms(got, ref)


def eval_int8(q64, k64, v64, ref) -> Optional[float]:
    if q64.shape[-1] not in INT8_HEAD_DIMS:
        return None
    qi, q_scale = quantize_symmetric(q64)
    ki, k_scale = quantize_symmetric(k64)
    vi, v_scale = quantize_symmetric(v64)
    qi, ki, vi = (t.to("cuda") for t in (qi, ki, vi))
    got = rdnattention.flash_attn_int8qk(qi, ki, vi, q_scale, k_scale, v_scale, is_causal=False)
    return rel_rms(got, ref)


def eval_int8_perchannel_v(q64, k64, v64, ref) -> Optional[float]:
    if q64.shape[-1] not in INT8_HEAD_DIMS:
        return None
    q, k, v = (t.to("cuda", dtype=torch.float16).contiguous() for t in (q64, k64, v64))
    return rel_rms(rdnattention.flash_attn_int8qk_quantized(q, k, v), ref)


def run_regime(name: str, cases):
    print(f"\n=== {name} ===")
    name_w = max(24, max((len(c) for c, *_ in cases), default=24) + 2)
    header = f"{'case':<{name_w}}{'FP16':<12}{'INT8':<12}"
    print(header)
    print("-" * len(header))
    for case_name, q64, k64, v64 in cases:
        ref = dense_attention_reference(q64, k64, v64)
        row = f"{case_name:<{name_w}}"
        row += f"{eval_fp16(q64, k64, v64, ref):.4f}".ljust(12)
        int8_err = eval_int8(q64, k64, v64, ref)
        row += f"{'n/a' if int8_err is None else f'{int8_err:.4f}':<12}"
        print(row)
        del q64, k64, v64, ref
        torch.cuda.empty_cache()


def run_captured_regime(cases):
    print("\n=== CAPTURED (real model activations - the only regime that means "
          "anything for a lossy backend) ===")
    for case_name, q64, k64, v64 in cases:
        n_q, n_k, head_dim = q64.shape[-2], k64.shape[-2], q64.shape[-1]
        print(f"\n{case_name}")
        print(f"  Nq={n_q} Nk={n_k} D={head_dim}")
        ref = dense_attention_reference(q64, k64, v64)

        print(f"    {'backend':<32}{'rel_rms':<10}note")
        print(f"    {'FP16':<32}{eval_fp16(q64, k64, v64, ref):<10.4f}exact-attention cost")
        int8_err = eval_int8(q64, k64, v64, ref)
        print(f"    {'INT8 (per-tensor)':<32}"
              + (f"{'n/a':<10}head_dim unsupported" if int8_err is None
                 else f"{int8_err:<10.4f}exact-attention cost"))
        pc_err = eval_int8_perchannel_v(q64, k64, v64, ref)
        if pc_err is not None:
            gain = f"{int8_err / pc_err:.2f}x vs per-tensor" if int8_err else ""
            print(f"    {'INT8 (per-channel V)':<32}{pc_err:<10.4f}{gain}")

        del ref
        torch.cuda.empty_cache()


def run():
    if not rdnattention.has_device():
        print("No usable HIP device found.")
        sys.exit(1)
    print(f"torch/ROCm device: {torch.cuda.get_device_name(0)}")

    random_cases = []
    structured_cases = []
    for head_dim in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            name = f"H{NUM_HEADS} N{seq_len} D{head_dim}"
            random_cases.append((name, *make_random_qkv(1, NUM_HEADS, seq_len, head_dim)))
            structured_cases.append((name, *make_structured_qkv(1, NUM_HEADS, seq_len, head_dim)))

    run_regime("RANDOM (floor/regression check only)", random_cases)
    run_regime("STRUCTURED (synthetic positional correlation - proxy only)", structured_cases)

    captured_cases = load_captured_qkv()
    if captured_cases:
        run_captured_regime(captured_cases)

    print("\nrel_rms = sqrt(sum((got-ref)^2) / sum(ref^2)) against exact fp64 dense attention.")


if __name__ == "__main__":
    run()
