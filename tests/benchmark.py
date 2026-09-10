"""Benchmark rdnattention's FP16 and INT8 kernels against fable_attn and a
naive rocBLAS baseline. Prints OPS, Time, Memory and Accuracy tables over a
full cross of head_dim, batch, shape, kv_heads and causal; each axis has a
flag that narrows it (see --help). A default run is 144 timing rows.

Needs a ROCm torch and fable_attn (a local source build of
2kiss/flash-attention-rdna2's package/), neither of them an rdnattention
dependency. Run from the repo root.

    python tests/benchmark.py
    python tests/benchmark.py -d 64,128 -s 512,1024 -c 0
"""
import argparse
import sys
import time
from dataclasses import dataclass
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

try:
    import fable_attn
except ImportError:
    print("fable_attn is not installed in this environment. It's a separate package "
          "(pip-packaged 2kiss/flash-attention-rdna2), not one of rdnattention's own "
          "dependencies - see this file's docstring for setup notes.")

import rdnattention

NUM_HEADS = 32
NUM_WARMUP = 5
NUM_RUNS = 20

HEAD_DIMS = [64, 128]
BATCH_SIZES = [1, 4, 16]

SHAPE_CASES = [
    ("self512", 512, 512),
    ("self1024", 1024, 1024),
    ("self2048", 2048, 2048),
    ("tail1300", 1300, 1300),
    ("decode", 1, 2048),
    ("prefill_chunk", 128, 2048),
]

GQA_RATIOS = [1, 8]  # kv_heads = NUM_HEADS // ratio
KV_HEADS = None      # -k overrides GQA_RATIOS
CAUSAL_MODES = [False, True]

# The naive column materializes the score matrix twice; ~9.2GB at batch=16,
# seq=2048. Re-check peak memory before raising this.
MAX_ROCBLAS_NAIVE_BATCH = 4

CAUSAL = False  # set per-iteration by run()

BASE_COLUMNS = ["rocBLAS (naive)", "Fable FP16", "FP16 (native HIP)", "INT8 (native HIP)"]

COLUMNS = list(BASE_COLUMNS)


@dataclass
class Result:
    ms: float
    ops: float  # TFLOP/s, or TOP/s for INT8
    mb: float


def ops_count(batch, heads, seq_len, key_seq_len, head_dim):
    """Both GEMMs, 2 FLOPs/MAC. Under causal this counts the exact valid
    (query,key) pairs rather than halving: a seq_len=1 decode query has
    delta = key_seq_len-1 and sees its entire cache, 0% masked."""
    if not CAUSAL:
        valid_pairs = seq_len * key_seq_len
    else:
        delta = key_seq_len - seq_len
        # Row q has min(q+delta+1, key_seq_len) valid keys; t rows are under the cap.
        t = max(0, min(key_seq_len - delta, seq_len))
        valid_pairs = t * (delta + 1) + t * (t - 1) // 2 + (seq_len - t) * key_seq_len
    return 4 * batch * heads * valid_pairs * head_dim


def quantize_symmetric(x: "torch.Tensor"):
    scale = x.abs().max().item() / 127.0
    if scale == 0.0:
        scale = 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def bench_rocblas_naive(batch, seq_len, key_seq_len, head_dim, kv_heads,
                         num_warmup=NUM_WARMUP, num_runs=NUM_RUNS) -> Optional[Result]:
    """Unfused matmul -> softmax -> matmul, the baseline tiling exists to beat."""
    if batch > MAX_ROCBLAS_NAIVE_BATCH:
        return None
    q_shape = (batch, NUM_HEADS, seq_len, head_dim)
    kv_shape = (batch, kv_heads, key_seq_len, head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    scale = head_dim ** -0.5
    ratio = NUM_HEADS // kv_heads
    delta = key_seq_len - seq_len

    causal_mask = None
    if CAUSAL:
        q_idx = torch.arange(seq_len, device="cuda").unsqueeze(1)
        k_idx = torch.arange(key_seq_len, device="cuda").unsqueeze(0)
        causal_mask = k_idx > (q_idx + delta)

    def forward():
        k_exp = k if ratio == 1 else k.repeat_interleave(ratio, dim=1)
        v_exp = v if ratio == 1 else v.repeat_interleave(ratio, dim=1)
        s = (q @ k_exp.transpose(-2, -1)) * scale
        if causal_mask is not None:
            s = s.masked_fill(causal_mask, float("-inf"))
        p = s.softmax(dim=-1)
        return p @ v_exp

    for _ in range(num_warmup):
        forward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_runs):
        forward()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_runs * 1000

    tflops = ops_count(batch, NUM_HEADS, seq_len, key_seq_len, head_dim) / (avg_ms / 1000) / 1e12
    mb = torch.cuda.max_memory_allocated() / 1e6
    return Result(avg_ms, tflops, mb)


def bench_fable(batch, seq_len, key_seq_len, head_dim, kv_heads,
                 num_warmup=NUM_WARMUP, num_runs=NUM_RUNS) -> Result:
    """fable_attn takes (batch, seqlen, nheads, headdim), not rdnattention's
    (batch, nheads, seqlen, headdim)."""
    q_shape = (batch, seq_len, NUM_HEADS, head_dim)
    kv_shape = (batch, key_seq_len, kv_heads, head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.float16)

    for _ in range(num_warmup):
        _ = fable_attn.flash_attn_func(q, k, v, causal=CAUSAL)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = fable_attn.flash_attn_func(q, k, v, causal=CAUSAL)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_runs * 1000

    tflops = ops_count(batch, NUM_HEADS, seq_len, key_seq_len, head_dim) / (avg_ms / 1000) / 1e12
    mb = torch.cuda.max_memory_allocated() / 1e6
    return Result(avg_ms, tflops, mb)


def bench_rdna_f16(batch, seq_len, key_seq_len, head_dim, kv_heads,
                    num_warmup=NUM_WARMUP, num_runs=NUM_RUNS) -> Result:
    q_shape = (batch, NUM_HEADS, seq_len, head_dim)
    kv_shape = (batch, kv_heads, key_seq_len, head_dim)
    q = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.float16)

    for _ in range(num_warmup):
        rdnattention.flash_attn(q, k, v, is_causal=CAUSAL)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_runs):
        rdnattention.flash_attn(q, k, v, is_causal=CAUSAL)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_runs * 1000

    tflops = ops_count(batch, NUM_HEADS, seq_len, key_seq_len, head_dim) / (avg_ms / 1000) / 1e12
    mb = torch.cuda.max_memory_allocated() / 1e6
    return Result(avg_ms, tflops, mb)


def bench_rdna_int8(batch, seq_len, key_seq_len, head_dim, kv_heads,
                     num_warmup=NUM_WARMUP, num_runs=NUM_RUNS) -> Optional[Result]:
    """None for unsupported configs, so the caller prints n/a."""
    if head_dim not in rdnattention.SUPPORTED_HEAD_DIMS_INT8:
        return None
    q_shape = (batch, NUM_HEADS, seq_len, head_dim)
    kv_shape = (batch, kv_heads, key_seq_len, head_dim)
    qf = torch.randn(q_shape, device="cuda", dtype=torch.float16)
    kf = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    vf = torch.randn(kv_shape, device="cuda", dtype=torch.float16)
    q, q_scale = quantize_symmetric(qf)
    k, k_scale = quantize_symmetric(kf)
    v, v_scale = quantize_symmetric(vf)
    del qf, kf, vf  # keep the harness's fp16 staging out of peak memory

    for _ in range(num_warmup):
        rdnattention.flash_attn_int8qk(q, k, v, q_scale, k_scale, v_scale, is_causal=CAUSAL)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_runs):
        rdnattention.flash_attn_int8qk(q, k, v, q_scale, k_scale, v_scale, is_causal=CAUSAL)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_runs * 1000

    tops = ops_count(batch, NUM_HEADS, seq_len, key_seq_len, head_dim) / (avg_ms / 1000) / 1e12
    mb = torch.cuda.max_memory_allocated() / 1e6
    return Result(avg_ms, tops, mb)


# Same kernel and card; only fdot2()'s lowering changes. Instruction cost on
# this card, not a prediction for Tier 2/3 silicon.
TIER_COLUMNS = ["Tier2 FP16 (mix)", "Tier3 FP16 (cvtfma)"]
TIER_FORCE_PATH = {"Tier2 FP16 (mix)": 3, "Tier3 FP16 (cvtfma)": 4}
_tier_libs = {}


class _ForcedPathLib:
    """One forced-fdot2 build, via ctypes rather than the rdnattention
    package: three must be live at once, and the package binds one at import.
    No process restart between samples keeps clock drift common-mode."""

    def __init__(self, lib_path):
        import ctypes
        self._lib = ctypes.CDLL(str(lib_path))  # keep the handle alive
        fn = self._lib.rdna_attention_forward
        fn.restype = ctypes.c_int32
        fn.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int64] * 12 +
                       [ctypes.c_uint32] * 6 + [ctypes.c_float] +
                       [ctypes.c_int32] * 3 + [ctypes.c_void_p])
        self._fn = fn
        self._ctypes = ctypes

    def flash_attn(self, q, k, v, is_causal=False):
        c = self._ctypes
        out = torch.empty_like(q)
        batch, heads, seq_len, head_dim = q.shape
        rc = self._fn(
            c.c_void_p(q.data_ptr()), c.c_void_p(k.data_ptr()),
            c.c_void_p(v.data_ptr()), c.c_void_p(out.data_ptr()), None, None,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            batch, heads, k.shape[1], seq_len, k.shape[2], head_dim,
            head_dim ** -0.5, 1 if is_causal else 0, 0, -1, None)
        if rc != 0:
            raise RuntimeError(f"rdna_attention_forward returned {rc}")
        return out


def load_tier_libs():
    """Build or reuse one library per forced lowering, returning the columns
    that came up. A missing toolchain drops columns rather than failing."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import dot_paths
    except ImportError as e:
        print(f"Tier columns unavailable (cannot import dot_paths: {e})")
        return []
    arch = dot_paths.detect_arch()
    if not arch:
        print("Tier columns unavailable (no HIP device reported by torch)")
        return []
    ok = []
    for col in TIER_COLUMNS:
        pid = TIER_FORCE_PATH[col]
        print(f"  building {col} (fdot2 path {pid}, arch {arch})...", flush=True)
        lib = dot_paths.build(pid, arch)
        if lib is None or not Path(lib).exists():
            print(f"  {col}: build failed, column will show n/a")
            continue
        try:
            _tier_libs[col] = _ForcedPathLib(lib)
            ok.append(col)
        except Exception as e:
            print(f"  {col}: load failed ({e}), column will show n/a")
    return ok


def bench_tier(col, batch, seq_len, key_seq_len, head_dim, kv_heads,
               num_warmup=NUM_WARMUP, num_runs=NUM_RUNS) -> Optional[Result]:
    lib = _tier_libs.get(col)
    if lib is None:
        return None
    q = torch.randn((batch, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, kv_heads, key_seq_len, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, kv_heads, key_seq_len, head_dim), device="cuda", dtype=torch.float16)

    for _ in range(num_warmup):
        lib.flash_attn(q, k, v, is_causal=CAUSAL)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_runs):
        lib.flash_attn(q, k, v, is_causal=CAUSAL)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_runs * 1000

    tflops = ops_count(batch, NUM_HEADS, seq_len, key_seq_len, head_dim) / (avg_ms / 1000) / 1e12
    return Result(avg_ms, tflops, torch.cuda.max_memory_allocated() / 1e6)


def collect_row(head_dim, batch, seq_len, key_seq_len, kv_heads):
    """Every backend for one config; one failure must not lose the row."""
    tag = f"[d={head_dim},B={batch},kvH={kv_heads},seq={seq_len},kseq={key_seq_len},causal={CAUSAL}]"
    results = {}

    try:
        results["rocBLAS (naive)"] = bench_rocblas_naive(batch, seq_len, key_seq_len, head_dim, kv_heads)
    except Exception as e:
        print(f"  {tag} rocBLAS (naive) error: {e}")
        results["rocBLAS (naive)"] = None

    try:
        results["Fable FP16"] = bench_fable(batch, seq_len, key_seq_len, head_dim, kv_heads)
    except Exception as e:
        print(f"  {tag} fable_attn error: {e}")
        results["Fable FP16"] = None

    try:
        results["FP16 (native HIP)"] = bench_rdna_f16(batch, seq_len, key_seq_len, head_dim, kv_heads)
    except Exception as e:
        print(f"  {tag} rdnattention (FP16) error: {e}")
        results["FP16 (native HIP)"] = None

    try:
        results["INT8 (native HIP)"] = bench_rdna_int8(batch, seq_len, key_seq_len, head_dim, kv_heads)
    except Exception as e:
        print(f"  {tag} rdnattention (INT8) error: {e}")
        results["INT8 (native HIP)"] = None

    for col in TIER_COLUMNS:
        if col not in COLUMNS:
            continue
        try:
            results[col] = bench_tier(col, batch, seq_len, key_seq_len, head_dim, kv_heads)
        except Exception as e:
            print(f"  {tag} {col} error: {e}")
            results[col] = None

    return results


# A separate pass from bench_*(): error needs every backend on identical
# inputs, and batch cannot affect per-element error, so the timing rows
# collapse to far fewer accuracy configs.
ACCURACY_COLUMNS = BASE_COLUMNS + ["INT8 +per-chan V"]

ACCURACY_BATCH = 1
ACCURACY_HEADS = 4  # stable RMS, cheap fp64 reference


def reference_attention(q, k, v, causal):
    """fp64, looped per (batch, head) - a batched fp64 score matrix is the
    one thing here that can exhaust VRAM at the larger shapes."""
    batch, heads, seq_len, head_dim = q.shape
    kv_heads, key_seq_len = k.shape[1], k.shape[2]
    group = heads // kv_heads
    delta = key_seq_len - seq_len
    out = torch.empty(batch, heads, seq_len, head_dim, device=q.device, dtype=torch.float64)
    q_idx = torch.arange(seq_len, device=q.device).unsqueeze(1)
    k_idx = torch.arange(key_seq_len, device=q.device).unsqueeze(0)
    mask = (k_idx > q_idx + delta) if causal else None
    for b in range(batch):
        for h in range(heads):
            scores = (q[b, h].double() @ k[b, h // group].double().T) * (head_dim ** -0.5)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            out[b, h] = scores.softmax(dim=-1) @ v[b, h // group].double()
    return out


def rel_rms(got, ref):
    got = got.double()
    denom = (ref ** 2).mean().sqrt()
    if denom == 0:
        return float("nan")
    return (((got - ref) ** 2).mean().sqrt() / denom).item()


def accuracy_row(head_dim, kv_heads_full, seq_len, key_seq_len, causal):
    """Every backend on one shared set of inputs. kv_heads_full is the row's
    kv_heads at NUM_HEADS; the GQA ratio survives the scale to ACCURACY_HEADS."""
    ratio = NUM_HEADS // kv_heads_full
    heads = ACCURACY_HEADS
    kv_heads = max(1, heads // ratio)
    heads = kv_heads * ratio  # keep the ratio exact after rounding

    torch.manual_seed(0)
    q = torch.randn(ACCURACY_BATCH, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(ACCURACY_BATCH, kv_heads, key_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(ACCURACY_BATCH, kv_heads, key_seq_len, head_dim, device="cuda", dtype=torch.float16)
    ref = reference_attention(q, k, v, causal)

    errors = {c: None for c in ACCURACY_COLUMNS}

    try:
        k_exp = k if ratio == 1 else k.repeat_interleave(ratio, dim=1)
        v_exp = v if ratio == 1 else v.repeat_interleave(ratio, dim=1)
        scores = (q @ k_exp.transpose(-2, -1)) * (head_dim ** -0.5)
        if causal:
            delta = key_seq_len - seq_len
            qi = torch.arange(seq_len, device=q.device).unsqueeze(1)
            ki = torch.arange(key_seq_len, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(ki > qi + delta, float("-inf"))
        errors["rocBLAS (naive)"] = rel_rms(scores.softmax(dim=-1) @ v_exp, ref)
    except Exception as e:
        print(f"  accuracy rocBLAS (naive) error: {e}")

    # Fable expects (batch, seqlen, nheads, headdim); transpose in and back.
    try:
        got = fable_attn.flash_attn_func(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(), causal=causal)
        errors["Fable FP16"] = rel_rms(got.transpose(1, 2), ref)
    except Exception as e:
        print(f"  accuracy fable_attn error: {e}")

    try:
        errors["FP16 (native HIP)"] = rel_rms(
            rdnattention.flash_attn(q, k, v, is_causal=causal), ref)
    except Exception as e:
        print(f"  accuracy rdnattention (FP16) error: {e}")

    for col in TIER_COLUMNS:
        if col not in ACCURACY_COLUMNS:
            continue
        try:
            errors[col] = rel_rms(_tier_libs[col].flash_attn(q, k, v, is_causal=causal), ref)
        except Exception as e:
            print(f"  accuracy {col} error: {e}")

    if head_dim in rdnattention.SUPPORTED_HEAD_DIMS_INT8:
        try:
            qi8, q_scale = quantize_symmetric(q)
            ki8, k_scale = quantize_symmetric(k)
            vi8, v_scale = quantize_symmetric(v)
            errors["INT8 (native HIP)"] = rel_rms(
                rdnattention.flash_attn_int8qk(qi8, ki8, vi8, q_scale, k_scale, v_scale,
                                               is_causal=causal), ref)
        except Exception as e:
            print(f"  accuracy rdnattention (INT8) error: {e}")
        try:
            errors["INT8 +per-chan V"] = rel_rms(
                rdnattention.flash_attn_int8qk_quantized(q, k, v, is_causal=causal), ref)
        except Exception as e:
            print(f"  accuracy rdnattention (INT8 +per-chan V) error: {e}")

    return errors


def print_accuracy_table(rows):
    print("\n=== Accuracy (rel_rms vs fp64 reference; lower is better) ===")
    id_header = f"{'d':<5}{'kvH':<5}{'seq':<7}{'kseq':<7}{'causal':<8}"
    header = id_header + "".join(f"{c:<20}" for c in ACCURACY_COLUMNS)
    print(header)
    print("-" * len(header))
    for head_dim, kv_heads, seq_len, key_seq_len, causal, errors in rows:
        id_cells = f"{head_dim:<5}{kv_heads:<5}{seq_len:<7}{key_seq_len:<7}{str(causal):<8}"
        cells = []
        for col in ACCURACY_COLUMNS:
            e = errors.get(col)
            cells.append(f"{e:<20.3e}" if e is not None else f"{'n/a':<20}")
        print(id_cells + "".join(cells))


def print_table(title, rows, field, fmt):
    """field: 'ops', 'ms', or 'mb'."""
    print(f"\n=== {title} ===")
    id_header = f"{'d':<5}{'B':<5}{'kvH':<5}{'seq':<7}{'kseq':<7}{'causal':<8}"
    header = id_header + "".join(f"{c:<20}" for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for head_dim, batch, kv_heads, seq_len, key_seq_len, causal, results in rows:
        id_cells = f"{head_dim:<5}{batch:<5}{kv_heads:<5}{seq_len:<7}{key_seq_len:<7}{str(causal):<8}"
        cells = []
        for col in COLUMNS:
            r = results.get(col)
            cells.append(f"{getattr(r, field):<20{fmt}}" if r is not None else f"{'n/a':<20}")
        print(id_cells + "".join(cells))


def run(with_tiers=True):
    global CAUSAL, COLUMNS, ACCURACY_COLUMNS

    if not rdnattention.has_device():
        print("No usable HIP device found.")
        sys.exit(1)
    print(f"torch/ROCm device: {torch.cuda.get_device_name(0)}")
    print(f"H={NUM_HEADS}, warmup={NUM_WARMUP}, runs={NUM_RUNS}")
    kv_head_counts = KV_HEADS if KV_HEADS is not None else [NUM_HEADS // r for r in GQA_RATIOS]
    print(f"head_dims={HEAD_DIMS}, batches={BATCH_SIZES}, "
          f"shape_cases={[(s, k) for _, s, k in SHAPE_CASES]}, kv_heads={kv_head_counts}, causal={CAUSAL_MODES}")
    total_rows = len(HEAD_DIMS) * len(BATCH_SIZES) * len(SHAPE_CASES) * len(kv_head_counts) * len(CAUSAL_MODES)
    print(f"Total rows this run: {total_rows}\n")

    if with_tiers:
        print("Tier 2/3 columns: building one library per fdot2 lowering "
              "(cached in build/_dotpaths, ~1 min on a cold tree)")
        available = load_tier_libs()
        COLUMNS = BASE_COLUMNS + available
        ACCURACY_COLUMNS = BASE_COLUMNS + available + ["INT8 +per-chan V"]
        print()

    rows = []
    script_start = time.perf_counter()
    for head_dim in HEAD_DIMS:
        for batch in BATCH_SIZES:
            for _, seq_len, key_seq_len in SHAPE_CASES:
                for kv_heads in kv_head_counts:
                    for causal in CAUSAL_MODES:
                        CAUSAL = causal
                        results = collect_row(head_dim, batch, seq_len, key_seq_len, kv_heads)
                        rows.append((head_dim, batch, kv_heads, seq_len, key_seq_len, causal, results))
    script_elapsed = time.perf_counter() - script_start

    acc_keys = []
    for head_dim, _batch, kv_heads, seq_len, key_seq_len, causal, _r in rows:
        key = (head_dim, kv_heads, seq_len, key_seq_len, causal)
        if key not in acc_keys:
            acc_keys.append(key)
    print(f"Measuring accuracy: {len(acc_keys)} distinct configs "
          f"(from {len(rows)} timing rows; batch does not affect accuracy)\n")
    acc_start = time.perf_counter()
    acc_rows = [(hd, kvh, s, ks, c, accuracy_row(hd, kvh, s, ks, c))
                for hd, kvh, s, ks, c in acc_keys]
    acc_elapsed = time.perf_counter() - acc_start

    print_table("OPS  (TFLOP/s for FP16 columns, TOP/s for INT8 columns - same op-count formula)",
                rows, "ops", ".2f")
    print_table("Time (ms)", rows, "ms", ".3f")
    print_table("Memory (MB)", rows, "mb", ".1f")
    print_accuracy_table(acc_rows)
    print(f"\nAccuracy pass: {acc_elapsed:.2f}s at batch={ACCURACY_BATCH}, "
          f"heads={ACCURACY_HEADS} (GQA ratio preserved).")

    print(f"\nTotal wall time for this run: {script_elapsed:.2f}s")
    print(f"kvH = key/value heads (< {NUM_HEADS} is GQA); seq/kseq = query/key length "
          f"(unequal is cross-attention/decode).")
    print(f"rocBLAS (naive): unfused matmul->softmax->matmul, skipped above batch="
          f"{MAX_ROCBLAS_NAIVE_BATCH}. Fable FP16: fable_attn. FP16/INT8 (native HIP): "
          f"rdnattention.flash_attn() and flash_attn_int8qk().")
    print("INT8 +per-chan V (accuracy only): flash_attn_int8qk_quantized().")
    print("Accuracy: rel_rms against an fp64 reference on identical inputs.")
    if any(c in COLUMNS for c in TIER_COLUMNS):
        print("Tier2/Tier3: the same kernel with fdot2() forced to those lowerings - the cost of "
              "losing v_dot2 on this card, not a prediction for Tier 2/3 silicon.")


def int_list(text):
    """"512,1024" -> [512, 1024]. Duplicates are dropped, order kept."""
    try:
        vals = [int(x) for x in text.replace(" ", "").split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return list(dict.fromkeys(vals))


def bool_list(text):
    """"0,1" / "false,true" / "both" -> [False, True]."""
    out = []
    for tok in text.replace(" ", "").lower().split(","):
        if tok in ("0", "false", "f", "no", "n"):
            out.append(False)
        elif tok in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif tok == "both":
            out.extend((False, True))
        elif tok:
            raise argparse.ArgumentTypeError(f"expected 0/1/both, got {tok!r}")
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return list(dict.fromkeys(out))


def build_shape_cases(seqs, kseqs):
    """Cross seq with key_seq. kseqs=None means self-attention at each seq."""
    cases = []
    for s_len in seqs:
        for ks_len in (kseqs if kseqs else [s_len]):
            tag = f"self{s_len}" if s_len == ks_len else f"cross{s_len}x{ks_len}"
            cases.append((tag, s_len, ks_len))
    return cases


def main(argv=None):
    global NUM_HEADS, HEAD_DIMS, BATCH_SIZES, SHAPE_CASES, KV_HEADS, CAUSAL_MODES

    ap = argparse.ArgumentParser(
        description="Benchmark rdnattention (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Every axis takes a comma-separated list and defaults to the full sweep
in the module docstring, so flags narrow the run rather than replace it:

  python tests/benchmark.py -d 64,128 -s 512,1024
  python tests/benchmark.py -d 384 -s 4096 -c 0 --no-tiers
  python tests/benchmark.py -s 1,128 -ks 2048          # decode + prefill

-s alone means self-attention (key_seq = seq). -s and -ks together are
CROSSED, not zipped: -s 1,128 -ks 2048 is two cross-attention shapes against
one KV cache, and -s 512,1024 -ks 512,1024 is all four combinations.""")
    ap.add_argument("-d", "--head-dim", type=int_list, metavar="LIST",
                    help=f"head dimensions (default {HEAD_DIMS})")
    ap.add_argument("-b", "--batch", type=int_list, metavar="LIST",
                    help=f"batch sizes (default {BATCH_SIZES}); rocBLAS is skipped above "
                         f"{MAX_ROCBLAS_NAIVE_BATCH}, see module docstring")
    ap.add_argument("-k", "--kv-heads", type=int_list, metavar="LIST",
                    help=f"key/value head counts, must divide --heads (default: "
                         f"{[NUM_HEADS // r for r in GQA_RATIOS]}, i.e. GQA ratios {GQA_RATIOS})")
    ap.add_argument("-s", "--seq", type=int_list, metavar="LIST", help="query sequence lengths")
    ap.add_argument("-ks", "--kseq", type=int_list, metavar="LIST",
                    help="key/value sequence lengths, crossed with -s (requires -s)")
    ap.add_argument("-c", "--causal", type=bool_list, metavar="LIST",
                    help="causal modes: 0, 1, or 0,1 (default both)")
    ap.add_argument("--heads", type=int, metavar="N",
                    help=f"query head count (default {NUM_HEADS}); -k values must divide it")
    ap.add_argument("--no-tiers", action="store_true",
                    help="skip the Tier 2/3 columns (avoids building two extra libraries)")
    args = ap.parse_args(argv)

    if args.heads is not None:
        if args.heads < 1:
            ap.error(f"--heads must be >= 1, got {args.heads}")
        NUM_HEADS = args.heads
    if args.head_dim:
        bad = [d for d in args.head_dim if d not in rdnattention.SUPPORTED_HEAD_DIMS]
        if bad:
            ap.error(f"-d: head_dim {bad} not supported (multiples of 32 from "
                     f"{rdnattention.SUPPORTED_HEAD_DIMS[0]} to {rdnattention.SUPPORTED_HEAD_DIMS[-1]})")
        HEAD_DIMS = args.head_dim
    if args.batch:
        if min(args.batch) < 1:
            ap.error(f"-b: batch sizes must be >= 1, got {args.batch}")
        BATCH_SIZES = args.batch
    if args.kv_heads:
        bad = [n for n in args.kv_heads if n < 1 or NUM_HEADS % n != 0]
        if bad:
            ap.error(f"-k: kv_heads {bad} must be >= 1 and divide --heads ({NUM_HEADS}); "
                     f"valid here: {[n for n in range(1, NUM_HEADS + 1) if NUM_HEADS % n == 0]}")
        KV_HEADS = args.kv_heads
    if args.kseq and not args.seq:
        ap.error("-ks needs -s: key_seq is crossed with seq, and there is no sensible "
                 "default query length to cross it against")
    if args.seq:
        if min(args.seq + (args.kseq or [1])) < 1:
            ap.error("-s/-ks: sequence lengths must be >= 1")
        SHAPE_CASES = build_shape_cases(args.seq, args.kseq)
    if args.causal:
        CAUSAL_MODES = args.causal

    run(with_tiers=not args.no_tiers)


if __name__ == "__main__":
    main()
