"""Accuracy of the lossy attention backends against an fp64 CPU reference.
tests/benchmark.py measures speed; this measures error.

MonarchAttention is an opt-in external comparison, not a dependency of this
project: point MONARCH_ATTENTION_ROOT at a checkout of
github.com/cjyaras/monarch-attention, and MONARCHRT_ROOT at MonarchRT for its
tiled variant. Both are skipped when absent.

Unlike speed, a structural approximation's error depends on the Q/K/V values
and not just their shape - random Q/K carries no structure for it to exploit,
so the three regimes below are not interchangeable. Only CAPTURED says
anything about real generation quality.

    python tests/accuracy.py
"""
import os
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

MONARCH_ATTENTION_ROOT = Path(os.environ.get("MONARCH_ATTENTION_ROOT", r"D:\dev\monarch-attention"))
sys.path.insert(0, str(MONARCH_ATTENTION_ROOT))
try:
    from ma.ma_triton import monarch_attention_triton
    MONARCH_AVAILABLE = True
except ImportError:
    MONARCH_AVAILABLE = False
    print(f"monarch-attention not found at {MONARCH_ATTENTION_ROOT} (set MONARCH_ATTENTION_ROOT "
          "to override) - MonarchAttention rows will be skipped.")

# Loaded by file path rather than by putting MonarchRT's root on sys.path: its
# module is named "attention", too generic to import globally. monarch.py has no
# relative imports, needs only torch + einops, and is pure PyTorch.
MONARCHRT_ROOT = Path(os.environ.get("MONARCHRT_ROOT", r"D:\dev\MonarchRT_AMD"))
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "monarchrt_monarch", MONARCHRT_ROOT / "attention" / "monarch.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    tiled_monarch_attn = _mod.monarch_attn
    MONARCHRT_AVAILABLE = True
except Exception as _e:
    MONARCHRT_AVAILABLE = False
    print(f"MonarchRT not usable at {MONARCHRT_ROOT} ({type(_e).__name__}: {_e}) - "
          "set MONARCHRT_ROOT to override; tiled Monarch rows will be skipped.")

NUM_HEADS = 8
HEAD_DIMS = [64, 128]
SEQ_LENS = [1024, 4096]

# (block_size, num_steps) for the synthetic regimes only - they have no spatial
# grid for a block size to align to, so only generic sizes mean anything there.
# Not good settings for real activations; see aligned_monarch_configs().
MONARCH_CONFIGS = [(32, 1), (32, 3), (64, 1)]

# The paper's convergence study (app. C.4): T=1 closes most of the gap and T=2
# is already stationary, so there is no point going past 3.
MONARCH_STEPS = [1, 3]

# A deliberate contrast row on captured data - B=32 scores ~6x worse than an
# aligned config on real WAN activations. Kept visible so that regression cannot
# quietly come back.
MONARCH_MISALIGNED_CONTRAST = 32

# ma_triton sizes tiles as next_pow2(B) x next_pow2(D) and next_pow2(N/B) x
# next_pow2(D). RDNA2 has 64KB of LDS per workgroup, and a 512-wide tile does not
# merely run slowly - Triton's ROCm backend thrashes in register allocation
# instead of failing (measured: >950s of compile CPU, no kernel produced).
MAX_MONARCH_TILE = 256

CAPTURED_QKV_DIR = Path(__file__).parent / ".data" / "captured_qkv"

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
    """Returns (name, q, k, v, grid) per capture. grid is the (f,h,w) token
    layout, either recorded by the capture hook or recovered by detect_grid();
    None when neither works, which costs only the aligned Monarch configs."""
    if not CAPTURED_QKV_DIR.is_dir():
        print(f"\n[CAPTURED regime] {CAPTURED_QKV_DIR} does not exist - skipping.")
        print('  Each file is a torch.save\'d dict of "q"/"k"/"v" tensors shaped '
              '[batch, heads, seq, head_dim], plus an optional "grid" of (f,h,w).')
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
        grid = d.get("grid")
        if grid is not None:
            grid = tuple(int(x) for x in grid)
            if grid[0] * grid[1] * grid[2] != q.shape[-2]:
                print(f"  [warn] {f.stem}: grid {grid} does not match seq "
                      f"{q.shape[-2]} - ignoring it")
                grid = None
        rel = f.relative_to(CAPTURED_QKV_DIR).parent
        name = f.stem if rel == Path(".") else f"{rel.as_posix()}/{f.stem}"
        out.append([name, q, k, v, grid])

    # Architectures that publish no grid get it recovered from the data, but the
    # grid belongs to the image layout rather than to any one block, and shallow
    # blocks often carry too little positional structure to detect at all
    # (measured on Anima: block 27 shows w=64 at 9.6x baseline while blocks
    # 0/9/18 are flat to within ~2x). So detect per capture, then share the
    # strongest detection across every capture of the same length. Scoped per
    # model directory too: two models can share a length and not a grid.
    def group(stem, q):
        return (stem.rsplit("/", 1)[0] if "/" in stem else "", q.shape[-2])

    best = {}
    for stem, q, k, v, grid in out:
        if grid is not None or q.shape[-2] != k.shape[-2]:
            continue
        detected, note, score = detect_grid(q.half(), k.half())
        if detected and score > best.get(group(stem, q), (None, 0.0, ""))[1]:
            best[group(stem, q)] = (detected, score, note)
    for item in out:
        stem, q, k, v, grid = item
        if grid is not None or q.shape[-2] != k.shape[-2]:
            continue
        hit = best.get(group(stem, q))
        if hit:
            item[4] = hit[0]
            print(f"  [grid] {stem}: (f,h,w)={hit[0]} from strongest detection "
                  f"at this length ({hit[2]})")
        else:
            print(f"  [grid] {stem}: no grid detected - aligned configs unavailable")
    return [tuple(x) for x in out]


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


def _diag_means(q, k, offsets, max_heads=8, max_rows=512):
    """Mean post-softmax attention at each requested diagonal offset, averaged
    over heads. Evaluated on an evenly-spaced subset of query rows: each row's
    softmax is still exact (all N keys participate), there are just fewer rows
    in the average. Two orders of magnitude cheaper than materializing every
    row, and the periods this looks for survive the subsampling."""
    _, heads, n, d = q.shape
    scale = 1.0 / (d ** 0.5)
    nh = min(heads, max_heads)
    rows = torch.arange(0, n, max(1, n // max_rows), device="cuda")[:max_rows]
    offs = torch.tensor([int(o) for o in offsets], device="cuda")
    cols = rows[:, None] + offs[None, :]
    valid = (cols < n).float()
    cols = cols.clamp_(max=n - 1)
    denom = valid.sum(0).clamp(min=1.0)
    acc = torch.zeros(len(offsets), dtype=torch.float64, device="cuda")
    for h_i in range(nh):
        qh = q[0, h_i].to("cuda", torch.float32)
        kh = k[0, h_i].to("cuda", torch.float32)
        a = torch.softmax(qh[rows] @ kh.T * scale, dim=-1)
        acc += ((a.gather(1, cols) * valid).sum(0) / denom).double()
        del a, qh, kh
        torch.cuda.empty_cache()
    return (acc / nh).cpu()


def detect_grid(q, k, min_ratio=4.0, max_heads=40):
    """Recover a token grid from attention band structure, for architectures that
    flatten to "b (t h w) d" without exporting (t,h,w). 3D attention has a
    separable positional component, so mean attention at diagonal offset delta
    peaks at delta=w and again at delta=h*w. Validated against WAN, where it
    recovered (11,28,18) at 34x and 31x over baseline.

    A single image has no frame period, so a run with no strong second peak is
    reported as (1, N/w, w). Returns (grid, note, score); grid is None when
    nothing convincing is found."""
    n = q.shape[-2]
    probe = list(range(1, min(257, n - 2)))
    prof = _diag_means(q, k, probe, max_heads)
    base = prof.median().item()
    if base <= 0:
        return None, "degenerate attention profile", 0.0

    # w is the divisor of n with the largest mean attention, deliberately not a
    # local-maximum test: some heads attend to the diagonal neighbour (row+1,
    # col-1), which lands at offset w-1 and can outrank w itself in a small head
    # subsample (WAN: peak at 17 over 8 heads, at the true 18 over 40). Averaging
    # over many heads is what makes a plain magnitude comparison reliable.
    # Offsets below 4 are skipped - near-diagonal attention is always strong and
    # small numbers divide almost everything.
    cands = [(prof[o - 1].item(), o) for o in range(4, probe[-1] + 1) if n % o == 0]
    if not cands:
        return None, "no candidate row period divides the sequence length", 0.0
    w_val, w = max(cands)
    w_ratio = w_val / base
    if w_ratio < min_ratio:
        return None, f"strongest period w={w} only {w_ratio:.1f}x baseline", w_ratio

    # Every multiple of w is elevated just by being a harmonic of the row period,
    # so magnitude alone gives false positives - on a 2D image grid w=64 sits at
    # 9.6x and 2w=128 at 5.2x, and taking 128 as a "frame" would invent a third
    # axis. A real frame period also rises above both its neighbours: on WAN,
    # 486/504/522 read 5.2x/10.4x/6.3x.
    frames = [f for f in range(2 * w, n, w) if n % f == 0]
    if frames:
        probe2 = sorted({o for f in frames for o in (f - w, f, f + w) if 0 < o < n})
        fp = _diag_means(q, k, probe2, max_heads)
        val = {o: fp[i].item() for i, o in enumerate(probe2)}
        best = None
        for f in frames:
            if val[f] < min_ratio * base:
                continue
            if val[f] <= val.get(f - w, 0.0) or val[f] <= val.get(f + w, 0.0):
                continue
            if best is None or val[f] > val[best]:
                best = f
        if best is not None:
            return (n // best, best // w, w), (
                f"w={w} at {w_ratio:.1f}x, frame={best} at {val[best] / base:.1f}x baseline"), w_ratio
    return (1, n // w, w), f"w={w} at {w_ratio:.1f}x baseline, no frame period (2D grid)", w_ratio


def _next_pow2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def aligned_monarch_configs(grid):
    """The video-aligned Monarch factorizations for a token grid (f,h,w).

    A parameterization is aligned iff each of f, h and w lies wholly inside one
    block dimension (MonarchRT sec 6.1). Splitting any one of them makes the
    separable positional structure of 3D attention unrepresentable, which is not
    a small effect - on real WAN activations a misaligned block size scores 0.76
    rel_rms where an aligned one scores 0.12. That admits exactly six
    configurations, excluding the degenerate dense ones.

    Ordered most-accurate-first by measurement, not theory: the paper treats the
    two halves of a split as interchangeable, but that holds only for its
    idealized purely-positional model. On real activations (w,fh) beats (fh,w) by
    1.4x because the alternating optimization is asymmetric - L initializes to
    identity and R updates first - so the small dimension belongs in b1.

    Each entry is (label, axis_order, b2), b2 being the contiguous-chunk block
    size under that ordering. Configs whose tiles exceed MAX_MONARCH_TILE are
    dropped; on WAN's grid that removes (f,hw) and (hw,f), which the roofline
    also ranks worst."""
    f, h, w = grid
    n = f * h * w
    sizes = {"f": f, "h": h, "w": w}
    out, seen = [], set()
    for label, order, b in (("(w,fh)", "wfh", f * h),
                            ("(h,fw)", "hfw", f * w),
                            ("(fh,w)", "fhw", w),
                            ("(fw,h)", "fwh", h),
                            ("(f,hw)", "fhw", h * w),
                            ("(hw,f)", "hwf", f)):
        if b < 1 or n % b:
            continue
        if max(_next_pow2(b), _next_pow2(n // b)) > MAX_MONARCH_TILE:
            continue
        # A 2D grid has f == 1, and an axis of extent 1 contributes nothing to the
        # ordering, so several of the six collapse onto each other - (w,fh) and
        # (fw,h) become the same transpose. Deduplicate on the effective
        # permutation so the table does not repeat a config.
        key = ("".join(c for c in order if sizes[c] > 1), b)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, order, b))
    return out


def dense_flops(n, d, heads):
    """QK^T then AV, both 2*n*n*d. Softmax is O(n^2) elementwise and not counted -
    real work, but not multiply-accumulate, and counting it would make these
    numbers incomparable to the usual attention-FLOPs figure."""
    return heads * 4 * n * n * d


def monarch_flops(n, d, heads, block, steps):
    """Untiled MonarchAttention, counted off the paper's pseudocode (fig. 4):
    each non-final step runs bmm(aR,Kb^T), bmm(R,Kb), bmm(aL,Qb^T), bmm(L,Qb) =
    4*n*d*(b+m); the final step instead runs al_y_cl (three b-side bmms) and
    z_kernel (two m-side), i.e. n*d*(6b+4m). b is the contiguous block size,
    m = n/b."""
    b, m = block, n // block
    return heads * ((steps - 1) * 4 * n * d * (b + m) + n * d * (6 * b + 4 * m))


def tiled_monarch_flops(n, d, heads, tiles, b1, b2, steps):
    """MonarchRT's tiled Monarch, counted off the einsums in
    _monarch_attention_chunk(). Every contraction carries both a query-tile index
    and a key-tile index, so cost picks up a factor of tiles^2, which against
    tiles*b1*b2 == n leaves cost proportional to tiles*(b1+b2) - tiling buys
    accuracy at a linear price. Non-final step: r_logits + a_l + l_logits + a_r.
    Final step: r_logits + a_l + l_logits + y + out. At tiles=1 this reduces to
    the untiled count above, as it must."""
    per_step = 4 * heads * n * d * tiles * (b1 + b2)
    final = 2 * heads * tiles * n * d * (2 * b1 + 3 * b2)
    return (steps - 1) * per_step + final


def tiled_monarch_configs(grid, max_tiles=64, limit=6):
    """A ladder of tilings from coarse to fine, one per distinct tile count.
    Among configs with the same tile count, prefers keeping w whole: at a fixed
    16-tile budget on real WAN activations, (7,18) with w intact scored 0.0836
    against (14,9) with w split at 0.0925 - same compute, better error."""
    f, h, w = grid
    cands = []
    for hr in (x for x in range(1, h + 1) if h % x == 0):
        for wr in (x for x in range(1, w + 1) if w % x == 0):
            tiles = f * hr * wr
            if tiles > max_tiles or h // hr < 2 or w // wr < 2:
                continue
            cands.append((tiles, wr != 1, hr, wr))
    out, seen = [], set()
    for tiles, _, hr, wr in sorted(cands):
        if tiles in seen:
            continue
        seen.add(tiles)
        out.append((1, hr, wr))
        if len(out) >= limit:
            break
    return out


def eval_tiled_monarch(q64, k64, v64, ref, grid, f_tied, h_reduce, w_reduce, num_iters):
    # MonarchRT expects BSHD; captures are BHSD. random_seed is pinned so the
    # q-init is reproducible run to run.
    f, h, w = grid
    q, k, v = (t.to("cuda", dtype=torch.float16).transpose(1, 2).contiguous()
               for t in (q64, k64, v64))
    got = tiled_monarch_attn(q, k, v, f_tied, h_reduce, w_reduce, h, w,
                             num_iters=num_iters, random_seed=0)
    return rel_rms(got.transpose(1, 2), ref)


def misaligned_contrast_block(grid, n):
    """A block size that genuinely splits w, kept as a contrast row - on real WAN
    activations block sizes keeping w whole land near 0.27 and ones splitting it
    near 0.76. Picks the candidate nearest MONARCH_MISALIGNED_CONTRAST so the row
    stays comparable across grids."""
    w = grid[2]
    best = None
    for b in range(2, n // 2 + 1):
        # Judged in natural (f,h,w) order, where a contiguous block of b tokens
        # keeps w whole exactly when b is a multiple of w. Divisors of w do NOT
        # keep it whole - a block of w/2 is half a row - so only multiples are
        # excluded. Alignment is order-dependent (b=h is aligned under (fw,h) but
        # splits rows under (f,h,w)), so aligned sizes are not excluded.
        if n % b or b % w == 0:
            continue
        if max(_next_pow2(b), _next_pow2(n // b)) > MAX_MONARCH_TILE:
            continue
        if best is None or abs(b - MONARCH_MISALIGNED_CONTRAST) < abs(best - MONARCH_MISALIGNED_CONTRAST):
            best = b
    return best


def monarch_permutation(grid, order):
    """idx[p] = index, in natural (f,h,w) order, of the p-th token under `order`.
    Dense attention is permutation-equivariant, so permuting q/k/v and
    un-permuting the output is exact; only how well the Monarch block structure
    fits the ordering changes."""
    axes = {"f": 0, "h": 1, "w": 2}
    return (torch.arange(grid[0] * grid[1] * grid[2])
            .reshape(*grid).permute(*[axes[c] for c in order]).reshape(-1))


def eval_monarch(q64, k64, v64, ref, block_size, num_steps,
                 grid=None, order="fhw") -> Optional[float]:
    """None for cross-attention: the upstream Triton kernel derives its sequence
    length from q.shape alone and never reconciles it against k, so an Nq != Nk
    call silently returns a wrong-shaped result rather than raising. Confirmed by
    direct test, and consistent with the algorithm's math, which has no Nq/Nk
    distinction to begin with."""
    if q64.shape[-2] != k64.shape[-2]:
        return None
    q, k, v = (t.to("cuda", dtype=torch.float16) for t in (q64, k64, v64))
    if grid is not None and order != "fhw":
        idx = monarch_permutation(grid, order).to(q.device)
        q, k, v = (t[:, :, idx].contiguous() for t in (q, k, v))
        got = monarch_attention_triton(q, k, v, None, num_steps, block_size, False)
        return rel_rms(got[:, :, torch.argsort(idx)], ref)
    got = monarch_attention_triton(q, k, v, None, num_steps, block_size, False)
    return rel_rms(got, ref)


def run_regime(name: str, cases):
    print(f"\n=== {name} ===")
    name_w = max(24, max((len(c) for c, *_ in cases), default=24) + 2)
    header = f"{'case':<{name_w}}{'FP16':<12}{'INT8':<12}" + "".join(
        f"MA(B={b},T={t})".ljust(16) for b, t in MONARCH_CONFIGS)
    print(header)
    print("-" * len(header))
    for case_name, q64, k64, v64 in cases:
        ref = dense_attention_reference(q64, k64, v64)
        is_cross_attn = q64.shape[-2] != k64.shape[-2]

        row = f"{case_name:<{name_w}}"
        row += f"{eval_fp16(q64, k64, v64, ref):.4f}".ljust(12)
        int8_err = eval_int8(q64, k64, v64, ref)
        row += f"{'n/a' if int8_err is None else f'{int8_err:.4f}':<12}"
        if MONARCH_AVAILABLE:
            for block_size, num_steps in MONARCH_CONFIGS:
                if is_cross_attn:
                    row += "n/a (cross-attn)".ljust(16)
                    continue
                try:
                    err = eval_monarch(q64, k64, v64, ref, block_size, num_steps)
                    row += f"{err:.4f}".ljust(16)
                except Exception as e:
                    row += f"FAIL({type(e).__name__})".ljust(16)
        print(row)
        del q64, k64, v64, ref
        torch.cuda.empty_cache()


def run_captured_regime(cases):
    # Per-case layout rather than the shared table: each capture carries its own
    # grid, so the set of Monarch configs worth running differs case by case and
    # cannot be a fixed column set.
    print("\n=== CAPTURED (real model activations - the only regime that means "
          "anything for a lossy backend) ===")
    for case_name, q64, k64, v64, grid in cases:
        n_q, n_k, head_dim = q64.shape[-2], k64.shape[-2], q64.shape[-1]
        print(f"\n{case_name}")
        print(f"  grid (f,h,w)={grid if grid else 'unknown'}  Nq={n_q} Nk={n_k} D={head_dim}")
        ref = dense_attention_reference(q64, k64, v64)

        print(f"    {'backend':<32}{'rel_rms':<10}{'FLOPs':<10}note")
        print(f"    {'FP16':<32}{eval_fp16(q64, k64, v64, ref):<10.4f}{'1.000x':<10}exact-attention cost")
        int8_err = eval_int8(q64, k64, v64, ref)
        print(f"    {'INT8 (per-tensor)':<32}"
              + (f"{'n/a':<10}{'-':<10}head_dim unsupported" if int8_err is None
                 else f"{int8_err:<10.4f}{'1.000x':<10}exact-attention cost"))
        pc_err = eval_int8_perchannel_v(q64, k64, v64, ref)
        if pc_err is not None:
            gain = f"{int8_err / pc_err:.2f}x vs per-tensor" if int8_err else ""
            print(f"    {'INT8 (per-channel V)':<32}{pc_err:<10.4f}{'1.000x':<10}{gain}")

        if MONARCH_AVAILABLE:
            run_monarch_rows(q64, k64, v64, ref, grid, n_q, n_k, head_dim)

        del ref
        torch.cuda.empty_cache()


def run_monarch_rows(q64, k64, v64, ref, grid, n_q, n_k, head_dim):
    if n_q != n_k:
        # eval_monarch() would return None - the upstream kernel cannot do
        # Nq != Nk at all.
        print(f"    {'MonarchAttention':<34}n/a (cross-attn)")
        return
    if not grid:
        print(f"    {'MonarchAttention':<34}skipped - capture has no grid recorded, so no "
              "aligned\n      block size can be derived (re-capture to measure this properly)")
        return

    heads = q64.shape[1]
    dense = dense_flops(n_q, head_dim, heads)

    def row(tag, err, flops, note):
        cost = "-" if flops is None else f"{flops / dense:.3f}x"
        print(f"    {tag:<32}{err:<10.4f}{cost:<10}{note}")

    for label, order, b in aligned_monarch_configs(grid):
        for t in MONARCH_STEPS:
            row(f"MA {label} = ({n_q // b},{b})  T={t}",
                eval_monarch(q64, k64, v64, ref, b, t, grid, order),
                monarch_flops(n_q, head_dim, heads, b, t), "aligned")
    b = misaligned_contrast_block(grid, n_q)
    if b is not None:
        t = max(MONARCH_STEPS)
        row(f"MA B={b}  T={t}",
            eval_monarch(q64, k64, v64, ref, b, t),
            monarch_flops(n_q, head_dim, heads, b, t), "MISALIGNED (contrast)")

    if not MONARCHRT_AVAILABLE:
        return
    f, h, w = grid
    # tiles=1 collapses tiled Monarch to untiled (f*h, w), reproducing the
    # MA (fh,w) row above - a cross-check of the two independent implementations.
    # On a 2D grid f == 1, so (f,1,1) is the same entry the ladder starts with.
    configs, seen_cfg = [], set()
    for c in [(f, 1, 1)] + tiled_monarch_configs(grid):
        if c not in seen_cfg:
            seen_cfg.add(c)
            configs.append(c)
    for ft, hr, wr in configs:
        if f % ft or h % hr or w % wr:
            continue
        tiles = (f // ft) * hr * wr
        b1, b2 = ft * (h // hr), w // wr
        note = "tiled" if tiles > 1 else "tiled, collapses to (fh,w)"
        for t in MONARCH_STEPS:
            try:
                err = eval_tiled_monarch(q64, k64, v64, ref, grid, ft, hr, wr, t)
            except RuntimeError as e:
                print(f"    {f'MRT {tiles}x ({b1},{b2})  T={t}':<32}"
                      f"FAILED ({type(e).__name__})")
                torch.cuda.empty_cache()
                continue
            row(f"MRT {tiles}x ({b1},{b2})  T={t}", err,
                tiled_monarch_flops(n_q, head_dim, heads, tiles, b1, b2, t), note)
            torch.cuda.empty_cache()


def run():
    if not rdnattention.has_device():
        print("No usable HIP device found.")
        sys.exit(1)
    print(f"torch/ROCm device: {torch.cuda.get_device_name(0)}")
    print(f"MonarchAttention available: {MONARCH_AVAILABLE} (tiled: {MONARCHRT_AVAILABLE})")

    random_cases = []
    structured_cases = []
    for head_dim in HEAD_DIMS:
        for seq_len in SEQ_LENS:
            name = f"H{NUM_HEADS} N{seq_len} D{head_dim}"
            random_cases.append((name, *make_random_qkv(1, NUM_HEADS, seq_len, head_dim)))
            structured_cases.append((name, *make_structured_qkv(1, NUM_HEADS, seq_len, head_dim)))

    run_regime("RANDOM (floor/regression check only for the approximate backends)", random_cases)
    run_regime("STRUCTURED (synthetic positional correlation - proxy only)", structured_cases)

    captured_cases = load_captured_qkv()
    if captured_cases:
        run_captured_regime(captured_cases)

    print("\nrel_rms = sqrt(sum((got-ref)^2) / sum(ref^2)) against exact fp64 dense attention.")


if __name__ == "__main__":
    run()
