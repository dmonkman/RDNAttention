<h1 align="center">RDNAttention</h1>

<p align="center">
  <strong>Native HIP FlashAttention-2 forward attention for AMD consumer GPUs</strong><br>
  No tensor cores. No Vulkan. No Triton. Just the dot-product ALUs the hardware already has.
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#features">Features</a> |
  <a href="#supported-hardware">Hardware</a> |
  <a href="#install">Install</a> |
  <a href="#build">Build</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="docs/api.md">API Reference</a>
</p>

---

**Version 0.1.0** - forward pass only. Inference, not training.

## Overview

RDNAttention is a FlashAttention-2-style attention kernel written directly in HIP for
AMD RDNA and GCN5 hardware - the consumer cards that have no matrix cores and that
upstream FlashAttention does not target. It builds the GEMMs out of native
dot-product ALU instructions (`v_dot2_f32_f16`, `v_dot4_i32_i8`) rather than WMMA, so
it runs on parts as old as a Radeon VII and as small as a Steam Deck APU.

The Python package is a thin `ctypes` wrapper. It reads the device pointer out of your
torch tensor and calls the kernel on it - no staging copies, no second GPU context, no
compilation at import time.

Two kernels ship:

- **FP16** (`fa2_forward_f16.hip`) - the general path. head_dim 32-512, RoPE in-kernel.
- **INT8** (`fa2_forward_int8qk.hip`) - both GEMMs in INT8, measured at 1.3-1.7x the
  fp16 kernel's throughput depending on shape, at a measurable accuracy cost.

## Features

| | FP16 kernel | INT8 kernel |
|---|---|---|
| MHA / GQA / MQA | ✅ (GQA K/V reuse, `G=2`) | ✅ (GQA by K/V re-read) |
| Causal masking | ✅ | ✅ |
| Sliding window (one- or two-sided) | ✅ | ✅ |
| RoPE, applied in-kernel | ✅ | ❌ |
| Cross-attention and decode (`seq_len != key_seq_len`) | ✅ | ✅ |
| Non-contiguous batch/head strides | ✅ | ✅ |
| head_dim | 32-512, any multiple of 32¹ | 64, 128 |
| Per-token Q/K quantization scales | n/a | ✅ |
| Per-channel V quantization scale | n/a | ✅ |

¹ head_dim 64 and 128 have measured tile sizes; other sizes are correct but untuned.
Prefer a multiple of **64** where the choice is yours - `head_dim % 64 == 0` lets the
compiler merge GEMM2's LDS reads, worth 36-49% at the sizes where it applies.

**Not implemented:** backward pass, dropout, ALiBi/relative position bias, softcap,
varlen (`cu_seqlens`), paged KV cache, BF16/FP8, arbitrary attention masks, and a
custom softmax scale (fixed at `1/sqrt(head_dim)`).

## Supported hardware

**Only gfx1030 (RX 6800 XT) is confirmed by actual runs.** Everything else is an
architecture-level claim: instruction availability was probed against the ROCm 7.2
assembler, and "builds" means both kernels compile for that target.

### Tier 1 - both kernels, native dot-product ALUs

The default build produces a fat binary covering all ten targets below.

| Gen | `gfx` | Retail parts | Status |
|---|---|---|---|
| RDNA2 | gfx1030 | RX 6800, 6800 XT, 6900 XT, 6950 XT, Pro W6800 | **Tested** |
| RDNA2 | gfx1031, gfx1032 | RX 6700 XT / 6750 XT, RX 6600 / 6600 XT / 6650 XT | Builds |
| RDNA2 | gfx1033-gfx1036 | Steam Deck APU, RX 6400 / 6500 XT, Radeon 680M/660M, Ryzen 7000 iGPU | Builds |
| RDNA1 | gfx1011, gfx1012 | Radeon Pro 5600M, RX 5500 / 5500 XT / 5300 | Builds |
| GCN5 | gfx906 | Radeon VII, Instinct MI50/MI60, Radeon Pro VII | Builds |

RDNA3, RDNA3.5 and RDNA4 (gfx1100-gfx1103, gfx1150/1151, gfx1200/1201 - the RX 7000
and RX 9000 series, Radeon 780M/890M) have both dot instructions and compile cleanly,
but are not in the default fat binary. Name them explicitly:

```
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

### Tier 2 - fp16 only, no dot-product ALUs

These parts have neither `v_dot2_f32_f16` nor `v_dot4_i32_i8`. The fp16 kernel runs
through a portable fallback; INT8 is refused rather than emulated, since an emulated
`sdot4` would run roughly 3x slower than simply using the fp16 kernel on the same
card. They need an fp16-only build:

```
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx1010 -DRDNATTENTION_ENABLE_INT8=OFF
```

| Gen | `gfx` | Retail parts | fp16 fallback cost |
|---|---|---|---|
| RDNA1 | gfx1010 | RX 5700 XT, RX 5700, RX 5600 XT | +15.6% VALU (`v_fma_mix_f32`) |
| GCN5 | gfx904 | Radeon Pro Vega 16/20 | +15.6% VALU (`v_fma_mix_f32`) |
| GCN5 | gfx900, gfx902, gfx90c | Vega 56/64, Vega Frontier, Vega 8/11 and Ryzen 4000G/5000G iGPUs | +25.3% VALU (`v_mad_mix_f32`, inline asm) |
| GCN4 | gfx803 | Polaris, RX 400/500 series | +36.9% VALU (cvt + FMA) |

VALU counts understate the real cost, because the extra instructions also cost
registers and occupancy. Forcing each lowering on gfx1030 and measuring, the mix-FMA
shape retains ~70% of native throughput and the cvt+FMA shape ~50%. Accuracy is
unaffected either way - the fallbacks keep the product in fp32 exactly as `v_dot2`
does, so a lower tier costs speed and not precision.

[`FEATURES.md`](FEATURES.md) has the full per-GPU instruction matrix and the tiering;
[`docs/hardware.md`](docs/hardware.md) has the per-architecture resource budgets.

## Install

```
pip install rdnattention
```

There is one wheel per platform - Linux x86_64 (glibc 2.28+) and Windows x64 - and
each covers every Python from 3.11 up, since the package is pure Python over a
`ctypes`-loaded library.

The wheels are built with ROCm 10.0.0 (AMD's pip-installed `rocm[devel]`) and link the
HIP 7 runtime, `libamdhip64.so.7` / `amdhip64_7.dll`, without bundling it. They work
in any environment that provides HIP 7 - tested against ROCm 7.2 and 10.0. The
runtime is taken from the pip-installed ROCm that a ROCm build of torch brings
along, falling back to `HIP_PATH` on Windows or the system library path on Linux.
Install torch first; its ROCm builds do not come from PyPI, so `rdnattention` does
not declare it as a dependency.

The wheels cover the [Tier 1](#tier-1---both-kernels-native-dot-product-alus)
targets. Anything else needs a source build.

## Build

Requirements: C++20, CMake >= 3.21, and a ROCm install with HIP. The `HIP_PATH` env
var selects which ROCm install CMake finds - keep the compiler and headers from the
same version. Python 3.11+ and a ROCm build of torch are needed only for the Python
bindings.

```
cmake -S . -B build
cmake --build build
```

Then install the package. In a source checkout the build output in `build/` is
preferred over any packaged copy in `rdnattention/lib/`, so a stale library cannot
shadow a rebuild:

```
pip install -e .
```

> **Do not leave `CMAKE_BUILD_TYPE` unset.** At `-O0` the vector-subscript LDS writes
> in the fp16 kernel's V path compile to real read-modify-writes, which race between
> the threads sharing a row-pair and silently corrupt output. `CMakeLists.txt` forces
> a default of `Release` for exactly this reason - don't remove that guard.

## Quick start

```python
import torch
from rdnattention import flash_attn

q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")

out = flash_attn(q, k, v, is_causal=True)
```

Under ROCm, torch still spells the device `"cuda"`. Tensors are
`[batch, heads, seq_len, head_dim]` and must be contiguous. Give K and V fewer heads
than Q for GQA/MQA, or a different sequence length for cross-attention and decode -
no extra arguments either way.

INT8 takes ordinary fp16 input and quantizes it for you:

```python
from rdnattention import flash_attn_int8qk_quantized

out = flash_attn_int8qk_quantized(q, k, v, is_causal=True)
```

[`docs/api.md`](docs/api.md) documents every entry point: the raw INT8 kernel and its
scale contract, the quantization helpers, RoPE tables, masking and window semantics,
error types, and the limits.

## Tests

The two correctness gates are standalone `hipcc` builds, not wired into CMake, and
neither needs torch. Both compare against a from-scratch fp64 CPU oracle. Run them
after any kernel change:

```
hipcc tests/hip_forward_gate.hip src/rdna/fa2_forward_f16.hip \
    -o hip_forward_gate --offload-arch=gfx1030 -O2 -std=c++17
./hip_forward_gate

hipcc tests/hip_forward_int8qk_gate.hip src/rdna/fa2_forward_int8qk.hip \
    -o hip_forward_int8qk_gate --offload-arch=gfx1030 -O3 -ffast-math -std=c++17
./hip_forward_int8qk_gate
```

Two more checks in the same spirit - the resource audit needs no GPU at all, only
`hipcc`:

```
python tests/head_dim_resources.py     # fails on any VGPR spill beyond its allowance
python tests/head_dim.py               # 144 cases: 16 head_dims x 9 variants
```

Benchmarks:

```
hipcc tests/hip_bench.hip src/rdna/fa2_forward_f16.hip -o hip_bench \
    --offload-arch=gfx1030 -O3 -ffast-math -std=c++17
./hip_bench tuned      # head_dim 64/128, for A/B against a known-good build
./hip_bench headdim    # one shape per dispatched head_dim
```

To prove a refactor left the tuned paths alone, build `hip_bench` twice - once
against the old `fa2_forward_f16.hip`, once against the new - and interleave the runs.
Do not run one build to completion and then the other: clock drift across a session is
larger than the margins being measured.

## Gotchas worth knowing before editing the kernels

- `hipcc` silently forces `-O3` on device code; CMake's HIP language support (raw
  `clang++`) respects `CMAKE_BUILD_TYPE` instead - the two can produce different
  codegen for the same source.
- GQA K/V-head-group size is `G=2`, not the textbook `G=4` - `G=4` was measured to
  regress via register pressure (221 VGPRs / occupancy 4); `G=2` (141 VGPRs /
  occupancy 6) is the actual win. Don't "fix" this back to 4 without re-benchmarking.
- Never combine performance counters from the same hardware block in one
  `rocprofv3 --pmc` invocation (e.g. `LDSBankConflict` + `SQC_LDS_BANK_CONFLICT`) -
  this hung a GPU/driver in this project's history.
- Several constants are measured, not derived - `G=2`, the BR=128 tile choice,
  `kTileVgprCap`. Re-benchmark before "correcting" one.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

- [Fable Attention](https://github.com/2kiss/flash-attention-rdna2) - another high
  performance HIP compute kernel that directly inspired some of the techniques used
  here. Still used as a competitive benchmark.
- [Aule Attention](https://github.com/AuleTechnologies/Aule-Attention) - hardware-agnostic
  FlashAttention with Triton and Vulkan backends. Inspired this project by showing how
  much performance default PyTorch was leaving on the table for the target hardware.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao and the
  Dao-AILab team, for the algorithm this is a re-implementation of.
