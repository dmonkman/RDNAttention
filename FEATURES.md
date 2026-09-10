# Attention Feature Support Matrix

Per-architecture LDS, register, occupancy and cache resources - as opposed to
instructions - are in [`docs/hardware.md`](docs/hardware.md).

Legend: ✅ native hardware path &nbsp; 🟡 works, emulated/no acceleration &nbsp; ❌ not available

Only gfx1030 (RX 6800 XT) is confirmed by actual runs. Everything else is an architecture-level
claim: instruction availability was probed against the ROCm 7.2 assembler (`llvm-mc -mcpu=<arch>`)
and "builds" means both kernels compile for that target - neither is a substitute for running them.

## Per-GPU instruction support

Dot-product hardware does **not** split by generation. RDNA1 and GCN5 each contain both
dot-capable and dot-less parts, so the per-generation table further down cannot express the real
boundary - this table is the authority.

Two instructions the kernels rely on are available on **every** architecture listed here and so
have no column: `v_mad_i32_i24` (the int8 fallback multiply) and the DPP row ops the softmax
reduction uses (`quad_perm`, `row_mirror`, `row_half_mirror`). The DPP reduction operates on
rows of 16 lanes, which is 16 lanes on both wave32 and wave64, so it is wave-size agnostic and
works unchanged on wave64-only GCN parts.

| Retail part | `gfx` | Gen | `v_dot2_f32_f16` | `v_dot4_i32_i8` | Mixed f16→f32 FMA | WMMA | Kernels build |
|---|---|---|---|---|---|---|---|
| RX Vega 56 / 64, Vega Frontier, Pro WX 8200/9100 | gfx900 | GCN5 | ❌ | ❌ | `v_mad_mix_f32`¹ | ❌ | fp16 only |
| Ryzen 2000G/3000G iGPU (Vega 8/11) | gfx902 | GCN5 | ❌ | ❌ | `v_mad_mix_f32`¹ | ❌ | fp16 only |
| Radeon Pro Vega 16/20 (MacBook Pro) | gfx904 | GCN5 | ❌ | ❌ | `v_fma_mix_f32` | ❌ | fp16 only |
| **Radeon VII, Instinct MI50/MI60, Pro VII** | gfx906 | GCN5 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| Ryzen 4000G/5000G, Ryzen 4000/5000 mobile iGPU | gfx90c | GCN5 | ❌ | ❌ | `v_mad_mix_f32`¹ | ❌ | fp16 only |
| **RX 5700 XT, RX 5700, RX 5600 XT** | gfx1010 | RDNA1 | ❌ | ❌ | `v_fma_mix_f32` | ❌ | fp16 only |
| Radeon Pro 5600M, Pro V520 | gfx1011 | RDNA1 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| **RX 5500 XT, RX 5500, RX 5300** | gfx1012 | RDNA1 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| **RX 6800, 6800 XT, 6900 XT, 6950 XT, Pro W6800** | gfx1030 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ (tested) |
| RX 6700 XT, RX 6750 XT | gfx1031 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| RX 6600, RX 6600 XT, RX 6650 XT | gfx1032 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| Steam Deck APU (Van Gogh) | gfx1033 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| RX 6500 XT, RX 6400 | gfx1034 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| Radeon 680M/660M (Ryzen 6000 mobile, Rembrandt) | gfx1035 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| Ryzen 7000 desktop iGPU (Raphael) | gfx1036 | RDNA2 | ✅ | ✅ | `v_fma_mix_f32` | ❌ | ✅ |
| RX 7900 XTX / XT / GRE | gfx1100 | RDNA3 | ✅ | ✅ | `v_fma_mix_f32` | ✅ | ✅ |
| RX 7800 XT, RX 7700 XT | gfx1101 | RDNA3 | ✅ | ✅ | `v_fma_mix_f32` | ✅ | ✅ |
| RX 7600, RX 7600 XT | gfx1102 | RDNA3 | ✅ | ✅ | `v_fma_mix_f32` | ✅ | ✅ |
| Radeon 780M (Ryzen 7040, Phoenix) | gfx1103 | RDNA3 | ✅ | ✅ | `v_fma_mix_f32` | ✅ | ✅ |
| Radeon 890M (Ryzen AI 300, Strix) | gfx1150/1151 | RDNA3.5 | ✅ | ✅ | `v_fma_mix_f32` | ✅ | ✅ |
| RX 9060 XT | gfx1200 | RDNA4 | ✅ | ✅ | `v_fma_mix_f32` | ✅² | ✅ |
| RX 9070, RX 9070 XT | gfx1201 | RDNA4 | ✅ | ✅ | `v_fma_mix_f32` | ✅² | ✅ |

¹ Vega10-class parts have the **unfused** `v_mad_mix_f32`, not `v_fma_mix_f32`. ROCm 6.4's backend
would not form it from any C++ source pattern tried (including `-ffast-math` and
`-ffp-contract=fast`), so reaching it needs inline asm - see the fallback-cost note below.
² RDNA4 WMMA uses a different operand arity than RDNA3 (4-VGPR sources rather than 8).

Bold rows are the parts most likely to matter: the shipping target, and the two dot-less consumer
cards (RX 5700 XT, Vega 56/64) that a fallback would exist to serve.

### The dot-less fallback: how it works and what it costs

`fdot2()` in [`src/rdna/arch.hpp`](src/rdna/arch.hpp) selects per target. Dot-capable hardware
takes the builtin, unchanged - verified byte-identical ISA on gfx1030 and gfx906 before and after
the fallback was introduced, so Tier 1 pays nothing for this. Dot-less targets take a portable
two-FMA form that clang lowers to whatever the target has.

Per dot operation, from emitted ISA (address arithmetic excluded):

| Operation | Native | gfx1010 / gfx904 / gfx1013 | gfx900-class | gfx803 |
|---|---|---|---|---|
| `fdot2` - 2 × fp16 MAC → fp32 | 1 × `v_dot2c_f32_f16` | 2 × `v_fma_mix_f32` | 2 × `v_mad_mix_f32` (inline asm) | 6 (4 × `v_cvt_f32_f16` + 2 × `v_fma_f32`) |
| `sdot4` - 4 × int8 MAC → int32 | 1 × `v_dot4c_i32_i8` | no fallback, by design | no fallback, by design | no fallback, by design |

Whole-kernel VALU for `fa2_forward_f16.hip`, which is the number that actually matters - the dots
are only part of the work, and operand conversions get reused across the register tile, so the
real cost is well below what the per-dot figures suggest:

| Target class | Total VALU | vs Tier 1 |
|---|---|---|
| Tier 1 (gfx1030) | 27,729 | baseline |
| `v_fma_mix_f32` parts (RX 5700 XT) | 32,065 | **+15.6%** |
| `v_mad_mix_f32` parts (Vega 56/64) | 34,741 | **+25.3%** |
| gfx803 (Polaris), no mixed FMA at all | 37,953 | **+36.9%** |

Vega10-class needs hand-written inline asm to reach 2 VALU: the hardware has `v_mad_mix_f32`, but
ROCm's backend will not form it from any C++ source pattern tried (including `-ffast-math` and
`-ffp-contract=fast`). Writing it out costs 3,212 VALU less than letting the portable form fall to
cvt+fma. The `op_sel` encoding is **verified on hardware, not assumed**: `v_mad_mix_f32` on gfx900
and `v_fma_mix_f32` on gfx906 assemble to byte-identical machine code, and the same `op_sel`
tested against the `v_dot2` builtin on gfx1030 matched within one fp32 ULP across 65,536 random
inputs (96.6% bit-exact; the residual is two roundings against a fused dot's one). What remains
unverified for Vega specifically is only that its silicon executes a documented ISA instruction as
documented, and `mad`'s unfused rounding versus `fma`'s fused.

The `v_pk_mul_f16` route was rejected outright: 5 VALU per dot2 *and* it rounds the product to
fp16 before widening.

All four lowerings are executable on one GPU via `python tests/dot_paths.py --isa`, which builds
a library per path with `-DRDNATTENTION_FORCE_DOT_PATH` and checks each against an fp64 reference
and against the native path's output. Note the portable path folds back into `v_dot2` on
dot-capable hardware, so there it covers the arithmetic rather than the `v_fma_mix_f32` lowering;
paths 3 and 4 are what exercise the fallbacks.

| Forced path | What it runs | gfx1030 VALU | rel_rms vs native |
|---|---|---|---|
| 1 `builtin` | `v_dot2c_f32_f16` (ships on Tier 1) | 27,729 | baseline |
| 2 `portable` | two `fmaf`; refolds to `v_dot2` here | 27,729 | 0 (identical ISA) |
| 3 `mixasm` | 2 x `v_fma_mix_f32` - the Tier 2 shape | 32,069 | 2.1e-05 |
| 4 `cvtfma` | cvt+fma, no dot and no mix - the Tier 3c shape | 36,869 | 2.1e-05 |

Path 3's 32,069 is within 4 instructions of what clang generates unaided on gfx1010, which is
the check that the hand-written asm is not costing anything.

Path 4 exists only to be forced; no target selects it, because gfx803 reaches the same code
through path 2's automatic lowering. It pins only the `v_cvt_f32_f16` pair in asm and leaves the
fmas in C++, deliberately: putting all six instructions in one asm block also blocks CSE and
re-converts every operand on every call (49,477 VALU), whereas real no-mix targets share converts
across calls - gfx900's own portable codegen emits roughly one convert per dot. Split this way
gfx1030 lands at 36,869 against the 37,953 gfx900 actually pays, so it is a fair proxy rather
than a worst case. It is mildly pessimistic in one direction only: RDNA has no
single-instruction high-half f16->f32 (no SDWA, and the assembler rejects `op_sel` on
`v_cvt_f32_f16`), so it spends an extra `v_lshrrev` per pair that gfx803 does not.

Because those paths run, they can also be *timed*: `tests/benchmark.py` reports
`Tier2 FP16 (mix)` and `Tier3 FP16 (cvtfma)` columns alongside the native one. Read them as the
cost of losing `v_dot2` **on this card** - only the dot lowering varies, so they say nothing
about a Vega 64's or RX 5700 XT's clocks, cache, bandwidth, or wave64 (which would change the
DPP reduction cost these numbers hold fixed). Pass `--no-tiers` to skip building them.

Measured over the full 144-row sweep on gfx1030 (throughput retained against Tier 1, median and
range across 120 rows; the 24 decode rows are excluded as they sit at 0.05-0.35 TFLOP/s where
the kernel is parallelization-bound, not ALU-bound):

| | Tier 2 (`v_fma_mix_f32`) | Tier 3c (cvt+fma) |
|---|---|---|
| all shapes | **70%** (65-87%) | **50%** (38-63%) |
| head_dim 64 | 74% (65-87%) | 53% (48-63%) |
| head_dim 128 | 69% (65-87%) | 45% (38-63%) |

So dropping `v_dot2` costs roughly 1.4x on Tier 2 and 2.0x on Tier 3c - materially more than the
+15.6% / +33% VALU counts suggest, because the extra instructions also cost registers and
occupancy. head_dim 128 suffers more, which is consistent with it already being the tighter
register budget.

Accuracy is unaffected, and that is the point of carrying those columns in the accuracy table
too: across all 48 accuracy configs both fallbacks stay within 0.64% of the native path's own
`rel_rms`. The fallbacks keep the product in fp32 exactly as `v_dot2` does, so a lower tier
costs speed and not precision.

INT8 is the opposite story and is refused rather than emulated. At ~12 VALU per `sdot4` it would
cost 3.0 VALU/element against the fp16 fallback's 1.0, so an emulated INT8 kernel would run
roughly 3× slower than simply using the fp16 kernel on the same hardware, and less accurately.
Building for a dot-less target therefore requires `-DRDNATTENTION_ENABLE_INT8=OFF`; CMake fails
with that instruction rather than letting the build hit a confusing `#error` later.

## Hardware-gated (the only rows where architecture actually changes anything)

Per-generation summary. Where a generation splits, the table above governs.

| Feature | RDNA2 | RDNA1 | GCN5/Vega | iGPU (RDNA3) |
|---|---|---|---|---|
| `v_dot2` register-tiled GEMM (fp16 mixed-precision dot) | ✅ | ✅ except gfx1010¹ | ✅ gfx906 only² | ✅ |
| INT8 dot-product (native DP4A-style) | ✅ | ✅ except gfx1010¹ | ✅ gfx906 only² | ✅ |
| DPP cross-lane reduction (this project's 16-lane form) | ✅ | ✅ | ✅ (wave64, verified to emit) | ✅ |
| Wave32 | ✅ | ✅ | ❌ (wave64-only) | ✅ |
| BF16 | 🟡 (no native accel) | 🟡 | 🟡 | ✅ |
| FP8 (E4M3/E5M2) | ❌ | ❌ | ❌ | ❌³ |
| This project's GQA K/V-reuse (shared workgroup) | ✅ (measured) | 🟡 (untested) | 🟡 (untested) | 🟡 (untested) |
| head_dim = 256 (LDS budget) | ✅ | ✅ | 🟡 (fits on Vega, not older GCN) | ✅ |
| head_dim = 512 (LDS budget, Br=16) | ✅ | ✅ | 🟡 (untested) | ✅ |

¹ RDNA1 splits: gfx1011/gfx1012 (Radeon Pro 5600M, RX 5500 XT) have both dot instructions;
gfx1010 (RX 5700 XT) has neither. Dot ALUs were not "added in RDNA2" as previously stated here.
² GCN5 splits: gfx906 (Radeon VII, MI50/MI60) has native `v_dot2_f32_f16` and `v_dot4_i32_i8`,
and both kernels compile for it. Other Vega parts (gfx900/902/90c) have neither.
³ Full-rate FP8 acceleration doesn't land until RDNA4 - RDNA2/RDNA3 (including RDNA3 iGPUs) lack it.

## What's actually implemented today

Forward pass only, both kernels - `src/rdna/fa2_forward_f16.hip` (fp16) and
`src/rdna/fa2_forward_int8qk.hip` (INT8, both GEMMs).

| Feature | FP16 kernel | INT8 QK^T kernel |
|---|---|---|
| MHA / GQA / MQA | ✅ (GQA: `G=2` reuse) | ✅ (GQA by K/V re-read, no reuse yet) |
| Causal masking | ✅ | ✅ |
| Sliding window (one/two-sided) | ✅ | ✅ |
| RoPE (full) | ✅ | ❌ |
| head_dim | 32-512, any multiple of 32¹ | 64, 128 |
| Cross-attention / decode (seq_len != key_seq_len) | ✅ | ✅ |
| Arbitrary strides (non-contiguous batch/head) | ✅ | ✅ |
| Per-token Q/K quantization scales | n/a | ✅ (`qScaleVec`/`kScaleVec`) |
| Per-channel V quantization scale | n/a | ✅ (exact, host-side) |
| Per-token V quantization scale | n/a | ❌ (the scale sits inside the PV sum) |

¹ head_dim 64 and 128 use measured tile sizes; the rest come from `chooseTile()`,
which fits the LDS and register budgets but has not been tuned per size. Prefer a
multiple of **64** where the choice is yours - `head_dim % 64 == 0` lets the
compiler merge GEMM2's LDS reads, worth 36-49% at the sizes where it applies.

The two kernels now cover the same shape surface; what still separates them is
numeric, not structural. Use `flash_attn_int8qk_quantized()` rather than the
raw kernel defaults - it applies per-token Q/K and per-channel V, worth a
measured 3.75x (0.0349 -> 0.0093 worst-case rel_rms on real captures).
INT8's remaining gaps are RoPE and a per-token V scale.

## Not yet implemented

Backward pass (training), dropout, ALiBi/relative position bias, softcap,
varlen (`cu_seqlens`), paged KV cache, BF16/FP8 numerics. None of these are
wired into either kernel - if a future change adds one, update this table
rather than assuming from the algorithm's general capability.
