# Per-architecture hardware resources

Companion to [`FEATURES.md`](../FEATURES.md), which covers *instructions*. This
covers the *resources* those instructions run against - the numbers that decide
tile sizes, occupancy, and whether a kernel is ALU- or bandwidth-bound.

**Read the source column.** Two very different kinds of number appear here:

- **Measured** - probed on this machine against the ROCm 7.2 toolchain and the
  HIP runtime. Reproduce with the commands given under each table.
- **Documented** - AMD whitepapers and ISA guides. Correct as far as it goes,
  but nothing here verified it, and per-SKU cache figures in particular are the
  most likely thing on this page to be wrong. Treat them as orientation, not as
  something to tune against.

Only gfx1030 (RX 6800 XT) has ever been *run*.

## 1. The per-workgroup budget - measured

This is the part that matters most for this kernel, and the headline is that it
barely varies: **the resource envelope a workgroup programs against is
essentially identical from Polaris to RDNA4.**

| Target class | LDS / workgroup | Max VGPRs / wave | Wave width | Max waves / SIMD |
|---|---|---|---|---|
| gfx803 (Polaris) | 64 KiB | 256 | 64 | 10 |
| gfx900-gfx90c (GCN5/Vega) | 64 KiB | 256 | 64 | 10 |
| gfx906 (Vega 20) | 64 KiB | 256 | 64 | 10 |
| gfx908 (CDNA1) | 64 KiB | 255 + 256 AGPR | 64 | 8 |
| gfx90a / gfx942 (CDNA2/3) | 64 KiB | 256 + 256 AGPR | 64 | 8 |
| gfx950 (CDNA4) | **160 KiB** | 256 + 256 AGPR | 64 | 8 |
| gfx1010-gfx1013 (RDNA1) | 64 KiB | 256 | 32 | **20** |
| gfx1030-gfx1036 (RDNA2) | 64 KiB | 256 | 32 | 16 |
| gfx1100+ (RDNA3/3.5/4) | 64 KiB | 256 | 32 | 16 |

Reproduce:

```
# LDS ceiling: ask for 200 KB of __shared__ and read the error.
hipcc -x hip --offload-arch=<arch> --cuda-device-only -S -o /dev/null lds_probe.hip
# VGPR ceiling and wave slots: a register-hungry kernel at __launch_bounds__(64).
hipcc ... -Rpass-analysis=kernel-resource-usage vgpr_probe.hip
```

Two of these carry real consequences:

- **gfx950 is the only target with more than 64 KiB of LDS** (160 KiB). Every
  tile-size decision in this project assumes 64 KiB, and `Br=128, d=128`
  already sits exactly on that ceiling at 65,536 bytes.
- **Wave width is the register story, not the VGPR count.** All targets cap a
  wave at 256 VGPRs, but a GCN/CDNA wave64 VGPR costs 256 bytes against a
  wave32 RDNA VGPR's 128. So "256 VGPRs" buys half as much per-SIMD residency
  on Vega as on RDNA2 - see table 3, where that turns into a 4-6x occupancy
  gap for this kernel.

## 2. Caches - documented, per SKU

Instruction cache, by generation (documented):

| Generation | L1 instruction cache | Shared by |
|---|---|---|
| GCN (gfx803, gfx9) | 32 KiB | up to 4 CUs |
| CDNA (gfx908-gfx950) | 32 KiB | 2 CUs |
| RDNA1 / RDNA2 | 32 KiB | 1 WGP (2 CUs) |
| RDNA3 / RDNA4 | 64 KiB | 1 WGP (2 CUs) |

This one is worth knowing for *this* project specifically: the fp16 kernel
compiles to ~27.7k VALU instructions across 16 template instantiations, and a
single instantiation of the `Br=128, d=128` variant is on the order of several
KB of code. Fully unrolling the GEMM loops - which `#pragma unroll 1` prevents -
took the file to 67.9k instructions, and a 32 KiB I-cache shared across 2-4 CUs
is the reason that is a bad idea rather than a neutral one.

Data caches, per SKU (documented). `(L2+IC)/CU` is the fair cross-generation
column: comparing Infinity Cache alone flatters RDNA2 against parts that simply
put their cache in L2 instead.

| Arch | Retail | CUs | L2 | Infinity Cache | **(L2+IC) / CU** |
|---|---|---|---|---|---|
| gfx803 | RX 480 / 580 | 36 | 2 MB | - | 0.06 MB |
| gfx900 | Vega 56 / 64 | 64 | 4 MB | - | 0.06 MB |
| gfx906 | Radeon VII / MI50 | 60 | 4 MB | - | 0.07 MB |
| gfx90c | Ryzen 4000/5000G iGPU | 8 | 1 MB | - | 0.12 MB |
| gfx1010 | RX 5700 XT | 40 | 4 MB | - | 0.10 MB |
| gfx1012 | RX 5500 XT | 22 | 2 MB | - | 0.09 MB |
| gfx1030 | RX 6900 XT | 80 | 4 MB | 128 MB | 1.65 MB |
| gfx1030 | RX 6800 XT | 72 | 4 MB | 128 MB | 1.83 MB |
| gfx1030 | RX 6800 | 60 | 4 MB | 128 MB | 2.20 MB |
| gfx1031 | RX 6700 XT | 40 | 3 MB | 96 MB | **2.48 MB** |
| gfx1032 | RX 6600 XT | 32 | 2 MB | 32 MB | 1.06 MB |
| gfx1034 | RX 6500 XT | 16 | 1 MB | 16 MB | 1.06 MB |
| gfx1035 | Radeon 680M | 12 | 2 MB | - | 0.17 MB |
| gfx1100 | RX 7900 XTX | 96 | 6 MB | 96 MB | 1.06 MB |
| gfx1101 | RX 7800 XT | 60 | 4 MB | 64 MB | 1.13 MB |
| gfx1102 | RX 7600 | 32 | 2 MB | 32 MB | 1.06 MB |
| gfx1201 | RX 9070 XT | 64 | 8 MB | 64 MB | 1.12 MB |

Normalised, the picture inverts the marketing one. Infinity Cache capacity per
CU is **flat at ~1.0-1.1 MB** across RDNA2's smaller dies, RDNA3 and RDNA4 -
the big numbers scale with CU count, not with what each CU gets. The genuine
outliers are Navi 21 and Navi 22, at 1.6-2.5 MB/CU, and those are the parts this
project actually targets. Everything without Infinity Cache sits an order of
magnitude lower, at 0.06-0.17 MB/CU.

For attention that number has a concrete meaning, because one head's K+V at fp16
is `2 * seq * head_dim * 2` bytes:

| | d=64 | d=128 |
|---|---|---|
| seq 512 | 0.12 MB | 0.25 MB |
| seq 2048 | 0.50 MB | **1.00 MB** |
| seq 4096 | 1.00 MB | 2.00 MB |

So at seq=2048, d=128 - the benchmark's headline shape - one head's K+V is
almost exactly one CU's share of Infinity Cache on a 6800 XT, and roughly ten
times a Vega 64 CU's share of L2. A flash-attention kernel re-reads K and V once
per query row-block, so that ratio is what decides whether those re-reads are
served on-die or from VRAM. It is also a caution against reading the Tier 2/3
benchmark columns as hardware predictions: those hold the memory system fixed at
RDNA2's and vary only the dot instruction.

## 3. What this kernel actually gets - measured

`fa2_forward_f16.hip` compiled per target, reporting the three shipping
instantiations. Occupancy is the compiler's combined VGPR *and* LDS limit.

| Target | d=64, Br=128 (41 KB LDS) | d=128, Br=128 (64 KB LDS) | d=128, Br=64 (48 KB LDS) |
|---|---|---|---|
| gfx900 (Vega) | 169 VGPR, **1 wave/SIMD** | 247 VGPR, **1** | 156 VGPR, **1** |
| gfx906 | 169 VGPR, **1** | 239 VGPR, **1** | 157 VGPR, **1** |
| gfx1010 (RDNA1) | 167 VGPR, 6 | 254 VGPR, 4 | 154 VGPR, 4 |
| gfx1030 (RDNA2) | 139 VGPR, 6 | 254 VGPR, 4 | 155 VGPR, 4 |
| gfx1100 (RDNA3) | 167 VGPR, 6 | 254 VGPR, 4 | 155 VGPR, 4 |
| gfx1200 (RDNA4) | 166 VGPR, 6 | 253 VGPR, 4 | 154 VGPR, 4 |

```
hipcc -x hip --offload-arch=<arch> -O3 -ffast-math -std=c++17 -Isrc -Isrc/rdna \
      --cuda-device-only -S -o /dev/null -Rpass-analysis=kernel-resource-usage \
      src/rdna/fa2_forward_f16.hip
```

**Every GCN5 instantiation lands at 1 wave per SIMD.** That is not the dot
instruction and it is not LDS - it is wave64 doubling the byte cost of the same
256 VGPRs. With one wave resident there is nothing to hide memory latency behind,
which is a second, independent Tier 2 penalty stacked on top of the +15.6% VALU
that `FEATURES.md` measures.

It also argues for retuning `Br`/`Bc` per tier: the current tile table was chosen
against a register file that a Vega does not have in wave32 terms, so a smaller
`Br` is likely to be right there for reasons that have nothing to do with
`v_dot2`. That remains a hypothesis - it needs a Tier 2 part to confirm, and the forced-lowering benchmark
columns cannot answer it, since they hold RDNA2's wave32 and memory system fixed.


## 4. The v_dot family - rate and usefulness, measured

All seven `v_dot` variants exist on gfx1030 and all are **full rate, one
instruction per lane per clock**. Measured with 16 independent dependency
chains so the pipeline stays full (`scratchpad/dotrate.hip`; the ISA is
checked to contain exactly 16 dots per loop before any timing is trusted):

| builtin | instruction | Ginstr/s | Gelem/s | vs fdot2 elems |
|---|---|---|---|---|
| `fdot2` | `v_dot2c_f32_f16` | 10348 | 20697 | 1.00x |
| `sdot2` | `v_dot2_i32_i16` | 10319 | 20637 | 1.00x |
| `udot2` | `v_dot2_u32_u16` | 10391 | 20781 | 1.00x |
| `sdot4` | `v_dot4c_i32_i8` | 10247 | 40987 | 1.98x |
| `udot4` | `v_dot4_u32_u8` | 10358 | 41433 | 2.00x |
| `sdot8` | `v_dot8_i32_i4` | 10425 | 83397 | 4.03x |
| `udot8` | `v_dot8_u32_u4` | 10129 | 81035 | 3.92x |
| - | `v_fma_f32` (reference) | 10380 | 10380 | 0.50x |

The instruction rate is flat at ~10,350 Ginstr/s across every variant, against
a ceiling of 72 CU x 64 lanes x ~2.25 GHz = 10,368. So **element throughput is
decided entirely by lanes per instruction**, and signed vs unsigned is free -
the choice between them is always numeric, never performance.

Two traps in measuring this, both of which produced physically impossible
numbers (10^7 Ginstr/s) before being fixed. Loop-invariant operands let LLVM
fold the whole loop into one multiply, so each dot must take its own
accumulator as an operand. And a store guarded by `threadIdx.x == 1024` is
provably false under `__launch_bounds__(256)`, which makes the store, and then
the entire loop, dead. Always check the emitted ISA against a physical
ceiling before believing a microbenchmark.

### What that means for this project

- **i16/u16 (`sdot2`/`udot2`): no reason to implement.** Identical rate to
  fp16 and identical 2 bytes per element, so there is no speed or memory win
  at all - only quantization plumbing. fp16's exponent suits attention scores
  better than a 16-bit integer, and the accumulator is fp32 either way.
- **u8/u4 vs i8/i4: same rate, so unsigned is only ever a numeric choice.**
  It has exactly one natural home here: `P`, the softmax output, is
  non-negative, so signed int8 wastes its sign bit. Unsigned would give 255
  levels instead of 127 - one free bit. Mixing it with signed `V` needs `V`
  offset by +128 and a `128 * sum(p)` correction; that sum is one extra
  `udot4` against `0x01010101` per (query, kv-quad), about `1/DPT` more GEMM2
  dots. Worth doing only once `P` is the accuracy limit, which it is not today.
- **i4/u4 (`sdot8`/`udot8`): 4x fp16's element rate, and the accuracy does
  not currently support it** - see below.
