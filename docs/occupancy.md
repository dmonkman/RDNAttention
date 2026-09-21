# Per-kernel register, LDS and occupancy budget

Every kernel instantiation the library compiles, with the VGPRs, LDS and
occupancy the compiler reports for it on gfx1030, under both toolchains the
project builds with. Companion to [`hardware.md`](hardware.md), which covers the
per-architecture envelope; this covers where each kernel sits inside it.

- **Measured**, not derived: compiler resource remarks, gfx1030 only.
- Kernel sources as of commit `4c12667`.
- **7.2** = AMD HIP SDK 7.2 for Windows (AMD clang 21).
  **10.0** = pip-installed ROCm `rocm[devel]==10.0.0` (AMD clang 23), the
  toolchain CI builds the wheels with.
- Each kernel is instantiated for 4 masking variants (causal x window). VGPRs
  are the maximum over the four, occupancy the minimum. The variants sit within
  9 VGPRs of each other except where marked.

Occupancy is not throughput. The Monarch launcher records Br=64 beating Br=32
despite two fewer waves per SIMD. Treat every row below as a place to look,
and benchmark before changing a tile.

## Occupancy model

The compiler's occupancy is the lower of a VGPR limit and an LDS limit. On
gfx1030, with the kernels' 256-thread workgroups:

- VGPRs are allocated in granules of 16 out of 1024 per SIMD lane.
- LDS is a 128 KiB pool per WGP (64 KiB maximum per workgroup). One 256-thread
  workgroup is 8 wave32 waves across the WGP's 4 SIMDs, so the LDS limit moves
  in steps of 2 waves/SIMD.

This model reproduces the compiler's reported occupancy for all 200 kernels
below, under both toolchains.

| waves/SIMD | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| VGPRs at most | 256 | 192 | 160 | 144 | 128 | 112 | 96 | 80 | - | 64 |
| LDS bytes at most | 65,536 | - | 43,690 | - | 32,768 | - | 26,214 | 21,845 | 18,724 | 16,384 |

"Limiter" in the tables names which side sets the occupancy; "both" means
raising either one alone would not help.

## FP16 kernel (`fa2_forward_f16.hip`)

| head_dim | Tile Br x Bc | Dispatched when | VGPRs 7.2 -> 10.0 | LDS bytes | waves/SIMD 7.2 / 10.0 | Limiter |
|---|---|---|---|---|---|---|
| 32 | 128x64 | always | 117 -> 115 | 28,800 | 8 / 8 | both |
| 64 | 128x64 | MHA | 147 -> 143 | 41,088 | 6 / 6 | LDS (10.0 VGPRs allow 7) |
| 64 | 64x64, G=2 | GQA | 142 -> 138 | 32,896 | 6 / 6 | LDS - **128 B over** the 8-wave line |
| 96 | 128x64 | always | 212 -> 211 | 53,376 | 4 / 4 | both |
| 128 | 128x64 | default | 254 -> 256 | 65,536 | 4 / 4 | both, at both hardware ceilings |
| 128 | 64x64 | causal, grid < 512 blocks | 155 -> 148 | 49,408 | 4 / 4 | LDS (VGPRs allow 6) |
| 160 | 128x32 | always | 248 -> 239 | 61,760 | 4 / 4 | both |
| 192 | 64x32 | always | 196 -> 184 | 49,536 | 4 / 4 | both / LDS |
| 224 | 64x32 | always | 221 -> 206 | 57,792 | 4 / 4 | both |
| 256 | 64x32 | always | 251 -> 235 | 65,536 | 4 / 4 | both |
| 288 | 32x32 | always | 225 -> 204 | 55,872 | 4 / 4 | both |
| 320 | 32x32 | always | 241 -> 221 | 62,080 | 4 / 4 | both |
| 352 | 32x32 | always | 211 -> 231 | 47,872 | 4 / 4 | both |
| 384 | 32x32 | always | 248 -> 246 | 52,032 | 4 / 4 | both |
| 416 | 32x32 | always | 252 -> **256, 33-37 spilled** | 56,192 | 4 / 4 | both |
| 448 | 16x32 | always | 226 -> 254 | 44,992 | 4 / 4 | both |
| 480 | 16x32 | always | 240 -> 237 | 48,128 | 4 / 4 | both |
| 512 | 16x32 | always | 219 -> 220 | 51,264 | 4 / 4 | both |

Tiles for head_dim other than 64 and 128 come from `chooseTile()` against
`kTileVgprCap`; the 64 and 128 tiles are hand-picked in `forwardF16()`.

## INT8 kernel (`fa2_forward_int8qk.hip`)

| head_dim | Tile Br x Bc | Dispatched when | VGPRs 7.2 -> 10.0 | LDS bytes | waves/SIMD 7.2 / 10.0 | Limiter |
|---|---|---|---|---|---|---|
| 64 | 64x64 | seq < 1024 | 88 -> 90 | 16,512 | 10 / 10 | VGPR (LDS allows 14) |
| 64 | 128x64 | seq >= 1024 | 141 -> **164** * | 24,704 | 7 / **5** | VGPR |
| 128 | 64x64 | otherwise | 125 -> 124 | 28,864 | 8 / 8 | both |
| 128 | 128x64 | seq >= 2048, or seq >= 1024 and batch*heads >= 128 | 255 -> 253 | 41,152 | 4 / 4 | VGPR (LDS allows 6) |

\* Only the unmasked variant (no causal, no window) regresses under 10.0; the
other three stay at 137-138 VGPRs and 7 waves. That variant is the default
path for non-causal attention at seq >= 1024.

## Monarch (`monarch_forward_f16.hip`)

Tile chosen from `min(block_b, seq / block_b)`.

| head_dim | Tile Br x Bc | Dispatched when min >= | Stage | VGPRs 7.2 -> 10.0 | LDS bytes | waves/SIMD 7.2 / 10.0 | Limiter |
|---|---|---|---|---|---|---|---|
| 64 | 64x64 | 64 | 1 | 160 -> 146 | 41,216 | 6 / 6 | both |
| 64 | 64x64 | 64 | 2 | 118 -> 114 | 33,280 | 6 / 6 | LDS - **512 B over** the 8-wave line |
| 64 | 48x32 | 48 | 1 | 108 -> 109 | 21,696 | 9 / 9 | VGPR |
| 64 | 48x32 | 48 | 2 | 86 -> 80 | 17,728 | 10 / 12 | VGPR |
| 64 | 32x32 | - | 1 | 96 -> 96 | 18,624 | 10 / 10 | VGPR |
| 64 | 32x32 | - | 2 | 76 -> 72 | 14,656 | 12 / 12 | VGPR |
| 128 | 64x32 | 64 | 1 | 233 -> 216 | 45,376 | 4 / 4 | both |
| 128 | 64x32 | 64 | 2 | 145 -> 132 | 37,312 | 6 / 6 | both / LDS |
| 128 | 48x32 | 48 | 1 | 210 -> 191 | 40,256 | 4 / 5 | VGPR |
| 128 | 48x32 | 48 | 2 | 131 -> 119 | 32,192 | 7 / 8 | VGPR / both |
| 128 | 32x32 | - | 1 | 187 -> 170 | 35,136 | 5 / 5 | VGPR |
| 128 | 32x32 | - | 2 | 117 -> 106 | 27,072 | 8 / 8 | both / LDS |

## Candidates

Open leads, none benchmarked yet.

- **Near misses on LDS.** GQA d=64 is 128 B over the 8-wave line and Monarch
  stage 2 d=64 (64x64) is 512 B over - small enough to be padding. Trimming it
  takes the Monarch kernel from 6 to 8 waves (its VGPRs already fit); GQA
  reaches 7, or 8 if it also sheds 10 VGPRs to 128.
- **LDS-bound with VGPR headroom.** d=128 64x64, the small-grid causal path,
  needs ~5.7 KB less LDS (49,408 -> 43,690) to go from 4 to 6 waves.
- **Both walls at once.** Every fp16 tile at head_dim >= 96 is pinned at 4
  waves by VGPRs and LDS together. Six waves needs <= 160 VGPRs *and*
  <= 43,690 B; cutting either alone buys nothing.
- **ROCm 10.0 regressions.** 448 went 226 -> 254, two VGPRs from the
  ceiling, and measured ~6% slower than under 7.2; 352 went 211 -> 231; INT8
  d=64 128x64 lost two waves as above. Elsewhere 10.0 mostly uses fewer VGPRs, by 10-22 at
  head_dim 160-320.

## The head_dim 416 spill is kept on purpose

Under ROCm 10.0, head_dim 416's 32x32 tile spills 33-37 VGPRs. It sits
exactly on `kTileVgprCap` (52 accumulator + 32 staged-V VGPRs = 84), so a cap
of 83 would move it - and only it - to 16x32 with no spill. Measured instead
(`hip_bench headdim`, b1 h8 seq 2048 non-causal, 6 interleaved rounds, spread
under 1%):

| head_dim 416 | TFLOP/s |
|---|---|
| ROCm 7.2, 32x32, no spill | 6.73 |
| **ROCm 10.0, 32x32, spilling** | **8.35** |
| ROCm 10.0, 16x32 (cap 83), no spill | 6.51 |

The spilling build is the fastest, so the spills evidently sit outside the
hot loop. `tests/head_dim_resources.py` allows up to 40 spilled VGPRs for
head_dim 416 and still fails beyond that. Only one shape was measured;
causal and longer sequences are unchecked.

## Reproduce

Per source file and toolchain; `HIP_PATH` selects the toolchain (for the pip
ROCm, `rocm-sdk path --root` after `rocm-sdk init`):

```
hipcc -x hip --offload-arch=gfx1030 -O3 -ffast-math -std=c++17 -Isrc -Isrc/rdna \
      --cuda-device-only -S -o /dev/null -Rpass-analysis=kernel-resource-usage \
      src/rdna/fa2_forward_f16.hip
```

`python tests/head_dim_resources.py` prints the fp16 part of this and fails on
any spill.
