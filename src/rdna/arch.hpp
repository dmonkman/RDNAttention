// Target-architecture feature detection and dot-product primitives.
#pragma once

#include <hip/hip_runtime.h>

// No v_dot2_f32_f16 / v_dot4_i32_i8.
#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx902__) || \
    defined(__gfx904__) || defined(__gfx909__) || defined(__gfx90c__) || \
    defined(__gfx1010__) || defined(__gfx1013__)
#  define RDNA_NO_DOT 1
#endif

// Mixed-precision f16->f32 FMA availability on the no-dot targets.
#if defined(__gfx904__) || defined(__gfx1010__) || defined(__gfx1013__)
#  define RDNA_HAS_FMA_MIX 1
#elif defined(__gfx900__) || defined(__gfx902__) || defined(__gfx909__) || \
    defined(__gfx90c__)
#  define RDNA_HAS_MAD_MIX_ONLY 1
#endif

// fdot2() lowering. -DRDNA_FORCE_DOT_PATH=<n> overrides the target pick.
#define RDNA_DOT_PATH_BUILTIN 1  // v_dot2_f32_f16
#define RDNA_DOT_PATH_PORTABLE 2 // two fmaf
#define RDNA_DOT_PATH_MIXASM 3   // hand-written mixed-FMA asm
#define RDNA_DOT_PATH_CVTFMA 4   // cvt+fma, force-only

#if defined(RDNA_FORCE_DOT_PATH)
#  define RDNA_DOT_PATH RDNA_FORCE_DOT_PATH
#elif defined(RDNA_HAS_MAD_MIX_ONLY)
#  define RDNA_DOT_PATH RDNA_DOT_PATH_MIXASM
#elif defined(RDNA_NO_DOT)
#  define RDNA_DOT_PATH RDNA_DOT_PATH_PORTABLE
#else
#  define RDNA_DOT_PATH RDNA_DOT_PATH_BUILTIN
#endif

#if defined(RDNA_HAS_MAD_MIX_ONLY)
#  define RDNA_MIX_OP "v_mad_mix_f32"
#else
#  define RDNA_MIX_OP "v_fma_mix_f32"
#endif

// Unpack a packed f16 pair to two f32 (CVTFMA path only).
#define RDNA_CVT2(lo, hi, src)             \
    asm("v_cvt_f32_f16_e32 %0, %2\n\t"     \
        "v_lshrrev_b32_e32 %1, 16, %2\n\t" \
        "v_cvt_f32_f16_e32 %1, %1"         \
        : "=&v"(lo), "=&v"(hi) : "v"(src))
// "=&v": early-clobber, or the allocator may alias an output onto a source.

namespace rdna {

typedef _Float16 half2x_t __attribute__((ext_vector_type(2)));

/// fp32 += a.x*b.x + a.y*b.y
__device__ __forceinline__ float fdot2(half2x_t a, half2x_t b, float acc) {
#if RDNA_DOT_PATH == RDNA_DOT_PATH_MIXASM
    float lo, hi;
    asm(RDNA_MIX_OP " %0, %1, %2, %3 op_sel_hi:[1,1,0]"
        : "=v"(lo) : "v"(a), "v"(b), "v"(acc));
    asm(RDNA_MIX_OP " %0, %1, %2, %3 op_sel:[1,1,0] op_sel_hi:[1,1,0]"
        : "=v"(hi) : "v"(a), "v"(b), "v"(lo));
    return hi;
#elif RDNA_DOT_PATH == RDNA_DOT_PATH_CVTFMA
    float a0, a1, b0, b1;
    RDNA_CVT2(a0, a1, a);
    RDNA_CVT2(b0, b1, b);
    return fmaf(a1, b1, fmaf(a0, b0, acc));
#elif RDNA_DOT_PATH == RDNA_DOT_PATH_PORTABLE
    acc = fmaf(static_cast<float>(a[0]), static_cast<float>(b[0]), acc);
    return fmaf(static_cast<float>(a[1]), static_cast<float>(b[1]), acc);
#else
    return __builtin_amdgcn_fdot2(a, b, acc, false);
#endif
}

/// int32 += 4 packed signed int8 lanes dotted
__device__ __forceinline__ int sdot4(int a, int b, int acc) {
    return __builtin_amdgcn_sdot4(a, b, acc, false);
}

} // namespace rdna
