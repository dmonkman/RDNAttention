// Forward attention kernel (fp16), FlashAttention-2 style.
#include "fa2_forward_f16.hpp"

#include "arch.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

namespace rdna {
namespace {

using half2_t = _Float16 __attribute__((ext_vector_type(2)));

using F16x8 = _Float16 __attribute__((ext_vector_type(8)));
static_assert(alignof(F16x8) == 16, "F16x8 must be naturally 128-bit-aligned for the reinterpret_cast loads below");

template <int I, int J>
__device__ __forceinline__ half2_t halfPair(F16x8 v) {
    return __builtin_shufflevector(v, v, I, J);
}
template <int I>
__device__ __forceinline__ half2_t halfPairAcross(F16x8 a, F16x8 b) {
    return __builtin_shufflevector(a, b, I, 8 + I);
}

// 16-lane DPP butterfly reduction.
constexpr int kDppSwap1 = 0xB1;
constexpr int kDppSwap2 = 0x4E;
constexpr int kDppSwap4 = 0x141;
constexpr int kDppSwap8 = 0x140;

template <int Ctrl>
__device__ __forceinline__ float dppMov(float x) {
    union {
        float f;
        int i;
    } a, b;
    a.f = x;
    b.i = __builtin_amdgcn_update_dpp(0, a.i, Ctrl, 0xF, 0xF, false);
    return b.f;
}
__device__ __forceinline__ float red16Max(float x) {
    x = fmaxf(x, dppMov<kDppSwap1>(x));
    x = fmaxf(x, dppMov<kDppSwap2>(x));
    x = fmaxf(x, dppMov<kDppSwap4>(x));
    x = fmaxf(x, dppMov<kDppSwap8>(x));
    return x;
}
__device__ __forceinline__ float red16Sum(float x) {
    x += dppMov<kDppSwap1>(x);
    x += dppMov<kDppSwap2>(x);
    x += dppMov<kDppSwap4>(x);
    x += dppMov<kDppSwap8>(x);
    return x;
}

__device__ __forceinline__ int iabs(int x) {
    return x < 0 ? -x : x;
}

/// RoPE for one 8-element group, in place.
__device__ __forceinline__ void applyRope(F16x8& x8, const float* rotCos, const float* rotSin, int rot0) {
    float c[4], sn[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i] = rotCos[rot0 + i];
        sn[i] = rotSin[rot0 + i];
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float a = (float)x8[2 * i], b = (float)x8[2 * i + 1];
        x8[2 * i] = (_Float16)(a * c[i] - b * sn[i]);
        x8[2 * i + 1] = (_Float16)(a * sn[i] + b * c[i]);
    }
}

constexpr float kLog2E = 1.442695041f;
constexpr float kMaskedSentinel = -1e30f;

// Measured BR=128 crossover, head_dim=128 causal.
constexpr uint64_t kD128CausalBr128MinGrid = 512;

struct KernelParams {
    const _Float16* q;
    const _Float16* k;
    const _Float16* v;
    _Float16* o;
    const float* rotCos;
    const float* rotSin;
    int64_t qStrideB, qStrideH;
    int32_t qStrideS;
    int64_t kStrideB, kStrideH;
    int32_t kStrideS;
    int64_t vStrideB, vStrideH;
    int32_t vStrideS;
    int64_t oStrideB, oStrideH;
    int32_t oStrideS;
    uint32_t numHeads, numKvHeads, seqLen, keySeqLen;
    float scale;
    uint32_t hasRope;
    int32_t windowSize; // <= 0: no window
    uint32_t nullKeys;
};

/// Reciprocal of the softmax denominator, scaling acco on the way out.
/// Dropped null keys scored exactly 0 and carried no value, so they add
/// nullKeys * exp2(-m) to the denominator and nothing to acco. Renormalising
/// to max(rm, 0) keeps both terms finite when every real logit is negative,
/// and returns 0 rather than NaN when rm is still the sentinel.
__device__ inline float softmaxInv(float rl, float rm, uint32_t nullKeys) {
    if (nullKeys == 0) {
        return rl > 0.0f ? 1.0f / rl : 0.0f;
    }
    const float m = fmaxf(rm, 0.0f);
    const float corr = exp2f(rm - m);
    const float denom = rl * corr + (float)nullKeys * exp2f(-m);
    return denom > 0.0f ? corr / denom : 0.0f;
}

constexpr int kLdsBudget = 65536;

constexpr int cmax(int a, int b) {
    return a > b ? a : b;
}

constexpr int ldsPlainBytes(int H, int Br, int Bc, bool swz, int G) {
    return 4 * (G * (H / 2) * Br) + 4 * cmax((H / 2) * (swz ? Bc : Bc + 1), (Bc / 2) * (swz ? Br : Br + 1)) + 4 * ((Bc / 2) * H);
}
constexpr int ldsAliasBytes(int H, int Br, int Bc, bool swz, int G) {
    return 4 * (G * (H / 2) * Br) + 4 * cmax((H / 2) * (swz ? Bc : Bc + 1), (Bc / 2) * H) + 4 * ((Bc / 2) * (swz ? Br : Br + 1));
}

/// Smallest-intrusion LDS layout that fits; -1 if none does.
constexpr int pickLayout(int H, int Br, int Bc, int G) {
    return ldsPlainBytes(H, Br, Bc, false, G) <= kLdsBudget   ? 0
           : ldsPlainBytes(H, Br, Bc, true, G) <= kLdsBudget  ? 1
           : ldsAliasBytes(H, Br, Bc, false, G) <= kLdsBudget ? 2
           : ldsAliasBytes(H, Br, Bc, true, G) <= kLdsBudget  ? 3
                                                              : -1;
}

// Tile geometry, 256 threads as 16x16 tx/ty.
template <int HeadDim, int Br, int Bc, int G = 1>
struct Tile {
    static constexpr int HH = HeadDim / 2;
    static constexpr int BCH = Bc / 2;
    static constexpr int QPT = Br / 16;
    static constexpr int KPT = Bc / 16;
    static constexpr int DPT = HeadDim / 16;
    static constexpr int HeadDimVec8 = HeadDim / 8;
    static constexpr int HalfDim = HeadDim / 2;

    static_assert(HeadDim % 32 == 0, "head_dim must be a multiple of 32");
    static_assert(Bc >= 32 && Bc % 32 == 0, "Bc: the key mapping covers 32 columns per index pair");
    static_assert(Br >= 16 && Br % 16 == 0, "Br: ty covers 16 query rows");
    static_assert(HH % 4 == 0, "GEMM1 steps dh by 4");
    static_assert(BCH % 4 == 0, "GEMM2 steps kh by 4");

    static constexpr int Layout = pickLayout(HeadDim, Br, Bc, G);
    static_assert(Layout >= 0, "no LDS layout fits this (head_dim, Br, Bc) - shrink Br or Bc");
    static constexpr bool Swizzle = (Layout == 1 || Layout == 3);
    static constexpr bool Alias = (Layout >= 2);

    static constexpr int KStride = Swizzle ? Bc : Bc + 1;
    static constexpr int PStride = Swizzle ? Br : Br + 1;
    static constexpr int KElems = HH * KStride;
    static constexpr int PElems = BCH * PStride;
    static constexpr int VElems = BCH * HeadDim;
    static constexpr int BufAElems = Alias ? cmax(KElems, VElems) : cmax(KElems, PElems);
    static constexpr int BufBElems = Alias ? PElems : VElems;

    static_assert(!Swizzle || (Bc >= 32 && (Bc & (Bc - 1)) == 0), "kSwz would leave [0, Bc)");
    static_assert(!Swizzle || (Br >= BCH && (Br & (Br - 1)) == 0), "pSwz would leave [0, Br)");

    __device__ static __forceinline__ int kSwz(int dh, int kc) {
        return Swizzle ? (kc ^ (dh & 31)) : kc;
    }
    __device__ static __forceinline__ int pSwz(int kh, int row) {
        return Swizzle ? (row ^ (kh & 31)) : row;
    }
    __device__ static __forceinline__ int vSwz(int d) {
        return d ^ ((d >> 4) & 7);
    }
};

constexpr int VRegsIters(int H, int Bc) {
    return ((Bc / 2) * (H / 8) + 255) / 256;
}

template <int HeadDim, int Bc>
struct VRegs {
    static constexpr int Total = (Bc / 2) * (HeadDim / 8);
    static constexpr int Iters = VRegsIters(HeadDim, Bc);
    F16x8 v0[Iters];
    F16x8 v1[Iters];
};

/// One (kv row-pair, head_dim group) slot's V columns into LDS.
template <int HeadDim, int Br, int Bc, int G>
__device__ __forceinline__ void storeVPair(half2_t sV[], int kh, int dg, F16x8 v0, F16x8 v1) {
    using T = Tile<HeadDim, Br, Bc, G>;
    int d0 = dg * 8;
    sV[kh * HeadDim + T::vSwz(d0 + 0)] = halfPairAcross<0>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 1)] = halfPairAcross<1>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 2)] = halfPairAcross<2>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 3)] = halfPairAcross<3>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 4)] = halfPairAcross<4>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 5)] = halfPairAcross<5>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 6)] = halfPairAcross<6>(v0, v1);
    sV[kh * HeadDim + T::vSwz(d0 + 7)] = halfPairAcross<7>(v0, v1);
}

/// Drains staged V registers into LDS (alias layout only).
template <int HeadDim, int Br, int Bc, int G>
__device__ __forceinline__ void storeVRegs(half2_t sV[], const VRegs<HeadDim, Bc>& vr) {
    using VR = VRegs<HeadDim, Bc>;
    constexpr int nThreads = 256;
    constexpr int hdVec8 = HeadDim / 8;
    const int t = threadIdx.y * 16 + threadIdx.x;
    #pragma unroll
    for (int it = 0; it < VR::Iters; it++) {
        int e = t + it * nThreads;
        if (e >= VR::Total) {
            continue;
        }
        storeVPair<HeadDim, Br, Bc, G>(sV, e / hdVec8, e % hdVec8, vr.v0[it], vr.v1[it]);
    }
}

/// Loads one K/V block; V goes to registers under the alias layout.
template <int HeadDim, int Br, int Bc, int G>
__device__ __forceinline__ void loadKvBlock(
    const KernelParams& p, int kb, int64_t kRow0, int64_t vRow0, half2_t sA[], half2_t sB[],
    VRegs<HeadDim, Bc>& vr) {
    using T = Tile<HeadDim, Br, Bc, G>;
    const int t = threadIdx.y * 16 + threadIdx.x;
    constexpr int nThreads = 256;

    constexpr int totalK = Bc * T::HeadDimVec8;
    constexpr int itersK = (totalK + nThreads - 1) / nThreads;
    F16x8 k8s[itersK];
    bool kValid[itersK];
    #pragma unroll
    for (int it = 0; it < itersK; it++) {
        int e = t + it * nThreads;
        int dg = e % T::HeadDimVec8;
        int kc = e / T::HeadDimVec8;
        int sk = kb * Bc + kc;
        bool ok = (e < totalK) && (sk < (int)p.keySeqLen);
        kValid[it] = ok;
        if (ok) {
            const _Float16* kRow = p.k + kRow0 + sk * p.kStrideS;
            k8s[it] = *reinterpret_cast<const F16x8*>(kRow + dg * 8);
        } else {
            k8s[it] = (F16x8)(_Float16)0.0f;
        }
    }
    #pragma unroll
    for (int it = 0; it < itersK; it++) {
        int e = t + it * nThreads;
        if (e >= totalK) {
            continue;
        }
        int dg = e % T::HeadDimVec8;
        int kc = e / T::HeadDimVec8;
        int sk = kb * Bc + kc;

        F16x8 k8 = k8s[it];
        if (kValid[it] && p.hasRope) {
            applyRope(k8, p.rotCos, p.rotSin, sk * T::HalfDim + dg * 4);
        }
        int dh0 = dg * 4;
        sA[(dh0 + 0) * T::KStride + T::kSwz(dh0 + 0, kc)] = halfPair<0, 1>(k8);
        sA[(dh0 + 1) * T::KStride + T::kSwz(dh0 + 1, kc)] = halfPair<2, 3>(k8);
        sA[(dh0 + 2) * T::KStride + T::kSwz(dh0 + 2, kc)] = halfPair<4, 5>(k8);
        sA[(dh0 + 3) * T::KStride + T::kSwz(dh0 + 3, kc)] = halfPair<6, 7>(k8);
    }

    constexpr int totalV = T::BCH * T::HeadDimVec8;
    constexpr int itersV = (totalV + nThreads - 1) / nThreads;
    static_assert(itersV == VRegs<HeadDim, Bc>::Iters, "VRegs must be sized for this loop");
    #pragma unroll
    for (int it = 0; it < itersV; it++) {
        int e = t + it * nThreads;
        if (e >= totalV) {
            continue;
        }
        int dg = e % T::HeadDimVec8;
        int kh = e / T::HeadDimVec8;
        int sk0 = kb * Bc + 2 * kh;
        int sk1 = sk0 + 1;

        F16x8 v0 = (F16x8)(_Float16)0.0f, v1 = (F16x8)(_Float16)0.0f;
        if (sk0 < (int)p.keySeqLen) {
            const _Float16* vRow = p.v + vRow0 + sk0 * p.vStrideS;
            v0 = *reinterpret_cast<const F16x8*>(vRow + dg * 8);
        }
        if (sk1 < (int)p.keySeqLen) {
            const _Float16* vRow = p.v + vRow0 + sk1 * p.vStrideS;
            v1 = *reinterpret_cast<const F16x8*>(vRow + dg * 8);
        }
        if constexpr (T::Alias) {
            vr.v0[it] = v0;
            vr.v1[it] = v1;
        } else {
            storeVPair<HeadDim, Br, Bc, G>(sB, kh, dg, v0, v1);
        }
    }
}

/// [first, last) kv_block range this row-tile can touch.
template <int Br, int Bc, bool Causal, bool Window>
__device__ __forceinline__ void kvBlockRange(const KernelParams& p, int blockRow, int delta, int halfWindow,
                                             int& firstKvBlock, int& lastKvBlock) {
    int numKvBlocks = ((int)p.keySeqLen + Bc - 1) / Bc;
    firstKvBlock = 0;
    lastKvBlock = numKvBlocks;
    if (Window || Causal) {
        int kmax = (int)p.keySeqLen - 1;
        int k0 = blockRow * Br + delta;
        int k1 = min(blockRow * Br + Br, (int)p.seqLen) - 1 + delta;
        int firstValidK = 0;
        int lastValidK = Causal ? k1 : kmax;
        if (Window) {
            if (Causal) {
                firstValidK = k0 - p.windowSize + 1;
            } else {
                firstValidK = k0 - halfWindow;
                lastValidK = k1 + halfWindow;
            }
        }
        firstValidK = min(max(firstValidK, 0), kmax);
        lastValidK = min(max(lastValidK, 0), kmax);
        firstKvBlock = firstValidK / Bc;
        lastKvBlock = min(lastValidK / Bc + 1, numKvBlocks);
    }
}

template <bool Causal, bool Window>
__device__ __forceinline__ bool isMasked(const KernelParams& p, int key, int qq, int delta, int halfWindow,
                                         bool diag, bool rowActive) {
    bool masked = !rowActive;
    if (Causal) {
        if (diag && key > qq + delta) {
            masked = true;
        }
    } else if ((uint32_t)key >= p.keySeqLen) {
        masked = true;
    }
    if (Window) {
        if (Causal) {
            if (qq + delta - key >= p.windowSize) {
                masked = true;
            }
        } else {
            if (iabs(qq + delta - key) > halfWindow) {
                masked = true;
            }
        }
    }
    return masked;
}

/// GEMM1 -> softmax -> GEMM2 for one kv_block.
template <int HeadDim, int Br, int Bc, bool Causal, bool Window>
__device__ __forceinline__ void processKvBlock(
    const KernelParams& p, int kb, int tx, int ty, int blockRow, int delta, float scaleLog2e,
    int halfWindow, half2_t sQ[][Br], half2_t sA[], half2_t sB[], const VRegs<HeadDim, Bc>& vr,
    const int gQq[], const bool isActive[], float acco[][Tile<HeadDim, Br, Bc>::DPT],
    float rm[], float rl[]) {
    using T = Tile<HeadDim, Br, Bc>;
    half2_t* sK = sA;
    half2_t* sP = T::Alias ? sB : sA;
    half2_t* sV = T::Alias ? sA : sB;
    constexpr int QPT = T::QPT;
    constexpr int KPT = T::KPT;
    constexpr int DPT = T::DPT;
    constexpr int HH = T::HH;
    constexpr int BCH = T::BCH;

    /// Step 1: GEMM1 (QK^T)
    float accs[QPT][KPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            accs[i][j] = 0.0f;
        }
    }

    #pragma unroll 1
    for (int dh0 = 0; dh0 < HH; dh0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int dh = dh0 + s;
            half2_t kv[KPT];
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                int kc = (j >> 1) * 32 + 2 * tx + (j & 1);
                kv[j] = sK[dh * T::KStride + T::kSwz(dh, kc)];
            }
            half2_t qv[QPT];
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                qv[i] = sQ[dh][ty * QPT + i];
            }
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                #pragma unroll
                for (int j = 0; j < KPT; j++) {
                    accs[i][j] = fdot2(qv[i], kv[j], accs[i][j]);
                }
            }
        }
    }
    __syncthreads();

    if constexpr (T::Alias) {
        storeVRegs<HeadDim, Br, Bc, 1>(sV, vr);
    }

    /// Step 2: softmax
    bool diag = Causal && (int)(kb * Bc + Bc - 1) > blockRow * Br + delta;
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        int qq = gQq[i];
        bool rowActive = isActive[i];
        float sj[KPT];
        float lmax = kMaskedSentinel;
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            int key = kb * Bc + (j >> 1) * 32 + 2 * tx + (j & 1);
            bool masked = isMasked<Causal, Window>(p, key, qq, delta, halfWindow, diag, rowActive);
            float sv = masked ? kMaskedSentinel : (accs[i][j] * scaleLog2e);
            sj[j] = sv;
            lmax = fmaxf(lmax, sv);
        }
        float pr[KPT];
        float mnew = kMaskedSentinel;
        if (rowActive) {
            lmax = red16Max(lmax);
            mnew = fmaxf(rm[i], lmax);
        }
        if (mnew > kMaskedSentinel) {
            if (mnew > rm[i]) {
                float corr = exp2f(rm[i] - mnew);
                rl[i] *= corr;
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    acco[i][e] *= corr;
                }
            }
            float lsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                float pj = exp2f(sj[j] - mnew);
                pr[j] = pj;
                lsum += pj;
            }
            rl[i] += red16Sum(lsum);
            rm[i] = mnew;
        } else {
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                pr[j] = 0.0f;
            }
        }
        #pragma unroll
        for (int j = 0; j < KPT; j += 2) {
            int kh = (j >> 1) * 16 + tx;
            sP[kh * T::PStride + T::pSwz(kh, ty * QPT + i)] = half2_t{(_Float16)pr[j], (_Float16)pr[j + 1]};
        }
    }
    __syncthreads();

    /// Step 3: GEMM2 (PV)
    #pragma unroll 1
    for (int kh0 = 0; kh0 < BCH; kh0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int kh = kh0 + s;
            half2_t vv[DPT];
            #pragma unroll
            for (int e = 0; e < DPT; e++) {
                vv[e] = sV[kh * HeadDim + T::vSwz(e * 16 + tx)];
            }
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                half2_t pv = sP[kh * T::PStride + T::pSwz(kh, ty * QPT + i)];
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    acco[i][e] = fdot2(pv, vv[e], acco[i][e]);
                }
            }
        }
    }
}

/// GQA variant: G query heads sharing a kv_head process one loaded K/V tile.
template <int HeadDim, int Br, int Bc, int G, bool Causal, bool Window>
__device__ __forceinline__ void processKvBlockGqa(
    const KernelParams& p, int kb, int tx, int ty, int blockRow, int delta, float scaleLog2e, int halfWindow,
    half2_t sQ[][Tile<HeadDim, Br, Bc, G>::HH][Br], half2_t sA[], half2_t sB[], const VRegs<HeadDim, Bc>& vr,
    const int gQq[], const bool isActive[],
    float acco[][Tile<HeadDim, Br, Bc, G>::QPT][Tile<HeadDim, Br, Bc, G>::DPT],
    float rm[][Tile<HeadDim, Br, Bc, G>::QPT], float rl[][Tile<HeadDim, Br, Bc, G>::QPT]) {
    using T = Tile<HeadDim, Br, Bc, G>;
    half2_t* sK = sA;
    half2_t* sP = T::Alias ? sB : sA;
    half2_t* sV = T::Alias ? sA : sB;
    constexpr int QPT = T::QPT;
    constexpr int KPT = T::KPT;
    constexpr int DPT = T::DPT;
    constexpr int HH = T::HH;
    constexpr int BCH = T::BCH;

    /// Step 1: GEMM1, all G heads
    float accs[G][QPT][KPT];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                accs[g][i][j] = 0.0f;
            }
        }
    }

    #pragma unroll 1
    for (int dh0 = 0; dh0 < HH; dh0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int dh = dh0 + s;
            half2_t kv[KPT];
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                int kc = (j >> 1) * 32 + 2 * tx + (j & 1);
                kv[j] = sK[dh * T::KStride + T::kSwz(dh, kc)];
            }
            #pragma unroll
            for (int g = 0; g < G; g++) {
                half2_t qv[QPT];
                #pragma unroll
                for (int i = 0; i < QPT; i++) {
                    qv[i] = sQ[g][dh][ty * QPT + i];
                }
                #pragma unroll
                for (int i = 0; i < QPT; i++) {
                    #pragma unroll
                    for (int j = 0; j < KPT; j++) {
                        accs[g][i][j] = fdot2(qv[i], kv[j], accs[g][i][j]);
                    }
                }
            }
        }
    }
    __syncthreads();

    if constexpr (T::Alias) {
        storeVRegs<HeadDim, Br, Bc, G>(sV, vr);
    }

    bool diag = Causal && (int)(kb * Bc + Bc - 1) > blockRow * Br + delta;

    /// Step 2/3: per-head softmax + GEMM2
    #pragma unroll
    for (int g = 0; g < G; g++) {
        if (g > 0) {
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            int qq = gQq[i];
            bool rowActive = isActive[i];
            float sj[KPT];
            float lmax = kMaskedSentinel;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                int key = kb * Bc + (j >> 1) * 32 + 2 * tx + (j & 1);
                bool masked = isMasked<Causal, Window>(p, key, qq, delta, halfWindow, diag, rowActive);
                float sv = masked ? kMaskedSentinel : (accs[g][i][j] * scaleLog2e);
                sj[j] = sv;
                lmax = fmaxf(lmax, sv);
            }
            float pr[KPT];
            float mnew = kMaskedSentinel;
            if (rowActive) {
                lmax = red16Max(lmax);
                mnew = fmaxf(rm[g][i], lmax);
            }
            if (mnew > kMaskedSentinel) {
                if (mnew > rm[g][i]) {
                    float corr = exp2f(rm[g][i] - mnew);
                    rl[g][i] *= corr;
                    #pragma unroll
                    for (int e = 0; e < DPT; e++) {
                        acco[g][i][e] *= corr;
                    }
                }
                float lsum = 0.0f;
                #pragma unroll
                for (int j = 0; j < KPT; j++) {
                    float pj = exp2f(sj[j] - mnew);
                    pr[j] = pj;
                    lsum += pj;
                }
                rl[g][i] += red16Sum(lsum);
                rm[g][i] = mnew;
            } else {
                #pragma unroll
                for (int j = 0; j < KPT; j++) {
                    pr[j] = 0.0f;
                }
            }
            #pragma unroll
            for (int j = 0; j < KPT; j += 2) {
                int kh = (j >> 1) * 16 + tx;
                sP[kh * T::PStride + T::pSwz(kh, ty * QPT + i)] = half2_t{(_Float16)pr[j], (_Float16)pr[j + 1]};
            }
        }
        __syncthreads();

        #pragma unroll 1
        for (int kh0 = 0; kh0 < BCH; kh0 += 4) {
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                int kh = kh0 + s;
                half2_t vv[DPT];
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    vv[e] = sV[kh * HeadDim + T::vSwz(e * 16 + tx)];
                }
                #pragma unroll
                for (int i = 0; i < QPT; i++) {
                    half2_t pv = sP[kh * T::PStride + T::pSwz(kh, ty * QPT + i)];
                    #pragma unroll
                    for (int e = 0; e < DPT; e++) {
                        acco[g][i][e] = fdot2(pv, vv[e], acco[g][i][e]);
                    }
                }
            }
        }
    }
}

template <int HeadDim, int Br, int Bc, bool Causal, bool Window>
__global__ __launch_bounds__(256) void forwardKernel(KernelParams p) {
    using T = Tile<HeadDim, Br, Bc>;
    constexpr int QPT = T::QPT;
    constexpr int DPT = T::DPT;
    constexpr int HH = T::HH;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int batchHeadIdx = blockIdx.y;
    const int batchIdx = batchHeadIdx / (int)p.numHeads;
    const int headIdx = batchHeadIdx % (int)p.numHeads;
    const int ratio = (int)p.numHeads / (int)p.numKvHeads;
    const int kvHeadIdx = headIdx / ratio;

    const int64_t qRow0 = (int64_t)batchIdx * p.qStrideB + (int64_t)headIdx * p.qStrideH;
    const int64_t oRow0 = (int64_t)batchIdx * p.oStrideB + (int64_t)headIdx * p.oStrideH;
    const int64_t kRow0 = (int64_t)batchIdx * p.kStrideB + (int64_t)kvHeadIdx * p.kStrideH;
    const int64_t vRow0 = (int64_t)batchIdx * p.vStrideB + (int64_t)kvHeadIdx * p.vStrideH;

    const int blockRow = blockIdx.x;
    const int delta = (int)p.keySeqLen - (int)p.seqLen;
    const float scaleLog2e = p.scale * kLog2E;
    const int halfWindow = p.windowSize / 2;

    __shared__ half2_t sQ[HH][Br];
    __shared__ half2_t sA[T::BufAElems];
    __shared__ half2_t sB[T::BufBElems];
    VRegs<HeadDim, Bc> vr;

    float acco[QPT][DPT];
    float rm[QPT];
    float rl[QPT];
    bool isActive[QPT];
    int gQq[QPT];

    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        rm[i] = kMaskedSentinel;
        rl[i] = 0.0f;

        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            acco[i][e] = 0.0f;
        }
        int qq = blockRow * Br + ty * QPT + i;
        gQq[i] = qq;
        isActive[i] = (qq < (int)p.seqLen);
    }

    /// Step 0: Q load + RoPE
    {
        const int t = ty * 16 + tx;
        constexpr int nThreads = 256;
        constexpr int totalQ = Br * T::HeadDimVec8;
        constexpr int itersQ = (totalQ + nThreads - 1) / nThreads;
        #pragma unroll
        for (int it = 0; it < itersQ; it++) {
            int e = t + it * nThreads;
            if (e >= totalQ) {
                continue;
            }
            int dg = e % T::HeadDimVec8;
            int q = e / T::HeadDimVec8;
            int sq = blockRow * Br + q;
            bool valid = sq < (int)p.seqLen;
            F16x8 q8 = (F16x8)(_Float16)0.0f;
            if (valid) {
                const _Float16* qRow = p.q + qRow0 + sq * p.qStrideS;
                q8 = *reinterpret_cast<const F16x8*>(qRow + dg * 8);
                if (p.hasRope) {
                    applyRope(q8, p.rotCos, p.rotSin, sq * T::HalfDim + dg * 4);
                }
            }
            int dh0 = dg * 4;
            sQ[dh0 + 0][q] = halfPair<0, 1>(q8);
            sQ[dh0 + 1][q] = halfPair<2, 3>(q8);
            sQ[dh0 + 2][q] = halfPair<4, 5>(q8);
            sQ[dh0 + 3][q] = halfPair<6, 7>(q8);
        }
    }
    __syncthreads();

    int firstKvBlock, lastKvBlock;
    kvBlockRange<Br, Bc, Causal, Window>(p, blockRow, delta, halfWindow, firstKvBlock, lastKvBlock);

    for (int kb = firstKvBlock; kb < lastKvBlock; kb++) {
        __syncthreads();
        loadKvBlock<HeadDim, Br, Bc, 1>(p, kb, kRow0, vRow0, sA, sB, vr);
        __syncthreads();
        processKvBlock<HeadDim, Br, Bc, Causal, Window>(
            p, kb, tx, ty, blockRow, delta, scaleLog2e, halfWindow, sQ, sA, sB, vr, gQq, isActive, acco, rm, rl);
    }

    /// Step 4: normalize + store
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        if (!isActive[i]) {
            continue;
        }
        int qq = gQq[i];
        float inv = softmaxInv(rl[i], rm[i], p.nullKeys);
        _Float16* oRow = p.o + oRow0 + qq * p.oStrideS;
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            int d = e * 16 + tx;
            oRow[d] = (_Float16)(acco[i][e] * inv);
        }
    }
}

/// GQA variant: one workgroup covers G heads sharing a kv_head. head_dim=64 only.
template <int HeadDim, int Br, int Bc, int G, bool Causal, bool Window>
__global__ __launch_bounds__(256) void forwardKernelGqa(KernelParams p) {
    using T = Tile<HeadDim, Br, Bc, G>;
    constexpr int QPT = T::QPT;
    constexpr int DPT = T::DPT;
    constexpr int HH = T::HH;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int ratio = (int)p.numHeads / (int)p.numKvHeads;
    const int headGroupsPerKvHead = (ratio + G - 1) / G;
    const int groupsPerBatch = (int)p.numKvHeads * headGroupsPerKvHead;
    const int groupIdx = blockIdx.y;
    const int batchIdx = groupIdx / groupsPerBatch;
    const int rem = groupIdx % groupsPerBatch;
    const int kvHeadIdx = rem / headGroupsPerKvHead;
    const int headGroupIdx = rem % headGroupsPerKvHead;
    const int headBase = headGroupIdx * G;

    const int64_t kRow0 = (int64_t)batchIdx * p.kStrideB + (int64_t)kvHeadIdx * p.kStrideH;
    const int64_t vRow0 = (int64_t)batchIdx * p.vStrideB + (int64_t)kvHeadIdx * p.vStrideH;

    bool headValid[G];
    int64_t qRow0[G];
    int64_t oRow0[G];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        headValid[g] = (headBase + g) < ratio;
        int headIdxG = kvHeadIdx * ratio + headBase + g;
        qRow0[g] = (int64_t)batchIdx * p.qStrideB + (int64_t)headIdxG * p.qStrideH;
        oRow0[g] = (int64_t)batchIdx * p.oStrideB + (int64_t)headIdxG * p.oStrideH;
    }

    const int blockRow = blockIdx.x;
    const int delta = (int)p.keySeqLen - (int)p.seqLen;
    const float scaleLog2e = p.scale * kLog2E;
    const int halfWindow = p.windowSize / 2;

    __shared__ half2_t sQ[G][HH][Br];
    __shared__ half2_t sA[T::BufAElems];
    __shared__ half2_t sB[T::BufBElems];
    VRegs<HeadDim, Bc> vr;

    float acco[G][QPT][DPT];
    float rm[G][QPT];
    float rl[G][QPT];
    bool isActive[QPT];
    int gQq[QPT];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            rm[g][i] = kMaskedSentinel;
            rl[g][i] = 0.0f;
            #pragma unroll
            for (int e = 0; e < DPT; e++) {
                acco[g][i][e] = 0.0f;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        int qq = blockRow * Br + ty * QPT + i;
        gQq[i] = qq;
        isActive[i] = (qq < (int)p.seqLen);
    }

    /// Step 0: Q load + RoPE
    {
        const int t = ty * 16 + tx;
        constexpr int nThreads = 256;
        constexpr int totalQ = G * Br * T::HeadDimVec8;
        constexpr int itersQ = (totalQ + nThreads - 1) / nThreads;
        #pragma unroll
        for (int it = 0; it < itersQ; it++) {
            int e = t + it * nThreads;
            if (e >= totalQ) {
                continue;
            }
            int dg = e % T::HeadDimVec8;
            int rest = e / T::HeadDimVec8;
            int q = rest % Br;
            int g = rest / Br;
            int sq = blockRow * Br + q;
            bool valid = headValid[g] && (sq < (int)p.seqLen);
            F16x8 q8 = (F16x8)(_Float16)0.0f;
            if (valid) {
                const _Float16* qRow = p.q + qRow0[g] + sq * p.qStrideS;
                q8 = *reinterpret_cast<const F16x8*>(qRow + dg * 8);
                if (p.hasRope) {
                    applyRope(q8, p.rotCos, p.rotSin, sq * T::HalfDim + dg * 4);
                }
            }
            int dh0 = dg * 4;
            sQ[g][dh0 + 0][q] = halfPair<0, 1>(q8);
            sQ[g][dh0 + 1][q] = halfPair<2, 3>(q8);
            sQ[g][dh0 + 2][q] = halfPair<4, 5>(q8);
            sQ[g][dh0 + 3][q] = halfPair<6, 7>(q8);
        }
    }
    __syncthreads();

    int firstKvBlock, lastKvBlock;
    kvBlockRange<Br, Bc, Causal, Window>(p, blockRow, delta, halfWindow, firstKvBlock, lastKvBlock);

    for (int kb = firstKvBlock; kb < lastKvBlock; kb++) {
        __syncthreads();
        loadKvBlock<HeadDim, Br, Bc, G>(p, kb, kRow0, vRow0, sA, sB, vr);
        __syncthreads();
        processKvBlockGqa<HeadDim, Br, Bc, G, Causal, Window>(
            p, kb, tx, ty, blockRow, delta, scaleLog2e, halfWindow, sQ, sA, sB, vr, gQq, isActive, acco, rm, rl);
    }

    /// Step 4: normalize + store
    #pragma unroll
    for (int g = 0; g < G; g++) {
        if (!headValid[g]) {
            continue;
        }
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            if (!isActive[i]) {
                continue;
            }
            int qq = gQq[i];
            float inv = softmaxInv(rl[g][i], rm[g][i], p.nullKeys);
            _Float16* oRow = p.o + oRow0[g] + qq * p.oStrideS;
            #pragma unroll
            for (int e = 0; e < DPT; e++) {
                int d = e * 16 + tx;
                oRow[d] = (_Float16)(acco[g][i][e] * inv);
            }
        }
    }
}

thread_local char g_lastError[256] = {0};

void setError(const char* msg) {
    std::snprintf(g_lastError, sizeof(g_lastError), "%s", msg);
}

bool strideSFitsInt32(int64_t s) {
    return s >= INT32_MIN && s <= INT32_MAX;
}

bool makeKernelParams(const ForwardParams& params, KernelParams& kp) {
    if (!strideSFitsInt32(params.qStrideS) || !strideSFitsInt32(params.kStrideS) ||
        !strideSFitsInt32(params.vStrideS) || !strideSFitsInt32(params.oStrideS)) {
        setError("sequence-axis stride exceeds int32 range - not a realistic tensor size for this kernel");
        return false;
    }
    kp.q = reinterpret_cast<const _Float16*>(params.q);
    kp.k = reinterpret_cast<const _Float16*>(params.k);
    kp.v = reinterpret_cast<const _Float16*>(params.v);
    kp.o = reinterpret_cast<_Float16*>(params.o);
    kp.rotCos = reinterpret_cast<const float*>(params.rot_cos);
    kp.rotSin = reinterpret_cast<const float*>(params.rot_sin);
    kp.qStrideB = params.qStrideB;
    kp.qStrideH = params.qStrideH;
    kp.qStrideS = static_cast<int32_t>(params.qStrideS);
    kp.kStrideB = params.kStrideB;
    kp.kStrideH = params.kStrideH;
    kp.kStrideS = static_cast<int32_t>(params.kStrideS);
    kp.vStrideB = params.vStrideB;
    kp.vStrideH = params.vStrideH;
    kp.vStrideS = static_cast<int32_t>(params.vStrideS);
    kp.oStrideB = params.oStrideB;
    kp.oStrideH = params.oStrideH;
    kp.oStrideS = static_cast<int32_t>(params.oStrideS);
    kp.numHeads = params.numHeads;
    kp.numKvHeads = params.numKvHeads;
    kp.seqLen = params.seqLen;
    kp.keySeqLen = params.keySeqLen;
    kp.scale = params.scale;
    kp.hasRope = params.hasRope;
    kp.windowSize = params.windowSize;
    kp.nullKeys = params.nullKeys;
    return true;
}

// VGPR budget for acco[] + staged V regs; enforced by tests/head_dim_resources.py.
constexpr int kTileVgprCap = 84;

constexpr int kTileCand[][2] = {{128, 64}, {128, 32}, {64, 64}, {64, 32}, {32, 64}, {32, 32}, {16, 32}};
constexpr int kNumTileCand = sizeof(kTileCand) / sizeof(kTileCand[0]);

constexpr int vRegVgprs(int H, int Bc) {
    return VRegsIters(H, Bc) * 8;
}

constexpr bool tileFits(int H, int Br, int Bc) {
    const int layout = pickLayout(H, Br, Bc, 1);
    if (layout < 0) {
        return false;
    }
    const bool alias = layout >= 2;
    if (alias && Bc != 32) {
        return false;
    }
    const int acco = (Br / 16) * (H / 16);
    return acco + (alias ? vRegVgprs(H, Bc) : 0) <= kTileVgprCap;
}

/// First fitting candidate tile. Sel: 0 = Br, 1 = Bc.
constexpr int chooseTile(int H, int Sel) {
    for (int i = 0; i < kNumTileCand; i++) {
        if (tileFits(H, kTileCand[i][0], kTileCand[i][1])) {
            return kTileCand[i][Sel];
        }
    }
    return 0;
}

template <int HeadDim, int Br, int Bc>
int launchOne(const ForwardParams& params, hipStream_t stream) {
    KernelParams kp{};
    if (!makeKernelParams(params, kp)) {
        return -1;
    }

    dim3 block(16, 16, 1);
    dim3 grid((params.seqLen + Br - 1) / Br, params.batchSize * params.numHeads, 1);

    bool causal = params.causal != 0;
    bool window = params.windowSize > 0;
#define RDNA_LAUNCH(C, W) \
    forwardKernel<HeadDim, Br, Bc, C, W><<<grid, block, 0, stream>>>(kp)
    if (causal && window) {
        RDNA_LAUNCH(true, true);
    } else if (causal) {
        RDNA_LAUNCH(true, false);
    } else if (window) {
        RDNA_LAUNCH(false, true);
    } else {
        RDNA_LAUNCH(false, false);
    }
#undef RDNA_LAUNCH

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        setError(hipGetErrorString(err));
        return -1;
    }
    return 0;
}

/// GQA-aware launch: head_dim=64 with numKvHeads < numHeads only.
template <int HeadDim, int Br, int Bc, int G>
int launchOneGqa(const ForwardParams& params, hipStream_t stream) {
    KernelParams kp{};
    if (!makeKernelParams(params, kp)) {
        return -1;
    }

    const uint32_t ratio = params.numHeads / params.numKvHeads;
    const uint32_t headGroupsPerKvHead = (ratio + G - 1) / G;
    dim3 block(16, 16, 1);
    dim3 grid((params.seqLen + Br - 1) / Br, params.batchSize * params.numKvHeads * headGroupsPerKvHead, 1);

    bool causal = params.causal != 0;
    bool window = params.windowSize > 0;
#define RDNA_LAUNCH_GQA(C, W) \
    forwardKernelGqa<HeadDim, Br, Bc, G, C, W><<<grid, block, 0, stream>>>(kp)
    if (causal && window) {
        RDNA_LAUNCH_GQA(true, true);
    } else if (causal) {
        RDNA_LAUNCH_GQA(true, false);
    } else if (window) {
        RDNA_LAUNCH_GQA(false, true);
    } else {
        RDNA_LAUNCH_GQA(false, false);
    }
#undef RDNA_LAUNCH_GQA

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        setError(hipGetErrorString(err));
        return -1;
    }
    return 0;
}

} // namespace

int forwardF16(const ForwardParams& params, void* stream) {
    hipStream_t hipStream = reinterpret_cast<hipStream_t>(stream);
    if (params.headDim == 64) {
        if (params.numKvHeads > 0 && params.numKvHeads < params.numHeads) {
            return launchOneGqa<64, 64, 64, 2>(params, hipStream);
        }
        return launchOne<64, 128, 64>(params, hipStream);
    }
    if (params.headDim == 128) {
        if (params.causal) {
            const uint64_t grid128 =
                static_cast<uint64_t>((params.seqLen + 127) / 128) * params.batchSize * params.numHeads;
            return grid128 >= kD128CausalBr128MinGrid
                       ? launchOne<128, 128, 64>(params, hipStream)
                       : launchOne<128, 64, 64>(params, hipStream);
        }
        return launchOne<128, 128, 64>(params, hipStream);
    }
#define RDNA_HEAD_DIM_CASE(H)     case H:                           return launchOne<H, chooseTile(H, 0), chooseTile(H, 1)>(params, hipStream)
    switch (params.headDim) {
        RDNA_HEAD_DIM_CASE(32);
        RDNA_HEAD_DIM_CASE(96);
        RDNA_HEAD_DIM_CASE(160);
        RDNA_HEAD_DIM_CASE(192);
        RDNA_HEAD_DIM_CASE(224);
        RDNA_HEAD_DIM_CASE(256);
        RDNA_HEAD_DIM_CASE(288);
        RDNA_HEAD_DIM_CASE(320);
        RDNA_HEAD_DIM_CASE(352);
        RDNA_HEAD_DIM_CASE(384);
        RDNA_HEAD_DIM_CASE(416);
        RDNA_HEAD_DIM_CASE(448);
        RDNA_HEAD_DIM_CASE(480);
        RDNA_HEAD_DIM_CASE(512);
    default:
        break;
    }
#undef RDNA_HEAD_DIM_CASE

    setError("unsupported head_dim (multiples of 32 from 32 to 512)");
    return -1;
}

const char* lastError() {
    return g_lastError[0] != '\0' ? g_lastError : nullptr;
}

bool hasDevice() {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

} // namespace rdna
