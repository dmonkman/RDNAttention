// MonarchAttention forward (fp16), T=1 refinement step.
//
// Stage 1 (al_y_cl): per contiguous block k, R = softmax(scale * Q_k K_k^T),
// then aL = R K_k, y = R V_k, and cL = sum(R log R) via an online-entropy
// accumulator folded into the online softmax.
// Stage 2 (z): per strided group j, L = softmax(scale * Q_j aL_j^T - cL_j),
// then o = L y_j.
//
// Both stages are a small attention over an effective sequence length of
// blockB and blockM respectively, so they reuse the register-tiled v_dot2 GEMM
// and 16-lane DPP reductions of the exact kernel.
#include "monarch_forward_f16.hpp"

#include "arch.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>

namespace rdna {
namespace {

using half2_t = _Float16 __attribute__((ext_vector_type(2)));
using F16x8 = _Float16 __attribute__((ext_vector_type(8)));

template <int I, int J>
__device__ __forceinline__ half2_t halfPair(F16x8 v) {
    return __builtin_shufflevector(v, v, I, J);
}
template <int I>
__device__ __forceinline__ half2_t halfPairAcross(F16x8 a, F16x8 b) {
    return __builtin_shufflevector(a, b, I, 8 + I);
}

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

constexpr float kLog2E = 1.442695041f;
// -ffast-math makes INFINITY undefined behaviour; the exact kernel uses the
// same finite sentinel.
constexpr float kNegInf = -1e30f;

/// Tile geometry, 256 threads as 16x16 tx/ty. Outs is the number of head_dim
/// wide outputs GEMM2 produces (2 for stage 1: aL and y; 1 for stage 2).
template <int HeadDim, int Br, int Bc, int Outs>
struct Tile {
    static constexpr int HH = HeadDim / 2;
    static constexpr int BCH = Bc / 2;
    static constexpr int QPT = Br / 16;
    static constexpr int KPT = Bc / 16;
    static constexpr int DPT = HeadDim / 16;
    static constexpr int OW = Outs * HeadDim; // GEMM2 output width
    static constexpr int OPT = Outs * DPT;
    static constexpr int HeadDimVec8 = HeadDim / 8;

    static_assert(HeadDim % 32 == 0, "head_dim must be a multiple of 32");
    static_assert(Bc >= 32 && Bc % 32 == 0, "the key mapping covers 32 columns per index pair");
    static_assert(Br >= 16 && Br % 16 == 0, "ty covers 16 rows");
    static_assert(HH % 4 == 0, "GEMM1 steps dh by 4");
    static_assert(BCH % 4 == 0, "GEMM2 steps kh by 4");

    static constexpr int KStride = Bc + 1;  // +1: GEMM1 reads a column per lane
    static constexpr int PStride = Br + 1;

    static constexpr int QElems = HH * Br;
    static constexpr int KElems = HH * KStride;
    static constexpr int VKElems = BCH * OW;
    static constexpr int PElems = BCH * PStride;
    static constexpr int LdsBytes = 4 * (QElems + KElems + VKElems + PElems);
    static_assert(LdsBytes <= 65536, "LDS budget exceeded - shrink Bc or Br");

    // Spreads the GEMM2 head_dim reads across banks; low 3 bits only, so it
    // stays inside the [0, OW) row.
    __device__ static __forceinline__ int vSwz(int d) {
        return d ^ ((d >> 4) & 7);
    }
};

/// Stages Br rows of A into sQ[dh][row] as half2 pairs along head_dim.
template <int HeadDim, int Br, int Bc, int Outs>
__device__ __forceinline__ void loadRowsToDh(
    const _Float16* base, int64_t rowStride, int rows, half2_t sQ[]) {
    using T = Tile<HeadDim, Br, Bc, Outs>;
    constexpr int nThreads = 256;
    constexpr int total = Br * T::HeadDimVec8;
    constexpr int iters = (total + nThreads - 1) / nThreads;
    const int t = threadIdx.y * 16 + threadIdx.x;

    #pragma unroll
    for (int it = 0; it < iters; it++) {
        int e = t + it * nThreads;
        if (e >= total) {
            continue;
        }
        int dg = e % T::HeadDimVec8;
        int r = e / T::HeadDimVec8;
        F16x8 v = (F16x8)(_Float16)0.0f;
        if (r < rows) {
            v = *reinterpret_cast<const F16x8*>(base + r * rowStride + dg * 8);
        }
        int dh0 = dg * 4;
        sQ[(dh0 + 0) * Br + r] = halfPair<0, 1>(v);
        sQ[(dh0 + 1) * Br + r] = halfPair<2, 3>(v);
        sQ[(dh0 + 2) * Br + r] = halfPair<4, 5>(v);
        sQ[(dh0 + 3) * Br + r] = halfPair<6, 7>(v);
    }
}

/// Stages Bc rows of K into sK[dh][col] for GEMM1.
template <int HeadDim, int Br, int Bc, int Outs>
__device__ __forceinline__ void loadColsToDh(
    const _Float16* base, int64_t rowStride, int cols, half2_t sK[]) {
    using T = Tile<HeadDim, Br, Bc, Outs>;
    constexpr int nThreads = 256;
    constexpr int total = Bc * T::HeadDimVec8;
    constexpr int iters = (total + nThreads - 1) / nThreads;
    const int t = threadIdx.y * 16 + threadIdx.x;

    #pragma unroll
    for (int it = 0; it < iters; it++) {
        int e = t + it * nThreads;
        if (e >= total) {
            continue;
        }
        int dg = e % T::HeadDimVec8;
        int c = e / T::HeadDimVec8;
        F16x8 v = (F16x8)(_Float16)0.0f;
        if (c < cols) {
            v = *reinterpret_cast<const F16x8*>(base + c * rowStride + dg * 8);
        }
        int dh0 = dg * 4;
        sK[(dh0 + 0) * T::KStride + c] = halfPair<0, 1>(v);
        sK[(dh0 + 1) * T::KStride + c] = halfPair<2, 3>(v);
        sK[(dh0 + 2) * T::KStride + c] = halfPair<4, 5>(v);
        sK[(dh0 + 3) * T::KStride + c] = halfPair<6, 7>(v);
    }
}

/// Stages Bc rows of one GEMM2 operand into sVK[kh][colOff + d], pairing
/// adjacent rows so GEMM2 can dot two of them at once.
template <int HeadDim, int Br, int Bc, int Outs>
__device__ __forceinline__ void loadRowsToPairs(
    const _Float16* base, int64_t rowStride, int rows, int colOff, half2_t sVK[]) {
    using T = Tile<HeadDim, Br, Bc, Outs>;
    constexpr int nThreads = 256;
    constexpr int total = T::BCH * T::HeadDimVec8;
    constexpr int iters = (total + nThreads - 1) / nThreads;
    const int t = threadIdx.y * 16 + threadIdx.x;

    #pragma unroll
    for (int it = 0; it < iters; it++) {
        int e = t + it * nThreads;
        if (e >= total) {
            continue;
        }
        int dg = e % T::HeadDimVec8;
        int kh = e / T::HeadDimVec8;
        F16x8 a = (F16x8)(_Float16)0.0f, b = (F16x8)(_Float16)0.0f;
        if (2 * kh < rows) {
            a = *reinterpret_cast<const F16x8*>(base + (2 * kh) * rowStride + dg * 8);
        }
        if (2 * kh + 1 < rows) {
            b = *reinterpret_cast<const F16x8*>(base + (2 * kh + 1) * rowStride + dg * 8);
        }
        int d0 = colOff + dg * 8;
        half2_t* row = sVK + kh * T::OW;
        row[T::vSwz(d0 + 0)] = halfPairAcross<0>(a, b);
        row[T::vSwz(d0 + 1)] = halfPairAcross<1>(a, b);
        row[T::vSwz(d0 + 2)] = halfPairAcross<2>(a, b);
        row[T::vSwz(d0 + 3)] = halfPairAcross<3>(a, b);
        row[T::vSwz(d0 + 4)] = halfPairAcross<4>(a, b);
        row[T::vSwz(d0 + 5)] = halfPairAcross<5>(a, b);
        row[T::vSwz(d0 + 6)] = halfPairAcross<6>(a, b);
        row[T::vSwz(d0 + 7)] = halfPairAcross<7>(a, b);
    }
}

/// S = A B^T for this thread's QPT x KPT register tile.
template <int HeadDim, int Br, int Bc, int Outs>
__device__ __forceinline__ void gemm1(
    const half2_t sQ[], const half2_t sK[], int tx, int ty,
    float accs[Tile<HeadDim, Br, Bc, Outs>::QPT][Tile<HeadDim, Br, Bc, Outs>::KPT]) {
    using T = Tile<HeadDim, Br, Bc, Outs>;
    constexpr int QPT = T::QPT, KPT = T::KPT, HH = T::HH;

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
                kv[j] = sK[dh * T::KStride + (j >> 1) * 32 + 2 * tx + (j & 1)];
            }
            half2_t qv[QPT];
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                qv[i] = sQ[dh * Br + ty * QPT + i];
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
}

/// acc += P * VK over this column tile.
template <int HeadDim, int Br, int Bc, int Outs>
__device__ __forceinline__ void gemm2(
    const half2_t sP[], const half2_t sVK[], int tx, int ty,
    float acco[Tile<HeadDim, Br, Bc, Outs>::QPT][Tile<HeadDim, Br, Bc, Outs>::OPT]) {
    using T = Tile<HeadDim, Br, Bc, Outs>;
    constexpr int QPT = T::QPT, OPT = T::OPT, BCH = T::BCH;

    #pragma unroll 1
    for (int kh0 = 0; kh0 < BCH; kh0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int kh = kh0 + s;
            half2_t vv[OPT];
            #pragma unroll
            for (int e = 0; e < OPT; e++) {
                vv[e] = sVK[kh * T::OW + T::vSwz(e * 16 + tx)];
            }
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                half2_t pv = sP[kh * T::PStride + ty * QPT + i];
                #pragma unroll
                for (int e = 0; e < OPT; e++) {
                    acco[i][e] = fdot2(pv, vv[e], acco[i][e]);
                }
            }
        }
    }
}

struct Stage1Params {
    const _Float16* q;
    const _Float16* k;
    const _Float16* v;
    _Float16* aL; // [b][m][d]
    _Float16* y;  // [b][m][d]
    float* cL;    // [b][m], base-2 units
    uint32_t seqLen, headDim, blockB, blockM;
    float scaleLog2e;
};

/// One workgroup: Br rows of one contiguous block, all blockB columns.
template <int HeadDim, int Br, int Bc>
__launch_bounds__(256) __global__ void monarchStage1(Stage1Params p) {
    using T = Tile<HeadDim, Br, Bc, 2>;
    constexpr int QPT = T::QPT, KPT = T::KPT, OPT = T::OPT, DPT = T::DPT;

    __shared__ half2_t sQ[T::QElems];
    __shared__ half2_t sK[T::KElems];
    __shared__ half2_t sVK[T::VKElems];
    __shared__ half2_t sP[T::PElems];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int rowTile = blockIdx.x;
    const int kBlock = blockIdx.y;
    const int bh = blockIdx.z;

    const int b = (int)p.blockB, m = (int)p.blockM, d = (int)p.headDim;
    const int64_t bhOff = (int64_t)bh * p.seqLen * d;
    const int row0 = kBlock * b + rowTile * Br; // first query row of this tile
    const int col0 = kBlock * b;                // first key row of this block

    const int rowsValid = b - rowTile * Br < Br ? b - rowTile * Br : Br;
    loadRowsToDh<HeadDim, Br, Bc, 2>(p.q + bhOff + (int64_t)row0 * d, d, rowsValid, sQ);

    float acco[QPT][OPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int e = 0; e < OPT; e++) {
            acco[i][e] = 0.0f;
        }
    }
    float rm[QPT], rl[QPT], rh[QPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        rm[i] = kNegInf;
        rl[i] = 0.0f;
        rh[i] = 0.0f;
    }

    for (int cb = 0; cb < b; cb += Bc) {
        const int colsValid = b - cb < Bc ? b - cb : Bc;
        __syncthreads();
        loadColsToDh<HeadDim, Br, Bc, 2>(p.k + bhOff + (int64_t)(col0 + cb) * d, d, colsValid, sK);
        loadRowsToPairs<HeadDim, Br, Bc, 2>(p.v + bhOff + (int64_t)(col0 + cb) * d, d, colsValid, 0, sVK);
        loadRowsToPairs<HeadDim, Br, Bc, 2>(p.k + bhOff + (int64_t)(col0 + cb) * d, d, colsValid, HeadDim, sVK);
        __syncthreads();

        float accs[QPT][KPT];
        gemm1<HeadDim, Br, Bc, 2>(sQ, sK, tx, ty, accs);

        float pr[QPT][KPT];
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            float sj[KPT];
            float lmax = kNegInf;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                // Zero-filled padding columns would otherwise score 0 and take
                // real softmax weight.
                int col = (j >> 1) * 32 + 2 * tx + (j & 1);
                sj[j] = col < colsValid ? accs[i][j] * p.scaleLog2e : kNegInf;
                lmax = fmaxf(lmax, sj[j]);
            }
            lmax = red16Max(lmax);
            float mnew = fmaxf(rm[i], lmax);
            if (mnew > rm[i] && rm[i] > kNegInf) {
                float corr = exp2f(rm[i] - mnew);
                // Online entropy: the rescale also shifts every retained
                // logit's offset from the old max to the new one.
                rh[i] = rh[i] * corr + corr * (rm[i] - mnew) * rl[i];
                rl[i] *= corr;
                #pragma unroll
                for (int e = 0; e < OPT; e++) {
                    acco[i][e] *= corr;
                }
            }
            float lsum = 0.0f, hsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                float t = sj[j] - mnew;
                float pj = exp2f(t);
                pr[i][j] = pj;
                lsum += pj;
                hsum += pj * t; // log2(pj) is exactly t - no transcendental
            }
            rl[i] += red16Sum(lsum);
            rh[i] += red16Sum(hsum);
            rm[i] = mnew;
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            #pragma unroll
            for (int j = 0; j < KPT; j += 2) {
                int kh = (j >> 1) * 16 + tx;
                sP[kh * T::PStride + ty * QPT + i] = half2_t{(_Float16)pr[i][j], (_Float16)pr[i][j + 1]};
            }
        }
        __syncthreads();

        gemm2<HeadDim, Br, Bc, 2>(sP, sVK, tx, ty, acco);
    }

    // aL/y/cL are indexed [j][k]: stage 2 walks k contiguously.
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        int j = rowTile * Br + ty * QPT + i;
        if (j >= b) {
            continue;
        }
        float inv = 1.0f / rl[i];
        int64_t base = bhOff + ((int64_t)j * m + kBlock) * d;
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            p.y[base + e * 16 + tx] = (_Float16)(acco[i][e] * inv);
            p.aL[base + e * 16 + tx] = (_Float16)(acco[i][DPT + e] * inv);
        }
        if (tx == 0) {
            // sum(R log2 R) = h/l - log2(l); stage 2 works in base 2 too.
            p.cL[(int64_t)bh * p.seqLen + (int64_t)j * m + kBlock] = rh[i] * inv - log2f(rl[i]);
        }
    }
}

struct Stage2Params {
    const _Float16* q;
    const _Float16* aL;
    const _Float16* y;
    const float* cL;
    _Float16* o;
    uint32_t seqLen, headDim, blockB, blockM;
    float scaleLog2e;
};

/// One workgroup: Br rows of one strided group j, all blockM columns.
template <int HeadDim, int Br, int Bc>
__launch_bounds__(256) __global__ void monarchStage2(Stage2Params p) {
    using T = Tile<HeadDim, Br, Bc, 1>;
    constexpr int QPT = T::QPT, KPT = T::KPT, OPT = T::OPT;

    __shared__ half2_t sQ[T::QElems];
    __shared__ half2_t sK[T::KElems];
    __shared__ half2_t sVK[T::VKElems];
    __shared__ half2_t sP[T::PElems];
    __shared__ float sBias[Bc];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int rowTile = blockIdx.x;
    const int j = blockIdx.y;
    const int bh = blockIdx.z;

    const int b = (int)p.blockB, m = (int)p.blockM, d = (int)p.headDim;
    const int64_t bhOff = (int64_t)bh * p.seqLen * d;
    const int k0 = rowTile * Br;

    // Query row k of group j is global row k*b + j, i.e. stride b*d.
    const int rowsValid = m - k0 < Br ? m - k0 : Br;
    loadRowsToDh<HeadDim, Br, Bc, 1>(
        p.q + bhOff + ((int64_t)k0 * b + j) * d, (int64_t)b * d, rowsValid, sQ);

    const _Float16* aLj = p.aL + bhOff + (int64_t)j * m * d;
    const _Float16* yj = p.y + bhOff + (int64_t)j * m * d;
    const float* cLj = p.cL + (int64_t)bh * p.seqLen + (int64_t)j * m;

    float acco[QPT][OPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int e = 0; e < OPT; e++) {
            acco[i][e] = 0.0f;
        }
    }
    float rm[QPT], rl[QPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        rm[i] = kNegInf;
        rl[i] = 0.0f;
    }

    for (int cb = 0; cb < m; cb += Bc) {
        const int colsValid = m - cb < Bc ? m - cb : Bc;
        __syncthreads();
        loadColsToDh<HeadDim, Br, Bc, 1>(aLj + (int64_t)cb * d, d, colsValid, sK);
        loadRowsToPairs<HeadDim, Br, Bc, 1>(yj + (int64_t)cb * d, d, colsValid, 0, sVK);
        {
            int t = ty * 16 + tx;
            if (t < Bc) {
                sBias[t] = t < colsValid ? cLj[cb + t] : 0.0f;
            }
        }
        __syncthreads();

        float accs[QPT][KPT];
        gemm1<HeadDim, Br, Bc, 1>(sQ, sK, tx, ty, accs);

        float bias[KPT];
        #pragma unroll
        for (int jj = 0; jj < KPT; jj++) {
            bias[jj] = sBias[(jj >> 1) * 32 + 2 * tx + (jj & 1)];
        }

        float pr[QPT][KPT];
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            float sj[KPT];
            float lmax = kNegInf;
            #pragma unroll
            for (int jj = 0; jj < KPT; jj++) {
                int col = (jj >> 1) * 32 + 2 * tx + (jj & 1);
                sj[jj] = col < colsValid ? accs[i][jj] * p.scaleLog2e - bias[jj] : kNegInf;
                lmax = fmaxf(lmax, sj[jj]);
            }
            lmax = red16Max(lmax);
            float mnew = fmaxf(rm[i], lmax);
            if (mnew > rm[i] && rm[i] > kNegInf) {
                float corr = exp2f(rm[i] - mnew);
                rl[i] *= corr;
                #pragma unroll
                for (int e = 0; e < OPT; e++) {
                    acco[i][e] *= corr;
                }
            }
            float lsum = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < KPT; jj++) {
                float pj = exp2f(sj[jj] - mnew);
                pr[i][jj] = pj;
                lsum += pj;
            }
            rl[i] += red16Sum(lsum);
            rm[i] = mnew;
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < QPT; i++) {
            #pragma unroll
            for (int jj = 0; jj < KPT; jj += 2) {
                int kh = (jj >> 1) * 16 + tx;
                sP[kh * T::PStride + ty * QPT + i] = half2_t{(_Float16)pr[i][jj], (_Float16)pr[i][jj + 1]};
            }
        }
        __syncthreads();

        gemm2<HeadDim, Br, Bc, 1>(sP, sVK, tx, ty, acco);
    }

    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        int k = k0 + ty * QPT + i;
        if (k >= m) {
            continue;
        }
        float inv = 1.0f / rl[i];
        int64_t base = bhOff + ((int64_t)k * b + j) * d;
        #pragma unroll
        for (int e = 0; e < OPT; e++) {
            p.o[base + e * 16 + tx] = (_Float16)(acco[i][e] * inv);
        }
    }
}

const char* g_lastError = nullptr;

template <int HeadDim, int Br, int Bc>
void launchPair(hipStream_t stream, const Stage1Params& s1, const Stage2Params& s2,
                uint32_t b, uint32_t m, size_t nBh) {
    dim3 g1((b + Br - 1) / Br, m, (unsigned)nBh);
    hipLaunchKernelGGL((monarchStage1<HeadDim, Br, Bc>), g1, dim3(16, 16), 0, stream, s1);
    dim3 g2((m + Br - 1) / Br, b, (unsigned)nBh);
    hipLaunchKernelGGL((monarchStage2<HeadDim, Br, Bc>), g2, dim3(16, 16), 0, stream, s2);
}

template <int HeadDim>
int launch(const MonarchParams& p, hipStream_t stream) {
    const uint32_t b = p.blockB, m = p.seqLen / p.blockB;

    auto* ws = static_cast<_Float16*>(p.workspace);
    const size_t perBh = (size_t)p.seqLen * p.headDim;
    const size_t nBh = (size_t)p.batchSize * p.numHeads;
    _Float16* aL = ws;
    _Float16* y = ws + nBh * perBh;
    float* cL = reinterpret_cast<float*>(ws + 2 * nBh * perBh);

    const float scaleLog2e = p.scale * kLog2E;

    Stage1Params s1{static_cast<const _Float16*>(p.q), static_cast<const _Float16*>(p.k),
                    static_cast<const _Float16*>(p.v), aL, y, cL,
                    p.seqLen, p.headDim, b, m, scaleLog2e};
    Stage2Params s2{static_cast<const _Float16*>(p.q), aL, y, cL,
                    static_cast<_Float16*>(p.o), p.seqLen, p.headDim, b, m, scaleLog2e};

    // Largest row tile that both block dimensions can fill. Partial tiles are
    // masked, so this is a throughput choice, not a correctness one. Measured,
    // not derived: Br=64 beats Br=32 by 1.3x at N=4096 and 1.4x at N=16384
    // despite costing two waves/SIMD of occupancy - GEMM register-tile reuse
    // wins. Bc at head_dim 128 is bounded by LDS: stage 1 stages both K and V
    // for GEMM2.
    const uint32_t smaller = b < m ? b : m;
    if (smaller >= 64) {
        launchPair<HeadDim, 64, (HeadDim == 64 ? 64 : 32)>(stream, s1, s2, b, m, nBh);
    } else if (smaller >= 48) {
        launchPair<HeadDim, 48, 32>(stream, s1, s2, b, m, nBh);
    } else {
        launchPair<HeadDim, 32, 32>(stream, s1, s2, b, m, nBh);
    }

    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        g_lastError = hipGetErrorString(e);
        return -1;
    }
    return 0;
}

} // namespace

size_t monarchWorkspaceBytes(const MonarchParams& p) {
    const size_t nBh = (size_t)p.batchSize * p.numHeads;
    const size_t perBh = (size_t)p.seqLen * p.headDim;
    return 2 * nBh * perBh * sizeof(_Float16) + nBh * (size_t)p.seqLen * sizeof(float);
}

int monarchForwardF16(const MonarchParams& p, void* stream) {
    g_lastError = nullptr;
    if (p.blockB == 0 || p.seqLen % p.blockB) {
        g_lastError = "seqLen must be divisible by blockB";
        return -1;
    }
    auto s = static_cast<hipStream_t>(stream);
    switch (p.headDim) {
        case 64:
            return launch<64>(p, s);
        case 128:
            return launch<128>(p, s);
        default:
            g_lastError = "head_dim must be 64 or 128";
            return -1;
    }
}

const char* monarchLastError() {
    return g_lastError;
}

} // namespace rdna
