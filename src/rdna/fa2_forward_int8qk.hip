// INT8 forward-attention kernel, both GEMMs. head_dim 64/128, causal,
// sliding window, GQA, cross-attention. No RoPE.
#include "fa2_forward_int8qk.hpp"

#include "arch.hpp"

#if defined(RDNA_NO_DOT)
#  error "The INT8 kernel requires native v_dot4_i32_i8 and is unsupported on this target by design: emulating it would run slower than the fp16 kernel, and less accurately. Reconfigure with -DRDNATTENTION_ENABLE_INT8=OFF, or target a GPU that has v_dot4_i32_i8."
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace rdna {
namespace {

__device__ __forceinline__ int packI8x4(int8_t a, int8_t b, int8_t c, int8_t d) {
    union {
        int8_t v[4];
        int i;
    } u;
    u.v[0] = a;
    u.v[1] = b;
    u.v[2] = c;
    u.v[3] = d;
    return u.i;
}

typedef int8_t Int8x16 __attribute__((ext_vector_type(16)));
static_assert(alignof(Int8x16) == 16, "Int8x16 must be naturally 128-bit-aligned for the reinterpret_cast loads below");

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
    x = fmaxf(x, dppMov<0xB1>(x));
    x = fmaxf(x, dppMov<0x4E>(x));
    x = fmaxf(x, dppMov<0x141>(x));
    x = fmaxf(x, dppMov<0x140>(x));
    return x;
}
__device__ __forceinline__ float red16Sum(float x) {
    x += dppMov<0xB1>(x);
    x += dppMov<0x4E>(x);
    x += dppMov<0x141>(x);
    x += dppMov<0x140>(x);
    return x;
}

constexpr float kLog2E = 1.442695041f;
constexpr float kMaskedSentinel = -1e30f;
constexpr float kPScale = 1.0f / 127.0f;

struct KernelParamsI8 {
    const int8_t* q;
    const int8_t* k;
    const int8_t* v;
    _Float16* o;
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
    float qScale, kScale, vScale;
    const float* qScaleVec;
    const float* kScaleVec;
    int32_t windowSize;
};

template <int HeadDim, int Br, int Bc>
struct Tile {
    static constexpr int QPT = Br / 16;
    static constexpr int KPT = Bc / 16;
    static constexpr int DPT = HeadDim / 16;
};

template <int HeadDim, int Bc>
struct TileI8 {
    static constexpr int DP4 = HeadDim / 4;
    static constexpr int HeadDimVec16 = HeadDim / 16;
    static constexpr int KStride = Bc + 1;
};

__device__ __forceinline__ int iabs(int x) {
    return x < 0 ? -x : x;
}

template <int Br, int Bc, bool Causal, bool Window>
__device__ __forceinline__ void kvBlockRange(const KernelParamsI8& p, int blockRow, int delta, int halfWindow,
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
__device__ __forceinline__ bool isMasked(const KernelParamsI8& p, int key, int qq, int delta, int halfWindow,
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

template <int HeadDim, int Bc>
__device__ __forceinline__ void loadKBlockI8(const KernelParamsI8& p, int kb, int64_t kRow0, int sK8[][Bc + 1]) {
    using T = TileI8<HeadDim, Bc>;
    const int t = threadIdx.y * 16 + threadIdx.x;
    constexpr int nThreads = 256;
    constexpr int totalK = Bc * T::HeadDimVec16;
    constexpr int itersK = (totalK + nThreads - 1) / nThreads;
    #pragma unroll
    for (int it = 0; it < itersK; it++) {
        int e = t + it * nThreads;
        if (e >= totalK) {
            continue;
        }
        int dg = e % T::HeadDimVec16;
        int kc = e / T::HeadDimVec16;
        int sk = kb * Bc + kc;

        int4 k16 = make_int4(0, 0, 0, 0);
        if (sk < (int)p.keySeqLen) {
            const int8_t* kRow = p.k + kRow0 + (int64_t)sk * p.kStrideS;
            k16 = *reinterpret_cast<const int4*>(kRow + dg * 16);
        }
        int dp0 = dg * 4;
        sK8[dp0 + 0][kc] = k16.x;
        sK8[dp0 + 1][kc] = k16.y;
        sK8[dp0 + 2][kc] = k16.z;
        sK8[dp0 + 3][kc] = k16.w;
    }
}

template <int HeadDim, int Bc>
__device__ __forceinline__ void loadVBlockI8(const KernelParamsI8& p, int kb, int64_t vRow0, int sV8[16][HeadDim]) {
    using T = TileI8<HeadDim, Bc>;
    const int t = threadIdx.y * 16 + threadIdx.x;
    constexpr int nThreads = 256;
    constexpr int totalV = 16 * T::HeadDimVec16;
    constexpr int itersV = (totalV + nThreads - 1) / nThreads;
    #pragma unroll
    for (int it = 0; it < itersV; it++) {
        int e = t + it * nThreads;
        if (e >= totalV) {
            continue;
        }
        int dg = e % T::HeadDimVec16;
        int txg = e / T::HeadDimVec16;

        int sk0 = kb * Bc + 2 * txg;
        int sk1 = sk0 + 1;
        int sk2 = kb * Bc + 32 + 2 * txg;
        int sk3 = sk2 + 1;

        Int8x16 v0 = (Int8x16)(int8_t)0, v1 = (Int8x16)(int8_t)0;
        Int8x16 v2 = (Int8x16)(int8_t)0, v3 = (Int8x16)(int8_t)0;
        int off = dg * 16;
        if (sk0 < (int)p.keySeqLen) {
            v0 = *reinterpret_cast<const Int8x16*>(p.v + vRow0 + (int64_t)sk0 * p.vStrideS + off);
        }
        if (sk1 < (int)p.keySeqLen) {
            v1 = *reinterpret_cast<const Int8x16*>(p.v + vRow0 + (int64_t)sk1 * p.vStrideS + off);
        }
        if (sk2 < (int)p.keySeqLen) {
            v2 = *reinterpret_cast<const Int8x16*>(p.v + vRow0 + (int64_t)sk2 * p.vStrideS + off);
        }
        if (sk3 < (int)p.keySeqLen) {
            v3 = *reinterpret_cast<const Int8x16*>(p.v + vRow0 + (int64_t)sk3 * p.vStrideS + off);
        }

        #pragma unroll
        for (int lane = 0; lane < 16; lane++) {
            sV8[txg][off + lane] = packI8x4(v0[lane], v1[lane], v2[lane], v3[lane]);
        }
    }
}

/// GEMM1 (sdot4) -> dequant -> softmax -> P write -> GEMM2 (sdot4) for one kv_block.
template <int HeadDim, int Br, int Bc, bool Causal, bool Window>
__device__ __forceinline__ void processKvBlockInt8QK(
    const KernelParamsI8& p, int kb, int tx, int ty, int blockRow, int delta, int halfWindow,
    const float qsRow[], int64_t ksBase, float scaleLog2e, int sQ8[][Br], int sK8[][Bc + 1],
    int sP8[16][Br + 1], int sV8[16][HeadDim], const int gQq[], const bool isActive[],
    float acco[][Tile<HeadDim, Br, Bc>::DPT], float rm[], float rl[]) {
    using T = Tile<HeadDim, Br, Bc>;
    using TI8 = TileI8<HeadDim, Bc>;
    constexpr int QPT = T::QPT;
    constexpr int KPT = T::KPT;
    constexpr int DPT = T::DPT;
    constexpr int DP4 = TI8::DP4;
    static_assert(KPT == 4, "P/V packing below assumes KPT==4 (Bc==64) - matches sdot4's own 4-lane width");

    /// Step 1: GEMM1 (QK^T)
    int accs[QPT][KPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            accs[i][j] = 0;
        }
    }

    #pragma unroll 1
    for (int dp0 = 0; dp0 < DP4; dp0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int dp = dp0 + s;
            int kv[KPT];
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                int kc = (j >> 1) * 32 + 2 * tx + (j & 1);
                kv[j] = sK8[dp][kc];
            }
            int qv[QPT];
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                qv[i] = sQ8[dp][ty * QPT + i];
            }
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                #pragma unroll
                for (int j = 0; j < KPT; j++) {
                    accs[i][j] = sdot4(qv[i], kv[j], accs[i][j]);
                }
            }
        }
    }
    __syncthreads();

    bool diag = Causal && (int)(kb * Bc + Bc - 1) > blockRow * Br + delta;

    float ksCol[KPT];
    #pragma unroll
    for (int j = 0; j < KPT; j++) {
        int key = kb * Bc + (j >> 1) * 32 + 2 * tx + (j & 1);
        ksCol[j] = (p.kScaleVec != nullptr && key < (int)p.keySeqLen) ? p.kScaleVec[ksBase + key] : p.kScale;
    }

    /// Step 2: softmax
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
            float sv = masked ? kMaskedSentinel : ((float)accs[i][j] * qsRow[i] * ksCol[j] * scaleLog2e);
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
        // min(...,127): -ffast-math can push pr[j] fractionally above 1.
        int p0 = min(127, (int)lroundf(pr[0] * 127.0f));
        int p1 = min(127, (int)lroundf(pr[1] * 127.0f));
        int p2 = min(127, (int)lroundf(pr[2] * 127.0f));
        int p3 = min(127, (int)lroundf(pr[3] * 127.0f));
        sP8[tx][ty * QPT + i] = packI8x4((int8_t)p0, (int8_t)p1, (int8_t)p2, (int8_t)p3);
    }
    __syncthreads();

    /// Step 3: GEMM2 (PV)
    int accI32[QPT][DPT];
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            accI32[i][e] = 0;
        }
    }

    #pragma unroll 1
    for (int txg0 = 0; txg0 < 16; txg0 += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int txg = txg0 + s;
            int vv[DPT];
            #pragma unroll
            for (int e = 0; e < DPT; e++) {
                vv[e] = sV8[txg][e * 16 + tx];
            }
            #pragma unroll
            for (int i = 0; i < QPT; i++) {
                int pv = sP8[txg][ty * QPT + i];
                #pragma unroll
                for (int e = 0; e < DPT; e++) {
                    accI32[i][e] = sdot4(pv, vv[e], accI32[i][e]);
                }
            }
        }
    }

    const float pvScale = kPScale * p.vScale;
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            acco[i][e] += (float)accI32[i][e] * pvScale;
        }
    }
}

template <int HeadDim, int Br, int Bc, bool Causal, bool Window>
__global__ __launch_bounds__(256) void forwardKernelInt8QK(KernelParamsI8 p) {
    using T = Tile<HeadDim, Br, Bc>;
    using TI8 = TileI8<HeadDim, Bc>;
    constexpr int QPT = T::QPT;
    constexpr int DPT = T::DPT;
    constexpr int DP4 = TI8::DP4;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int batchHeadIdx = blockIdx.y;
    const int batchIdx = batchHeadIdx / (int)p.numHeads;
    const int headIdx = batchHeadIdx % (int)p.numHeads;
    // GQA by K/V re-read: every query head loads its own tile.
    const int ratio = (int)p.numHeads / (int)p.numKvHeads;
    const int kvHeadIdx = headIdx / ratio;

    const int64_t qRow0 = (int64_t)batchIdx * p.qStrideB + (int64_t)headIdx * p.qStrideH;
    const int64_t oRow0 = (int64_t)batchIdx * p.oStrideB + (int64_t)headIdx * p.oStrideH;
    const int64_t kRow0 = (int64_t)batchIdx * p.kStrideB + (int64_t)kvHeadIdx * p.kStrideH;
    const int64_t vRow0 = (int64_t)batchIdx * p.vStrideB + (int64_t)kvHeadIdx * p.vStrideH;

    const int blockRow = blockIdx.x;
    const int delta = (int)p.keySeqLen - (int)p.seqLen;
    const int halfWindow = p.windowSize / 2;
    const float scaleLog2e = p.scale * kLog2E;
    const int64_t qsBase = ((int64_t)batchIdx * p.numHeads + headIdx) * p.seqLen;
    const int64_t ksBase = ((int64_t)batchIdx * p.numKvHeads + kvHeadIdx) * p.keySeqLen;

    __shared__ int sQ8[DP4][Br];
    __shared__ int sK8[DP4][Bc + 1];
    __shared__ int sP8[16][Br + 1];
    __shared__ int sV8[16][HeadDim];

    float acco[QPT][DPT];
    float rm[QPT];
    float rl[QPT];
    bool isActive[QPT];
    int gQq[QPT];
    float qsRow[QPT];

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
        qsRow[i] = (p.qScaleVec != nullptr && isActive[i]) ? p.qScaleVec[qsBase + qq] : p.qScale;
    }

    /// Step 0: Q load
    {
        const int t = ty * 16 + tx;
        constexpr int nThreads = 256;
        constexpr int totalQ = Br * TI8::HeadDimVec16;
        constexpr int itersQ = (totalQ + nThreads - 1) / nThreads;
        #pragma unroll
        for (int it = 0; it < itersQ; it++) {
            int e = t + it * nThreads;
            if (e >= totalQ) {
                continue;
            }
            int dg = e % TI8::HeadDimVec16;
            int q = e / TI8::HeadDimVec16;
            int sq = blockRow * Br + q;
            int4 q16 = make_int4(0, 0, 0, 0);
            if (sq < (int)p.seqLen) {
                const int8_t* qRow = p.q + qRow0 + (int64_t)sq * p.qStrideS;
                q16 = *reinterpret_cast<const int4*>(qRow + dg * 16);
            }
            int dp0 = dg * 4;
            sQ8[dp0 + 0][q] = q16.x;
            sQ8[dp0 + 1][q] = q16.y;
            sQ8[dp0 + 2][q] = q16.z;
            sQ8[dp0 + 3][q] = q16.w;
        }
    }
    __syncthreads();

    int firstKvBlock, lastKvBlock;
    kvBlockRange<Br, Bc, Causal, Window>(p, blockRow, delta, halfWindow, firstKvBlock, lastKvBlock);
    for (int kb = firstKvBlock; kb < lastKvBlock; kb++) {
        __syncthreads();
        loadKBlockI8<HeadDim, Bc>(p, kb, kRow0, sK8);
        loadVBlockI8<HeadDim, Bc>(p, kb, vRow0, sV8);
        __syncthreads();
        processKvBlockInt8QK<HeadDim, Br, Bc, Causal, Window>(p, kb, tx, ty, blockRow, delta, halfWindow, qsRow,
                                                              ksBase, scaleLog2e, sQ8, sK8, sP8, sV8, gQq, isActive,
                                                              acco, rm, rl);
    }

    /// Step 4: normalize + store
    #pragma unroll
    for (int i = 0; i < QPT; i++) {
        if (!isActive[i]) {
            continue;
        }
        int qq = gQq[i];
        float inv = rl[i] > 0.0f ? 1.0f / rl[i] : 0.0f;
        _Float16* oRow = p.o + oRow0 + qq * p.oStrideS;
        #pragma unroll
        for (int e = 0; e < DPT; e++) {
            int d = e * 16 + tx;
            oRow[d] = (_Float16)(acco[i][e] * inv);
        }
    }
}

thread_local char g_lastErrorInt8QK[256] = {0};

void setErrorInt8QK(const char* msg) {
    std::snprintf(g_lastErrorInt8QK, sizeof(g_lastErrorInt8QK), "%s", msg);
}

bool strideSFitsInt32I8(int64_t s) {
    return s >= INT32_MIN && s <= INT32_MAX;
}

template <int HeadDim, int Br, int Bc>
int launchInt8QK(const ForwardParamsInt8QK& params, hipStream_t stream) {
    if (!strideSFitsInt32I8(params.qStrideS) || !strideSFitsInt32I8(params.kStrideS) ||
        !strideSFitsInt32I8(params.vStrideS) || !strideSFitsInt32I8(params.oStrideS)) {
        setErrorInt8QK("sequence-axis stride exceeds int32 range - not a realistic tensor size for this kernel");
        return -1;
    }
    KernelParamsI8 kp{};
    kp.q = reinterpret_cast<const int8_t*>(params.q);
    kp.k = reinterpret_cast<const int8_t*>(params.k);
    kp.v = reinterpret_cast<const int8_t*>(params.v);
    kp.o = reinterpret_cast<_Float16*>(params.o);
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
    kp.qScale = params.qScale;
    kp.kScale = params.kScale;
    kp.vScale = params.vScale;
    kp.windowSize = params.windowSize;
    kp.qScaleVec = reinterpret_cast<const float*>(params.qScaleVec);
    kp.kScaleVec = reinterpret_cast<const float*>(params.kScaleVec);

    dim3 block(16, 16, 1);
    dim3 grid((params.seqLen + Br - 1) / Br, params.batchSize * params.numHeads, 1);

    bool causal = params.causal != 0;
    bool window = params.windowSize > 0;
    #define RDNA_LAUNCH_I8(C, W) forwardKernelInt8QK<HeadDim, Br, Bc, C, W><<<grid, block, 0, stream>>>(kp)
    if (causal && window) {
        RDNA_LAUNCH_I8(true, true);
    } else if (causal) {
        RDNA_LAUNCH_I8(true, false);
    } else if (window) {
        RDNA_LAUNCH_I8(false, true);
    } else {
        RDNA_LAUNCH_I8(false, false);
    }
    #undef RDNA_LAUNCH_I8

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        setErrorInt8QK(hipGetErrorString(err));
        return -1;
    }
    return 0;
}

} // namespace

int forwardInt8QK(const ForwardParamsInt8QK& params, void* stream) {
    hipStream_t hipStream = reinterpret_cast<hipStream_t>(stream);
    if (params.numKvHeads == 0 || params.numHeads % params.numKvHeads != 0) {
        setErrorInt8QK("numHeads must be a positive multiple of numKvHeads");
        return -1;
    }
    const bool br128 = params.seqLen >= 1024;
    if (params.headDim == 64) {
        return br128 ? launchInt8QK<64, 128, 64>(params, hipStream) : launchInt8QK<64, 64, 64>(params, hipStream);
    }
    if (params.headDim == 128) {
        const bool br128_d128 =
            params.seqLen >= 2048 ||
            (params.seqLen >= 1024 && (uint64_t)params.batchSize * params.numHeads >= 128);
        return br128_d128 ? launchInt8QK<128, 128, 64>(params, hipStream)
                          : launchInt8QK<128, 64, 64>(params, hipStream);
    }
    setErrorInt8QK("unsupported head_dim (only 64 and 128 are compiled in for the int8 kernel)");
    return -1;
}

const char* lastErrorInt8QK() {
    return g_lastErrorInt8QK[0] != '\0' ? g_lastErrorInt8QK : nullptr;
}

} // namespace rdna
