// Standalone correctness gate for src/rdna/fa2_forward_int8qk.hip, against an
// fp64 oracle fed the same quantize-then-dequantize'd Q/K/V the kernel sees,
// so this measures the kernel's arithmetic and not quantization error.
//
// The oracle must reproduce the kernel's P asymmetry exactly: the row-sum
// comes from UNQUANTIZED softmax values (red16Sum runs before P is written),
// the PV numerator from QUANTIZED P (what sP8[] actually holds). Backwards,
// and a correct kernel looks wrong.
//
//   hipcc tests/hip_forward_int8qk_gate.hip src/rdna/fa2_forward_int8qk.hip \
//       -o hip_forward_int8qk_gate.exe --offload-arch=gfx1030 -O3 \
//       -ffast-math -std=c++17
//   ./hip_forward_int8qk_gate.exe

#include "../src/rdna/fa2_forward_int8qk.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                                          \
    do {                                                                                       \
        hipError_t e = (x);                                                                    \
        if (e != hipSuccess) {                                                                 \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__);  \
            std::exit(1);                                                                      \
        }                                                                                       \
    } while (0)

namespace {

constexpr double kPScale = 1.0 / 127.0;

uint32_t xrng(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
// 3-uniform sum: gives the per-tensor scale real outliers to clip.
float frandGaussianish(uint32_t& s) {
    float a = ((xrng(s) >> 16) & 4095) / 4096.0f - 0.5f;
    float b = ((xrng(s) >> 16) & 4095) / 4096.0f - 0.5f;
    float c = ((xrng(s) >> 16) & 4095) / 4096.0f - 0.5f;
    return (a + b + c) / 3.0f;
}

double relRms(const std::vector<float>& got, const std::vector<double>& ref) {
    double e = 0, r = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = (double)got[i] - ref[i];
        e += d * d;
        r += ref[i] * ref[i];
    }
    return r > 0 ? std::sqrt(e / r) : std::sqrt(e);
}

int8_t quantize(float x, float scale) {
    int v = (int)std::lround(x / scale);
    v = std::max(-127, std::min(127, v));
    return (int8_t)v;
}

// Q/K/V arrive already quantize-then-dequantize'd. See the header on P.
void refAttention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                   std::vector<double>& O, int batch, int numHeads, int numKvHeads, int seqLen, int keySeqLen,
                   int headDim, float scale, bool causal, int windowSize) {
    std::vector<double> pr(keySeqLen);
    const int ratio = numHeads / numKvHeads;
    const int delta = keySeqLen - seqLen;   // key k visible to query q iff k <= q + delta
    const int halfWindow = windowSize / 2;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < numHeads; h++) {
            const int kvh = h / ratio;
            const float* qBase = &Q[((size_t)(b * numHeads + h)) * seqLen * headDim];
            const float* kBase = &K[((size_t)(b * numKvHeads + kvh)) * keySeqLen * headDim];
            const float* vBase = &V[((size_t)(b * numKvHeads + kvh)) * keySeqLen * headDim];
            double* oBase = &O[((size_t)(b * numHeads + h)) * seqLen * headDim];
            for (int i = 0; i < seqLen; i++) {
                double mx = -1e300;
                std::vector<char> keep(keySeqLen, 1);
                for (int j = 0; j < keySeqLen; j++) {
                    if (causal && j > i + delta) keep[j] = 0;
                    if (windowSize > 0) {
                        if (causal) {
                            if (i + delta - j >= windowSize) keep[j] = 0;
                        } else if (std::abs(i + delta - j) > halfWindow) {
                            keep[j] = 0;
                        }
                    }
                    if (!keep[j]) { pr[j] = -1e300; continue; }
                    double s = 0;
                    for (int d = 0; d < headDim; d++) s += (double)qBase[i * headDim + d] * (double)kBase[j * headDim + d];
                    s *= scale;
                    pr[j] = s;
                    if (s > mx) mx = s;
                }
                double sum = 0; // unquantized - matches the kernel's rl[]/red16Sum
                for (int j = 0; j < keySeqLen; j++) {
                    pr[j] = keep[j] ? std::exp(pr[j] - mx) : 0.0;
                    sum += pr[j];
                }
                for (int d = 0; d < headDim; d++) {
                    double acc = 0;
                    for (int j = 0; j < keySeqLen; j++) {
                        int pq = (int)std::lround(pr[j] / kPScale);
                        pq = std::min(pq, 127); // pr[j] is in [0,1]; defensive only
                        double prDeq = (double)pq * kPScale;
                        acc += prDeq * (double)vBase[j * headDim + d];
                    }
                    oBase[i * headDim + d] = acc / sum;
                }
            }
        }
    }
}

struct Case {
    const char* name;
    int batch, numHeads, seqLen, keySeqLen, headDim;
    int numKvHeads = 0;      // 0 = same as numHeads (plain MHA)
    bool causal = false;
    int windowSize = -1;
    bool perTokenQK = false; // per-token Q/K dequant scales instead of per-tensor
};

bool runCase(const Case& c) {
    uint32_t seed = 424242;
    size_t qCount = (size_t)c.batch * c.numHeads * c.seqLen * c.headDim;
    const int kvHeads = c.numKvHeads ? c.numKvHeads : c.numHeads;
    size_t kvCount = (size_t)c.batch * kvHeads * c.keySeqLen * c.headDim;

    std::vector<float> qf(qCount), kf(kvCount), vf(kvCount);
    for (auto& x : qf) x = frandGaussianish(seed);
    for (auto& x : kf) x = frandGaussianish(seed);
    for (auto& x : vf) x = frandGaussianish(seed);

    auto maxAbs = [](const std::vector<float>& v) {
        float m = 0.0f;
        for (float x : v) m = std::max(m, std::fabs(x));
        return m;
    };
    float qScale = maxAbs(qf) / 127.0f;
    float kScale = maxAbs(kf) / 127.0f;
    float vScale = maxAbs(vf) / 127.0f;

    std::vector<int8_t> qi(qCount), ki(kvCount), vi(kvCount);
    std::vector<float> qDeq(qCount), kDeq(kvCount), vDeq(kvCount);
    std::vector<float> qScaleVec((size_t)c.batch * c.numHeads * c.seqLen, qScale);
    std::vector<float> kScaleVec((size_t)c.batch * kvHeads * c.keySeqLen, kScale);
    if (c.perTokenQK) {
        for (size_t r = 0; r < qScaleVec.size(); r++) {
            float m = 0.0f;
            for (int d = 0; d < c.headDim; d++) m = std::max(m, std::fabs(qf[r * c.headDim + d]));
            qScaleVec[r] = m > 0.0f ? m / 127.0f : 1.0f;
        }
        for (size_t r = 0; r < kScaleVec.size(); r++) {
            float m = 0.0f;
            for (int d = 0; d < c.headDim; d++) m = std::max(m, std::fabs(kf[r * c.headDim + d]));
            kScaleVec[r] = m > 0.0f ? m / 127.0f : 1.0f;
        }
    }
    for (size_t i = 0; i < qCount; i++) {
        float s = qScaleVec[i / c.headDim];
        qi[i] = quantize(qf[i], s);
        qDeq[i] = (float)qi[i] * s;
    }
    for (size_t i = 0; i < kvCount; i++) {
        float s = kScaleVec[i / c.headDim];
        ki[i] = quantize(kf[i], s);
        kDeq[i] = (float)ki[i] * s;
    }
    for (size_t i = 0; i < kvCount; i++) {
        vi[i] = quantize(vf[i], vScale);
        vDeq[i] = (float)vi[i] * vScale;
    }

    void *dQ, *dK, *dV, *dO, *dQS, *dKS;
    size_t oCount = qCount;
    HIP_CHECK(hipMalloc(&dQ, qCount * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dK, kvCount * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dV, kvCount * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&dO, oCount * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dQS, qScaleVec.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dKS, kScaleVec.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dQS, qScaleVec.data(), qScaleVec.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dKS, kScaleVec.data(), kScaleVec.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dQ, qi.data(), qCount * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, ki.data(), kvCount * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, vi.data(), kvCount * sizeof(int8_t), hipMemcpyHostToDevice));

    rdna::ForwardParamsInt8QK p{};
    p.q = dQ;
    p.k = dK;
    p.v = dV;
    p.o = dO;
    p.qStrideB = (int64_t)c.numHeads * c.seqLen * c.headDim;
    p.qStrideH = (int64_t)c.seqLen * c.headDim;
    p.qStrideS = c.headDim;
    p.kStrideB = (int64_t)kvHeads * c.keySeqLen * c.headDim;
    p.kStrideH = (int64_t)c.keySeqLen * c.headDim;
    p.kStrideS = c.headDim;
    p.vStrideB = p.kStrideB;
    p.vStrideH = p.kStrideH;
    p.vStrideS = c.headDim;
    p.oStrideB = p.qStrideB;
    p.oStrideH = p.qStrideH;
    p.oStrideS = c.headDim;
    p.batchSize = c.batch;
    p.numHeads = c.numHeads;
    p.numKvHeads = kvHeads;
    p.seqLen = c.seqLen;
    p.keySeqLen = c.keySeqLen;
    p.headDim = c.headDim;
    p.scale = 1.0f / std::sqrt((float)c.headDim);
    p.qScale = qScale;
    p.kScale = kScale;
    p.vScale = vScale;
    p.causal = c.causal ? 1 : 0;
    p.windowSize = c.windowSize;
    if (c.perTokenQK) {
        p.qScaleVec = dQS;
        p.kScaleVec = dKS;
    }

    int rc = rdna::forwardInt8QK(p, nullptr);
    if (rc != 0) {
        std::printf("[%s] launch failed: %s\n", c.name, rdna::lastErrorInt8QK());
        return false;
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<_Float16> oh(oCount);
    HIP_CHECK(hipMemcpy(oh.data(), dO, oCount * sizeof(_Float16), hipMemcpyDeviceToHost));
    std::vector<float> got(oCount);
    for (size_t i = 0; i < oCount; i++) got[i] = (float)oh[i];

    std::vector<double> ref(oCount);
    refAttention(qDeq, kDeq, vDeq, ref, c.batch, c.numHeads, kvHeads, c.seqLen, c.keySeqLen, c.headDim, p.scale,
                 c.causal, c.windowSize);

    double rms = relRms(got, ref);
    // Catches gross bugs; not a validated floor.
    bool ok = rms < 5e-2;
    std::printf("[%s] rel_rms=%.6g qScale=%.6g kScale=%.6g vScale=%.6g %s\n",
                c.name, rms, qScale, kScale, vScale, ok ? "OK" : "FAIL");

    (void)hipFree(dQ);
    (void)hipFree(dK);
    (void)hipFree(dV);
    (void)hipFree(dO);
    (void)hipFree(dQS);
    (void)hipFree(dKS);
    return ok;
}

} // namespace

int main() {
    std::vector<Case> cases = {
        {"d64_small", 1, 4, 96, 96, 64},
        {"d64_tail", 2, 4, 200, 200, 64},
        {"d64_batch4", 4, 8, 128, 128, 64},
        {"d64_kv_longer", 1, 4, 64, 300, 64},
        {"d64_single_head", 1, 1, 257, 257, 64},
        {"d64_large", 2, 16, 512, 512, 64},
        // seqLen>=1024 takes the BR=128 dispatch.
        {"d64_br128_clean", 1, 8, 1024, 1024, 64},
        {"d64_br128_tail", 2, 4, 1300, 1300, 64}, // not a multiple of 128

        // Causal
        {"d64_causal", 1, 4, 256, 256, 64, 0, true, -1},
        {"d64_causal_tail", 2, 4, 200, 200, 64, 0, true, -1},
        {"d64_causal_br128", 1, 4, 1024, 1024, 64, 0, true, -1},
        {"d64_causal_kv_longer", 1, 4, 64, 300, 64, 0, true, -1}, // delta != 0
        {"d64_causal_decode", 1, 4, 1, 256, 64, 0, true, -1},     // nothing masked

        // Sliding window
        {"d64_window", 1, 4, 256, 256, 64, 0, false, 64},
        {"d64_causal_window", 1, 4, 256, 256, 64, 0, true, 64},

        // GQA
        {"d64_gqa_r2", 1, 8, 256, 256, 64, 4, false, -1},
        {"d64_gqa_r8", 1, 8, 256, 256, 64, 1, false, -1},
        {"d64_mqa_causal", 2, 8, 300, 300, 64, 1, true, -1},
        {"d64_gqa_r4_br128", 1, 8, 1024, 1024, 64, 2, false, -1},

        // head_dim 128
        {"d128_small", 1, 4, 96, 96, 128},
        {"d128_tail", 2, 4, 200, 200, 128},
        {"d128_causal", 1, 4, 256, 256, 128, 0, true, -1},
        {"d128_gqa_causal", 1, 8, 300, 300, 128, 2, true, -1},
        {"d128_kv_longer", 1, 4, 64, 300, 128},
        {"d128_large", 1, 8, 1024, 1024, 128},
        // BR=128 dispatch. Head counts stay small: the CPU oracle is
        // O(B*H*Nq*Nk*D), so wide cases here take minutes.
        {"d128_br128", 1, 2, 2048, 2048, 128},
        {"d128_br128_tail", 1, 2, 2100, 2100, 128},         // not a multiple of 128
        {"d128_br128_bh", 8, 16, 1024, 64, 128},            // batch*heads=128, short KV
        {"d128_br128_causal", 1, 2, 2048, 2048, 128, 0, true, -1},

        // Per-token Q/K scales, crossed with masking and GQA: an out-of-range
        // key must take the fallback, not index past keySeqLen.
        {"ptok_d64",          1, 4, 256, 256, 64,  0, false, -1, true},
        {"ptok_d64_tail",     2, 4, 200, 200, 64,  0, false, -1, true},
        {"ptok_d64_causal",   1, 4, 256, 256, 64,  0, true,  -1, true},
        {"ptok_d64_window",   1, 4, 256, 256, 64,  0, true,  64, true},
        {"ptok_d64_gqa",      1, 8, 256, 256, 64,  2, false, -1, true},
        {"ptok_d64_kvlonger", 1, 4, 64,  300, 64,  0, false, -1, true},
        {"ptok_d64_br128",    1, 4, 1024, 1024, 64, 0, true, -1, true},
        {"ptok_d128",         1, 4, 256, 256, 128, 0, false, -1, true},
        {"ptok_d128_gqa_c",   1, 8, 300, 300, 128, 2, true,  -1, true},
    };
    bool allOk = true;
    for (const auto& c : cases) allOk &= runCase(c);
    std::printf(allOk ? "ALL PASS\n" : "SOME FAILED\n");
    return allOk ? 0 : 1;
}
