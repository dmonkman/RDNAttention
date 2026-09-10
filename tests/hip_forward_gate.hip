// Standalone correctness gate for src/rdna/fa2_forward_f16.hip, a from-
// scratch fp64 CPU oracle (rel_rms tolerance)
//
// Build directly, no CMake needed (both source files are required):
//   hipcc tests/hip_forward_gate.hip src/rdna/fa2_forward_f16.hip \
//       -o hip_forward_gate --offload-arch=gfx1030 -O2 -std=c++17
//   ./hip_forward_gate

#include "../src/rdna/fa2_forward_f16.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

uint32_t xrng(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}
float frand(uint32_t& s) {
    return ((xrng(s) >> 16) & 4095) / 4096.0f - 0.5f;
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

struct RefCfg {
    int batch, numHeads, numKvHeads, seqLen, keySeqLen, headDim;
    float scale;
    bool causal;
    int windowSize;
    bool hasRope;
    const float* rotCos = nullptr; // [seq_len, head_dim/2], shared batch/head
    const float* rotSin = nullptr;
};

void applyRope(std::vector<float>& x, int seqLen, int heads, int headDim, const float* cosB, const float* sinB) {
    int half = headDim / 2;
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seqLen; s++) {
            float* row = &x[((size_t)h * seqLen + s) * headDim];
            const float* c = &cosB[(size_t)s * half];
            const float* sn = &sinB[(size_t)s * half];
            for (int p = 0; p < half; p++) {
                float a = row[2 * p], b = row[2 * p + 1];
                row[2 * p] = a * c[p] - b * sn[p];
                row[2 * p + 1] = a * sn[p] + b * c[p];
            }
        }
    }
}

void refAttention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                   std::vector<double>& O, const RefCfg& c) {
    const int delta = c.keySeqLen - c.seqLen;
    const int gqa = c.numHeads / c.numKvHeads;
    const int halfWindow = c.windowSize / 2;
    std::vector<double> pr(c.keySeqLen);
    for (int b = 0; b < c.batch; b++) {
        for (int h = 0; h < c.numHeads; h++) {
            int kvh = h / gqa;
            const float* qBase = &Q[((size_t)(b * c.numHeads + h)) * c.seqLen * c.headDim];
            const float* kBase = &K[((size_t)(b * c.numKvHeads + kvh)) * c.keySeqLen * c.headDim];
            const float* vBase = &V[((size_t)(b * c.numKvHeads + kvh)) * c.keySeqLen * c.headDim];
            double* oBase = &O[((size_t)(b * c.numHeads + h)) * c.seqLen * c.headDim];
            for (int i = 0; i < c.seqLen; i++) {
                double mx = -1e300;
                int cnt = 0;
                for (int j = 0; j < c.keySeqLen; j++) {
                    bool masked = false;
                    if (c.causal && j > i + delta) masked = true;
                    if (c.windowSize > 0) {
                        if (c.causal) {
                            if (i + delta - j >= c.windowSize) masked = true;
                        } else if (std::abs(i + delta - j) > halfWindow) {
                            masked = true;
                        }
                    }
                    if (masked) {
                        pr[j] = -1e300;
                        continue;
                    }
                    double s = 0;
                    for (int d = 0; d < c.headDim; d++) s += (double)qBase[i * c.headDim + d] * (double)kBase[j * c.headDim + d];
                    s *= c.scale;
                    pr[j] = s;
                    if (s > mx) mx = s;
                    cnt++;
                }
                double sum = 0;
                for (int j = 0; j < c.keySeqLen; j++) {
                    if (pr[j] <= -1e299) {
                        pr[j] = 0;
                        continue;
                    }
                    pr[j] = std::exp(pr[j] - mx);
                    sum += pr[j];
                }
                for (int d = 0; d < c.headDim; d++) {
                    double acc = 0;
                    for (int j = 0; j < c.keySeqLen; j++) acc += pr[j] * (double)vBase[j * c.headDim + d];
                    oBase[i * c.headDim + d] = cnt ? acc / sum : 0.0;
                }
            }
        }
    }
}

struct Case {
    const char* name;
    int batch, numHeads, numKvHeads, seqLen, keySeqLen, headDim;
    bool causal;
    int windowSize;
    bool hasRope;
};

bool runCase(const Case& c) {
    uint32_t seed = 12345;
    size_t qCount = (size_t)c.batch * c.numHeads * c.seqLen * c.headDim;
    size_t kvCount = (size_t)c.batch * c.numKvHeads * c.keySeqLen * c.headDim;
    size_t oCount = qCount;

    std::vector<float> qf(qCount), kf(kvCount), vf(kvCount);
    for (auto& x : qf) x = frand(seed);
    for (auto& x : kf) x = frand(seed);
    for (auto& x : vf) x = frand(seed);

    std::vector<float> rotCos, rotSin;
    if (c.hasRope) {
        int half = c.headDim / 2;
        rotCos.resize((size_t)c.keySeqLen * half);
        rotSin.resize((size_t)c.keySeqLen * half);
        for (int s = 0; s < c.keySeqLen; s++) {
            for (int p = 0; p < half; p++) {
                float theta = (float)s / std::pow(10000.0f, (float)(2 * p) / c.headDim);
                rotCos[(size_t)s * half + p] = std::cos(theta);
                rotSin[(size_t)s * half + p] = std::sin(theta);
            }
        }
    }

    // Round first, so kernel and oracle see identical inputs.
    std::vector<_Float16> qh(qCount), kh(kvCount), vh(kvCount);
    for (size_t i = 0; i < qCount; i++) qh[i] = (_Float16)qf[i];
    for (size_t i = 0; i < kvCount; i++) kh[i] = (_Float16)kf[i];
    for (size_t i = 0; i < kvCount; i++) vh[i] = (_Float16)vf[i];
    for (size_t i = 0; i < qCount; i++) qf[i] = (float)qh[i];
    for (size_t i = 0; i < kvCount; i++) kf[i] = (float)kh[i];
    for (size_t i = 0; i < kvCount; i++) vf[i] = (float)vh[i];

    // Only the oracle's copy is rotated here; the kernel applies RoPE itself.
    std::vector<float> qfR = qf, kfR = kf;
    if (c.hasRope) {
        applyRope(qfR, c.seqLen, c.batch * c.numHeads, c.headDim, rotCos.data(), rotSin.data());
        applyRope(kfR, c.keySeqLen, c.batch * c.numKvHeads, c.headDim, rotCos.data(), rotSin.data());
    }

    void *dQ, *dK, *dV, *dO, *dRotCos = nullptr, *dRotSin = nullptr;
    HIP_CHECK(hipMalloc(&dQ, qCount * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dK, kvCount * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dV, kvCount * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dO, oCount * sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(dQ, qh.data(), qCount * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, kh.data(), kvCount * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, vh.data(), kvCount * sizeof(_Float16), hipMemcpyHostToDevice));
    if (c.hasRope) {
        int half = c.headDim / 2;
        HIP_CHECK(hipMalloc(&dRotCos, (size_t)c.keySeqLen * half * sizeof(float)));
        HIP_CHECK(hipMalloc(&dRotSin, (size_t)c.keySeqLen * half * sizeof(float)));
        HIP_CHECK(hipMemcpy(dRotCos, rotCos.data(), (size_t)c.keySeqLen * half * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRotSin, rotSin.data(), (size_t)c.keySeqLen * half * sizeof(float), hipMemcpyHostToDevice));
    }

    rdna::ForwardParams p{};
    p.q = dQ;
    p.k = dK;
    p.v = dV;
    p.o = dO;
    p.rot_cos = dRotCos;
    p.rot_sin = dRotSin;
    p.qStrideB = (int64_t)c.numHeads * c.seqLen * c.headDim;
    p.qStrideH = (int64_t)c.seqLen * c.headDim;
    p.qStrideS = c.headDim;
    p.kStrideB = (int64_t)c.numKvHeads * c.keySeqLen * c.headDim;
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
    p.numKvHeads = c.numKvHeads;
    p.seqLen = c.seqLen;
    p.keySeqLen = c.keySeqLen;
    p.headDim = c.headDim;
    p.scale = 1.0f / std::sqrt((float)c.headDim);
    p.causal = c.causal ? 1 : 0;
    p.hasRope = c.hasRope ? 1 : 0;
    p.windowSize = c.windowSize;

    int rc = rdna::forwardF16(p, nullptr);
    if (rc != 0) {
        std::printf("[%s] launch failed: %s\n", c.name, rdna::lastError());
        return false;
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<_Float16> oh(oCount);
    HIP_CHECK(hipMemcpy(oh.data(), dO, oCount * sizeof(_Float16), hipMemcpyDeviceToHost));
    std::vector<float> got(oCount);
    for (size_t i = 0; i < oCount; i++) got[i] = (float)oh[i];

    RefCfg rc2{c.batch,  c.numHeads, c.numKvHeads, c.seqLen,     c.keySeqLen,
               c.headDim, p.scale,   c.causal,     c.windowSize, c.hasRope};
    std::vector<double> ref(oCount);
    refAttention(qfR, kfR, vf, ref, rc2);

    double rms = relRms(got, ref);
    bool ok = rms < 5e-3;
    std::printf("[%s] rel_rms=%.6g %s\n", c.name, rms, ok ? "OK" : "FAIL");

    (void)hipFree(dQ);
    (void)hipFree(dK);
    (void)hipFree(dV);
    (void)hipFree(dO);
    if (dRotCos) (void)hipFree(dRotCos);
    if (dRotSin) (void)hipFree(dRotSin);
    return ok;
}

} // namespace

int main() {
    std::vector<Case> cases = {
        {"d64_dense_small", 1, 4, 4, 96, 96, 64, false, 0, false},
        {"d64_dense_tail", 2, 4, 4, 200, 200, 64, false, 0, false},
        {"d64_causal", 1, 4, 4, 257, 257, 64, true, 0, false},
        {"d64_gqa_causal", 1, 8, 2, 300, 300, 64, true, 0, false},

        {"d64_gqa_ratio2_exact", 1, 8, 4, 300, 300, 64, true, 0, false},
        {"d64_gqa_ratio8", 1, 16, 2, 300, 300, 64, true, 0, false},
        {"d64_gqa_ratio3_pad", 2, 6, 2, 257, 257, 64, true, 0, false},
        {"d64_gqa_ratio6_even", 1, 12, 2, 300, 300, 64, false, 0, false},
        {"d64_gqa_ratio5_pad", 2, 10, 2, 199, 199, 64, true, 64, true}, 
        {"d64_gqa_batch4", 4, 8, 2, 96, 96, 64, false, 0, false},

        {"d64_mqa", 2, 8, 1, 199, 199, 64, true, 0, false},
        {"d64_gqa_decode_1tok", 1, 8, 2, 1, 300, 64, false, 0, false},
        {"d64_gqa_decode_few", 2, 16, 4, 4, 300, 64, false, 0, false},
        {"d64_gqa_window", 1, 8, 2, 300, 300, 64, false, 64, false},
        {"d64_gqa_numkv8", 1, 32, 8, 256, 256, 64, true, 0, true},
        {"d64_window", 1, 4, 4, 300, 300, 64, false, 64, false},
        {"d64_causal_window", 1, 4, 4, 300, 300, 64, true, 32, false},
        {"d64_rope_causal", 2, 4, 4, 271, 271, 64, true, 0, true},
        {"d64_kv_longer", 1, 4, 4, 64, 300, 64, false, 0, false},
        {"d128_dense", 1, 4, 4, 130, 130, 128, false, 0, false},
        {"d128_causal", 1, 4, 4, 150, 150, 128, true, 0, false},
        {"d128_rope_gqa_causal", 1, 8, 4, 199, 199, 128, true, 0, true},

        {"d32_dense", 1, 2, 2, 100, 100, 32, false, 0, false},
        {"d96_dense", 1, 2, 2, 100, 100, 96, false, 0, false},
        {"d160_dense", 1, 2, 2, 100, 100, 160, false, 0, false},
        {"d192_dense", 1, 2, 2, 100, 100, 192, false, 0, false},
        {"d224_dense", 1, 2, 2, 100, 100, 224, false, 0, false},
        {"d256_dense", 1, 2, 2, 100, 100, 256, false, 0, false},
        {"d288_dense", 1, 2, 2, 100, 100, 288, false, 0, false},
        {"d320_dense", 1, 2, 2, 100, 100, 320, false, 0, false},
        {"d352_dense", 1, 2, 2, 100, 100, 352, false, 0, false},
        {"d384_dense", 1, 2, 2, 100, 100, 384, false, 0, false},
        {"d416_dense", 1, 2, 2, 100, 100, 416, false, 0, false},
        {"d448_dense", 1, 2, 2, 100, 100, 448, false, 0, false},
        {"d480_dense", 1, 2, 2, 100, 100, 480, false, 0, false},
        {"d512_dense", 1, 2, 2, 100, 100, 512, false, 0, false},

        {"d384_causal", 1, 2, 2, 100, 100, 384, true, 0, false},
        {"d384_gqa_causal", 1, 8, 2, 100, 100, 384, true, 0, false},
        {"d384_window", 1, 2, 2, 100, 100, 384, false, 32, false},
        {"d384_causal_window", 1, 2, 2, 100, 100, 384, true, 32, false},
        {"d384_rope_causal", 1, 2, 2, 100, 100, 384, true, 0, true},
        {"d384_kv_longer", 1, 2, 2, 40, 100, 384, false, 0, false},
        {"d384_decode_1tok", 1, 4, 2, 1, 100, 384, false, 0, false},
        {"d512_causal", 1, 2, 2, 100, 100, 512, true, 0, false},
        {"d512_gqa_rope_causal", 1, 4, 2, 100, 100, 512, true, 0, true},
        {"d512_window", 1, 2, 2, 100, 100, 512, false, 32, false},
        {"d512_kv_longer", 1, 2, 2, 40, 100, 512, false, 0, false},

        {"d256_rope_causal", 1, 2, 2, 100, 100, 256, true, 0, true},
        {"d256_gqa_window", 1, 8, 2, 100, 100, 256, false, 32, false},

        {"d352_causal_aligned", 1, 2, 2, 96, 96, 352, true, 0, false},
    };
    bool allOk = true;
    for (const auto& c : cases) allOk &= runCase(c);
    std::printf(allOk ? "ALL PASS\n" : "SOME FAILED\n");
    return allOk ? 0 : 1;
}
