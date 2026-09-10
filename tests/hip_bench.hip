// Standalone timing harness for src/rdna/fa2_forward_f16.hip - no torch, no
// CMake. Modes: "tuned" (head_dim 64/128) and "headdim" (one shape per size).
//
//   hipcc tests/hip_bench.hip src/rdna/fa2_forward_f16.hip \
//       -o hip_bench --offload-arch=gfx1030 -O3 -ffast-math -std=c++17
//   ./hip_bench tuned [--csv]
//
// When A/B-ing two builds, interleave the --csv runs: clock drift across a
// session exceeds any margin worth detecting here.

#include "../src/rdna/fa2_forward_f16.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                                          \
    do {                                                                                      \
        hipError_t e = (x);                                                                   \
        if (e != hipSuccess) {                                                                \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

namespace {

struct Shape {
    const char* name;
    int batch, numHeads, numKvHeads, seqLen, keySeqLen, headDim;
    bool causal;
};

const std::vector<Shape> kTuned = {
    {"d64_b1_1024", 1, 16, 16, 1024, 1024, 64, false},
    {"d64_b4_2048", 4, 16, 16, 2048, 2048, 64, false},
    {"d64_gqa_2048", 1, 16, 4, 2048, 2048, 64, false},
    {"d64_causal_4096", 2, 16, 16, 4096, 4096, 64, true},
    {"d128_causal_512", 1, 16, 16, 512, 512, 128, true},   // small grid -> Br=64
    {"d128_b1_1024", 1, 16, 16, 1024, 1024, 128, false},
    {"d128_causal_2048", 2, 16, 16, 2048, 2048, 128, true}, // large grid -> Br=128
    {"d128_b4_2048", 4, 16, 16, 2048, 2048, 128, false},
    {"d128_cross", 1, 16, 16, 4096, 512, 128, false},
};

std::vector<Shape> headDimSweep() {
    std::vector<Shape> v;
    static std::vector<std::string> names;
    names.clear();
    names.reserve(16);
    for (int d = 32; d <= 512; d += 32) {
        names.push_back("d" + std::to_string(d));
    }
    int i = 0;
    for (int d = 32; d <= 512; d += 32, i++) {
        v.push_back({names[i].c_str(), 1, 8, 8, 2048, 2048, d, false});
    }
    return v;
}

double flops(const Shape& s) {
    // QK^T and PV, 2 flops per MAC. Causal roughly halves the work.
    double f = 4.0 * s.batch * s.numHeads * (double)s.seqLen * s.keySeqLen * s.headDim;
    return s.causal ? f * 0.5 : f;
}

double timeShape(const Shape& s, int warmup, int iters) {
    size_t qN = (size_t)s.batch * s.numHeads * s.seqLen * s.headDim;
    size_t kvN = (size_t)s.batch * s.numKvHeads * s.keySeqLen * s.headDim;

    std::vector<_Float16> hq(qN), hk(kvN), hv(kvN);
    uint32_t seed = 12345;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)((seed >> 16) & 4095) / 4096.0f - 0.5f;
    };
    for (auto& x : hq) { x = (_Float16)rnd(); }
    for (auto& x : hk) { x = (_Float16)rnd(); }
    for (auto& x : hv) { x = (_Float16)rnd(); }

    void *dq, *dk, *dv, *dout;
    HIP_CHECK(hipMalloc(&dq, qN * 2));
    HIP_CHECK(hipMalloc(&dk, kvN * 2));
    HIP_CHECK(hipMalloc(&dv, kvN * 2));
    HIP_CHECK(hipMalloc(&dout, qN * 2));
    HIP_CHECK(hipMemcpy(dq, hq.data(), qN * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dk, hk.data(), kvN * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dv, hv.data(), kvN * 2, hipMemcpyHostToDevice));

    rdna::ForwardParams p{};
    p.q = dq;
    p.k = dk;
    p.v = dv;
    p.o = dout;
    p.qStrideB = (int64_t)s.numHeads * s.seqLen * s.headDim;
    p.qStrideH = (int64_t)s.seqLen * s.headDim;
    p.qStrideS = s.headDim;
    p.kStrideB = (int64_t)s.numKvHeads * s.keySeqLen * s.headDim;
    p.kStrideH = (int64_t)s.keySeqLen * s.headDim;
    p.kStrideS = s.headDim;
    p.vStrideB = p.kStrideB;
    p.vStrideH = p.kStrideH;
    p.vStrideS = s.headDim;
    p.oStrideB = p.qStrideB;
    p.oStrideH = p.qStrideH;
    p.oStrideS = s.headDim;
    p.batchSize = s.batch;
    p.numHeads = s.numHeads;
    p.numKvHeads = s.numKvHeads;
    p.seqLen = s.seqLen;
    p.keySeqLen = s.keySeqLen;
    p.headDim = s.headDim;
    p.scale = 1.0f / std::sqrt((float)s.headDim);
    p.causal = s.causal ? 1u : 0u;
    p.windowSize = -1;

    for (int i = 0; i < warmup; i++) {
        if (rdna::forwardF16(p, nullptr) != 0) {
            std::printf("launch failed: %s\n", rdna::lastError());
            std::exit(1);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < iters; i++) {
        rdna::forwardF16(p, nullptr);
    }
    HIP_CHECK(hipEventRecord(e1, nullptr));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipFree(dq));
    HIP_CHECK(hipFree(dk));
    HIP_CHECK(hipFree(dv));
    HIP_CHECK(hipFree(dout));
    return ms / iters;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "tuned";
    bool csv = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--csv") == 0) {
            csv = true;
        }
    }
    if (!rdna::hasDevice()) {
        std::printf("no HIP device\n");
        return 1;
    }
    std::vector<Shape> shapes = (mode == "headdim") ? headDimSweep() : kTuned;

    if (!csv) {
        std::printf("%-20s %10s %10s\n", "shape", "ms", "TFLOP/s");
    }
    for (const auto& s : shapes) {
        double ms = timeShape(s, 5, 20);
        double tf = flops(s) / (ms / 1000.0) / 1e12;
        if (csv) {
            std::printf("%s,%.6f,%.4f\n", s.name, ms, tf);
        } else {
            std::printf("%-20s %10.4f %10.2f\n", s.name, ms, tf);
        }
    }
    return 0;
}
