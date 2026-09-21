// Correctness gate and timing harness for src/rdna/monarch_forward_f16.hip.
// MonarchAttention (T=1) is approximate, so "correct" here means it matches a
// from-scratch fp64 CPU oracle of the same Monarch algorithm - not softmax
// attention, whose error is reported separately as the approximation cost.
//
//   hipcc tests/monarch_gate.hip src/rdna/monarch_forward_f16.hip \
//       src/rdna/fa2_forward_f16.hip -o monarch_gate \
//       --offload-arch=gfx1030 -O3 -ffast-math -std=c++17
//   ./monarch_gate gate
//   ./monarch_gate bench
//
// When A/B-ing two builds, interleave the bench runs: clock drift across a
// session exceeds any margin worth detecting here.

#include "../src/rdna/fa2_forward_f16.hpp"
#include "../src/rdna/monarch_forward_f16.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
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

using half = _Float16;

struct Shape {
    const char* name;
    int batch, heads, seqLen, headDim, blockB;
    bool withExact = true; // the fp64 exact oracle is O(N^2 d) on one core
};

double relRms(const std::vector<double>& got, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

/// fp64 MonarchAttention, T=1, for one (batch, head). q/k/v are [N][d].
void monarchOracle(const double* q, const double* k, const double* v, int N, int d, int b,
                   double scale, std::vector<double>& out) {
    const int m = N / b;
    std::vector<double> aL((size_t)b * m * d), y((size_t)b * m * d), cL((size_t)b * m);
    std::vector<double> s(b > m ? b : m);

    for (int kb = 0; kb < m; kb++) {
        for (int j = 0; j < b; j++) {
            const double* qr = q + (size_t)(kb * b + j) * d;
            double mx = -1e300;
            for (int i = 0; i < b; i++) {
                const double* kr = k + (size_t)(kb * b + i) * d;
                double acc = 0.0;
                for (int e = 0; e < d; e++) {
                    acc += qr[e] * kr[e];
                }
                s[i] = acc * scale;
                if (s[i] > mx) {
                    mx = s[i];
                }
            }
            double sum = 0.0;
            for (int i = 0; i < b; i++) {
                s[i] = std::exp(s[i] - mx);
                sum += s[i];
            }
            double ent = 0.0;
            double* aLr = &aL[((size_t)j * m + kb) * d];
            double* yr = &y[((size_t)j * m + kb) * d];
            for (int e = 0; e < d; e++) {
                aLr[e] = 0.0;
                yr[e] = 0.0;
            }
            for (int i = 0; i < b; i++) {
                double r = s[i] / sum;
                if (r > 0.0) {
                    ent += r * std::log(r);
                }
                const double* kr = k + (size_t)(kb * b + i) * d;
                const double* vr = v + (size_t)(kb * b + i) * d;
                for (int e = 0; e < d; e++) {
                    aLr[e] += r * kr[e];
                    yr[e] += r * vr[e];
                }
            }
            cL[(size_t)j * m + kb] = ent;
        }
    }

    out.assign((size_t)N * d, 0.0);
    for (int j = 0; j < b; j++) {
        for (int k0 = 0; k0 < m; k0++) {
            const double* qr = q + (size_t)(k0 * b + j) * d;
            double mx = -1e300;
            for (int k1 = 0; k1 < m; k1++) {
                const double* ar = &aL[((size_t)j * m + k1) * d];
                double acc = 0.0;
                for (int e = 0; e < d; e++) {
                    acc += qr[e] * ar[e];
                }
                s[k1] = acc * scale - cL[(size_t)j * m + k1];
                if (s[k1] > mx) {
                    mx = s[k1];
                }
            }
            double sum = 0.0;
            for (int k1 = 0; k1 < m; k1++) {
                s[k1] = std::exp(s[k1] - mx);
                sum += s[k1];
            }
            double* o = &out[(size_t)(k0 * b + j) * d];
            for (int k1 = 0; k1 < m; k1++) {
                double l = s[k1] / sum;
                const double* yr = &y[((size_t)j * m + k1) * d];
                for (int e = 0; e < d; e++) {
                    o[e] += l * yr[e];
                }
            }
        }
    }
}

/// fp64 exact softmax attention for one (batch, head).
void exactOracle(const double* q, const double* k, const double* v, int N, int d, double scale,
                 std::vector<double>& out) {
    out.assign((size_t)N * d, 0.0);
    std::vector<double> s(N);
    for (int i = 0; i < N; i++) {
        const double* qr = q + (size_t)i * d;
        double mx = -1e300;
        for (int j = 0; j < N; j++) {
            const double* kr = k + (size_t)j * d;
            double acc = 0.0;
            for (int e = 0; e < d; e++) {
                acc += qr[e] * kr[e];
            }
            s[j] = acc * scale;
            if (s[j] > mx) {
                mx = s[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            s[j] = std::exp(s[j] - mx);
            sum += s[j];
        }
        double* o = &out[(size_t)i * d];
        for (int j = 0; j < N; j++) {
            double pj = s[j] / sum;
            const double* vr = v + (size_t)j * d;
            for (int e = 0; e < d; e++) {
                o[e] += pj * vr[e];
            }
        }
    }
}

struct Buffers {
    half *dq = nullptr, *dk = nullptr, *dv = nullptr, *doo = nullptr;
    void* dws = nullptr;
    std::vector<half> hq, hk, hv;
    size_t elems = 0;
};

Buffers makeInputs(const Shape& s, unsigned seed, bool structured) {
    Buffers b;
    b.elems = (size_t)s.batch * s.heads * s.seqLen * s.headDim;
    b.hq.resize(b.elems);
    b.hk.resize(b.elems);
    b.hv.resize(b.elems);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < b.elems; i++) {
        b.hv[i] = (half)nd(rng);
    }
    if (!structured) {
        for (size_t i = 0; i < b.elems; i++) {
            b.hq[i] = (half)nd(rng);
            b.hk[i] = (half)nd(rng);
        }
    } else {
        // Smoothly varying along the sequence axis, so the Monarch fit has
        // some structure to latch onto. Only affects the reported error.
        for (int bh = 0; bh < s.batch * s.heads; bh++) {
            for (int t = 0; t < s.seqLen; t++) {
                for (int e = 0; e < s.headDim; e++) {
                    double ph = (double)t * (e / 2 + 1) * 8.0 * 3.14159 / (s.seqLen * s.headDim);
                    double base = (e % 2 == 0) ? std::sin(ph) : std::cos(ph);
                    size_t idx = ((size_t)bh * s.seqLen + t) * s.headDim + e;
                    b.hq[idx] = (half)(base + 0.3 * nd(rng));
                    b.hk[idx] = (half)(base + 0.3 * nd(rng));
                }
            }
        }
    }

    size_t bytes = b.elems * sizeof(half);
    HIP_CHECK(hipMalloc(&b.dq, bytes));
    HIP_CHECK(hipMalloc(&b.dk, bytes));
    HIP_CHECK(hipMalloc(&b.dv, bytes));
    HIP_CHECK(hipMalloc(&b.doo, bytes));
    HIP_CHECK(hipMemcpy(b.dq, b.hq.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.dk, b.hk.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.dv, b.hv.data(), bytes, hipMemcpyHostToDevice));
    return b;
}

void freeBuffers(Buffers& b) {
    (void)hipFree(b.dq);
    (void)hipFree(b.dk);
    (void)hipFree(b.dv);
    (void)hipFree(b.doo);
    if (b.dws) {
        (void)hipFree(b.dws);
    }
}

rdna::MonarchParams monarchParams(const Shape& s, Buffers& b) {
    rdna::MonarchParams p;
    p.q = b.dq;
    p.k = b.dk;
    p.v = b.dv;
    p.o = b.doo;
    p.batchSize = s.batch;
    p.numHeads = s.heads;
    p.seqLen = s.seqLen;
    p.headDim = s.headDim;
    p.blockB = s.blockB;
    p.scale = 1.0f / std::sqrt((float)s.headDim);
    if (!b.dws) {
        HIP_CHECK(hipMalloc(&b.dws, rdna::monarchWorkspaceBytes(p)));
    }
    p.workspace = b.dws;
    return p;
}

rdna::ForwardParams exactParams(const Shape& s, Buffers& b) {
    rdna::ForwardParams p;
    p.q = b.dq;
    p.k = b.dk;
    p.v = b.dv;
    p.o = b.doo;
    p.batchSize = s.batch;
    p.numHeads = s.heads;
    p.numKvHeads = s.heads;
    p.seqLen = s.seqLen;
    p.keySeqLen = s.seqLen;
    p.headDim = s.headDim;
    p.scale = 1.0f / std::sqrt((float)s.headDim);
    p.qStrideS = p.kStrideS = p.vStrideS = p.oStrideS = s.headDim;
    p.qStrideH = p.kStrideH = p.vStrideH = p.oStrideH = (int64_t)s.seqLen * s.headDim;
    p.qStrideB = p.kStrideB = p.vStrideB = p.oStrideB = (int64_t)s.heads * s.seqLen * s.headDim;
    return p;
}

int runGate() {
    const std::vector<Shape> shapes = {
        {"N2048 d64  b64/m32", 1, 2, 2048, 64, 64},
        {"N2048 d128 b64/m32", 1, 2, 2048, 128, 64},
        {"N4096 d64  b64/m64", 1, 1, 4096, 64, 64},
        {"N4096 d128 b64/m64", 1, 1, 4096, 128, 64},
        {"N1024 d128 b32/m32", 1, 2, 1024, 128, 32},
        {"N9216 d128 b96/m96", 1, 1, 9216, 128, 96, false},
        {"N5544 d128 b308/m18", 1, 2, 5544, 128, 308},
        {"N5544 d128 b18/m308", 1, 2, 5544, 128, 18},
        {"N5544 d64  b198/m28", 1, 2, 5544, 64, 198},
        {"N11970 d128 b630/m19", 1, 1, 11970, 128, 630, false},
        {"N9216 d64  b96/m96", 1, 1, 9216, 64, 96, false},
    };

    int failures = 0;
    std::printf("%-22s %-12s %-12s %s\n", "case", "vs oracle", "vs exact", "verdict");
    std::printf("%s\n", std::string(62, '-').c_str());

    for (const auto& s : shapes) {
        Buffers b = makeInputs(s, 1234, true);
        rdna::MonarchParams mp = monarchParams(s, b);
        if (rdna::monarchForwardF16(mp, nullptr) != 0) {
            std::printf("%-22s LAUNCH FAILED: %s\n", s.name, rdna::monarchLastError());
            freeBuffers(b);
            failures++;
            continue;
        }
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<half> hOut(b.elems);
        HIP_CHECK(hipMemcpy(hOut.data(), b.doo, b.elems * sizeof(half), hipMemcpyDeviceToHost));

        const int N = s.seqLen, d = s.headDim;
        const double scale = 1.0 / std::sqrt((double)d);
        std::vector<double> q((size_t)N * d), k((size_t)N * d), v((size_t)N * d);
        std::vector<double> got, refMonarch, refExact;

        double accMon = 0.0, accMonDen = 0.0, accEx = 0.0, accExDen = 0.0;
        for (int bh = 0; bh < s.batch * s.heads; bh++) {
            size_t off = (size_t)bh * N * d;
            for (size_t i = 0; i < (size_t)N * d; i++) {
                q[i] = (double)b.hq[off + i];
                k[i] = (double)b.hk[off + i];
                v[i] = (double)b.hv[off + i];
            }
            monarchOracle(q.data(), k.data(), v.data(), N, d, s.blockB, scale, refMonarch);
            if (s.withExact) {
                exactOracle(q.data(), k.data(), v.data(), N, d, scale, refExact);
            }
            got.assign((size_t)N * d, 0.0);
            for (size_t i = 0; i < (size_t)N * d; i++) {
                got[i] = (double)hOut[off + i];
            }
            for (size_t i = 0; i < got.size(); i++) {
                double dm = got[i] - refMonarch[i];
                accMon += dm * dm;
                accMonDen += refMonarch[i] * refMonarch[i];
                if (s.withExact) {
                    double de = got[i] - refExact[i];
                    accEx += de * de;
                    accExDen += refExact[i] * refExact[i];
                }
            }
        }
        double errOracle = std::sqrt(accMon / accMonDen);
        double errExact = s.withExact ? std::sqrt(accEx / accExDen) : -1.0;
        // fp16 storage of aL/y plus fp32 accumulation; the exact kernel's own
        // gate sits near 1e-3 on the same metric.
        bool pass = errOracle < 5e-3 && std::isfinite(errOracle);
        if (!pass) {
            failures++;
        }
        char ex[16];
        if (s.withExact) {
            std::snprintf(ex, sizeof(ex), "%.5f", errExact);
        } else {
            std::snprintf(ex, sizeof(ex), "skipped");
        }
        std::printf("%-22s %-12.5f %-12s %s\n", s.name, errOracle, ex, pass ? "PASS" : "FAIL");
        freeBuffers(b);
    }

    std::printf("\nvs oracle = fp16 kernel vs fp64 CPU MonarchAttention (must be small).\n");
    std::printf("vs exact  = the Monarch approximation error itself (reported, not gated).\n");
    return failures;
}

double timeIt(const std::function<void()>& launch, int iters) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    for (int i = 0; i < 5; i++) {
        launch();
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; i++) {
        launch();
    }
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return ms / iters;
}

/// Measured device memory each path needs beyond Q/K/V/O.
int runMem() {
    const std::vector<Shape> shapes = {
        {"N1024  d64  H16", 1, 16, 1024, 64, 32},
        {"N4096  d128 H16", 1, 16, 4096, 128, 64},
        {"N9216  d128 H8", 1, 8, 9216, 128, 96},
        {"N16384 d128 H8", 1, 8, 16384, 128, 128},
        {"N16384 d128 H40", 1, 40, 16384, 128, 128},
    };

    std::printf("%-17s %10s %10s %11s %9s %9s\n", "case", "QKVO MB", "exact MB", "monarch MB",
                "vs QKVO", "total x");
    std::printf("%s\n", std::string(72, '-').c_str());

    for (const auto& s : shapes) {
        double qkvo = 4.0 * s.batch * s.heads * (double)s.seqLen * s.headDim * sizeof(half);

        Buffers b = makeInputs(s, 7, false);
        size_t freeBefore = 0, total = 0;
        HIP_CHECK(hipMemGetInfo(&freeBefore, &total));
        rdna::MonarchParams mp = monarchParams(s, b); // allocates the workspace
        size_t freeAfter = 0;
        HIP_CHECK(hipMemGetInfo(&freeAfter, &total));
        double wsMeasured = (double)(freeBefore - freeAfter);
        double wsFormula = (double)rdna::monarchWorkspaceBytes(mp);

        // The exact kernel allocates nothing - its scratch is LDS and registers.
        std::printf("%-17s %10.1f %10.1f %11.1f %8.2fx %8.2fx\n", s.name, qkvo / 1e6, 0.0,
                    wsMeasured / 1e6, wsMeasured / qkvo, (qkvo + wsMeasured) / qkvo);
        if (wsMeasured + 1e-9 < wsFormula) {
            std::printf("  [warn] measured %.1f MB < formula %.1f MB\n", wsMeasured / 1e6,
                        wsFormula / 1e6);
        }
        freeBuffers(b);
    }

    std::printf("\nexact MB   = device memory the FlashAttention kernel needs beyond Q/K/V/O.\n");
    std::printf("monarch MB = the aL/y/cL workspace, measured via hipMemGetInfo.\n");
    std::printf("Neither path materializes an N x N score matrix.\n");
    return 0;
}

int runBench() {
    const std::vector<Shape> shapes = {
        {"N1024  d64  H16", 1, 16, 1024, 64, 32},
        {"N1024  d128 H16", 1, 16, 1024, 128, 32},
        {"N4096  d64  H16", 1, 16, 4096, 64, 64},
        {"N4096  d128 H16", 1, 16, 4096, 128, 64},
        {"N9216  d64  H8", 1, 8, 9216, 64, 96},
        {"N9216  d128 H8", 1, 8, 9216, 128, 96},
        {"N16384 d64  H8", 1, 8, 16384, 64, 128},
        {"N16384 d128 H8", 1, 8, 16384, 128, 128},
    };

    std::printf("%-17s %10s %10s %8s %10s %10s %8s %8s\n", "case", "exact ms", "monarch",
                "speedup", "exactTF/s", "monTF/s", "FLOPcut", "capture");
    std::printf("%s\n", std::string(94, '-').c_str());

    for (const auto& s : shapes) {
        Buffers b = makeInputs(s, 99, false);
        rdna::MonarchParams mp = monarchParams(s, b);
        rdna::ForwardParams ep = exactParams(s, b);

        if (rdna::monarchForwardF16(mp, nullptr) != 0) {
            std::printf("%-17s monarch launch failed: %s\n", s.name, rdna::monarchLastError());
            freeBuffers(b);
            continue;
        }
        if (rdna::forwardF16(ep, nullptr) != 0) {
            std::printf("%-17s exact launch failed: %s\n", s.name, rdna::lastError());
            freeBuffers(b);
            continue;
        }
        HIP_CHECK(hipDeviceSynchronize());

        double tExact = timeIt([&] { rdna::forwardF16(ep, nullptr); }, 50);
        double tMon = timeIt([&] { rdna::monarchForwardF16(mp, nullptr); }, 50);

        const double N = s.seqLen, d = s.headDim, bh = (double)s.batch * s.heads;
        const double bb = s.blockB, mm = s.seqLen / (double)s.blockB;
        double fExact = 4.0 * bh * N * N * d;
        double fMon = bh * N * d * (6.0 * bb + 4.0 * mm);
        double flopCut = fExact / fMon;
        double speedup = tExact / tMon;

        std::printf("%-17s %10.3f %10.3f %7.2fx %10.2f %10.2f %7.1fx %7.0f%%\n", s.name, tExact,
                    tMon, speedup, fExact / (tExact * 1e9), fMon / (tMon * 1e9), flopCut,
                    100.0 * speedup / flopCut);
        freeBuffers(b);
    }

    std::printf("\nFLOPcut = exact FLOPs / Monarch FLOPs (the theoretical ceiling).\n");
    std::printf("capture = achieved speedup as a fraction of that ceiling.\n");
    std::printf("Monarch is T=1 and includes both stages plus its workspace traffic.\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!rdna::hasDevice()) {
        std::printf("No usable HIP device.\n");
        return 1;
    }
    std::string mode = argc > 1 ? argv[1] : "gate";
    if (mode == "gate") {
        int f = runGate();
        std::printf("\n%s\n", f ? "GATE FAILED" : "GATE PASSED");
        return f ? 1 : 0;
    }
    if (mode == "bench") {
        return runBench();
    }
    if (mode == "mem") {
        return runMem();
    }
    std::printf("usage: monarch_gate [gate|bench|mem]\n");
    return 1;
}
