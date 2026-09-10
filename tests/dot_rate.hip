// Issue-rate microbenchmark for the v_dot family.
//
// Each dot takes its own accumulator as an operand so LLVM cannot hoist the
// loop; NACC independent chains keep the pipeline full, measuring issue rate
// rather than dependent latency.
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef short s2 __attribute__((ext_vector_type(2)));
typedef unsigned short u2 __attribute__((ext_vector_type(2)));
#define NACC 16
#define ITERS 8192

template <int OP>
__global__ __launch_bounds__(256) void bench(int* out, int seed) {
    int ai[NACC];
    float af[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; i++) { ai[i] = i * 7 + seed + threadIdx.x; af[i] = (float)(i + 1); }
    h2 hy = {(_Float16)0.5f, (_Float16)1.25f};
    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        #pragma unroll
        for (int i = 0; i < NACC; i++) {
            int a = ai[i];
            if (OP == 0) af[i] = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), hy, af[i], false);
            if (OP == 1) ai[i] = __builtin_amdgcn_sdot2(__builtin_bit_cast(s2, a), __builtin_bit_cast(s2, a), a, false);
            if (OP == 2) ai[i] = __builtin_amdgcn_udot2(__builtin_bit_cast(u2, a), __builtin_bit_cast(u2, a), a, false);
            if (OP == 3) ai[i] = __builtin_amdgcn_sdot4(a, a, a, false);
            if (OP == 4) ai[i] = __builtin_amdgcn_udot4(a, a, a, false);
            if (OP == 5) ai[i] = __builtin_amdgcn_sdot8(a, a, a, false);
            if (OP == 6) ai[i] = __builtin_amdgcn_udot8(a, a, a, false);
            if (OP == 7) af[i] = fmaf(af[i], 1.000001f, af[i]);
        }
    }
    int s = 0;
    #pragma unroll
    for (int i = 0; i < NACC; i++) s += ai[i] + (int)af[i];
    if (s == 0x7fffffff) out[0] = s;
}

// Explicit instantiation, or --cuda-device-only emits nothing to inspect.
template __global__ void bench<0>(int*, int);
template __global__ void bench<1>(int*, int);
template __global__ void bench<2>(int*, int);
template __global__ void bench<3>(int*, int);
template __global__ void bench<4>(int*, int);
template __global__ void bench<5>(int*, int);
template __global__ void bench<6>(int*, int);
template __global__ void bench<7>(int*, int);

const char* NAMES[8] = {"fdot2  f16x2", "sdot2  i16x2", "udot2  u16x2", "sdot4  i8x4",
                        "udot4  u8x4",  "sdot8  i4x8",  "udot8  u4x8",  "v_fma_f32 (ref)"};
const int ELEMS[8] = {2, 2, 2, 4, 4, 8, 8, 1};
double base = 0;

template <int OP>
void run(int* d) {
    dim3 g(2048), b(256);
    for (int i = 0; i < 3; i++) { bench<OP><<<g, b>>>(d, i); }
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    const int REPS = 5;
    hipEventRecord(e0);
    for (int i = 0; i < REPS; i++) { bench<OP><<<g, b>>>(d, i); }
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    double instrs = (double)g.x * b.x * NACC * ITERS * REPS;
    double gi = instrs / (ms / 1e3) / 1e9, ge = gi * ELEMS[OP];
    if (OP == 0) base = ge;
    std::printf("  %-16s %7.1f Ginstr/s  %8.1f Gelem/s   %5.2fx fdot2 elem rate\n",
                NAMES[OP], gi, ge, base > 0 ? ge / base : 1.0);
    hipEventDestroy(e0); hipEventDestroy(e1);
}
int main() {
    int* d;
    (void)hipMalloc(&d, 4);
    std::printf("v_dot issue rate on gfx1030 - %d independent chains, %d iters\n", NACC, ITERS);
    run<0>(d); run<1>(d); run<2>(d); run<3>(d); run<4>(d); run<5>(d); run<6>(d); run<7>(d);
    (void)hipFree(d);
    return 0;
}
