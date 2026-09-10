#pragma once

// Native HIP entry point for the fp16 forward attention kernel.

#include <cstdint>

namespace rdna {

/// Q/K/V/output point at _Float16, head_dim axis contiguous. rot_cos/rot_sin
/// (if has_rope) point at float, [max(seq_len, key_seq_len), head_dim/2],
/// shared across batch/heads.
struct ForwardParams {
    const void* q = nullptr;
    const void* k = nullptr;
    const void* v = nullptr;
    void* o = nullptr;
    const void* rot_cos = nullptr;
    const void* rot_sin = nullptr;

    // Strides in elements (not bytes).
    int64_t qStrideB = 0, qStrideH = 0, qStrideS = 0;
    int64_t kStrideB = 0, kStrideH = 0, kStrideS = 0;
    int64_t vStrideB = 0, vStrideH = 0, vStrideS = 0;
    int64_t oStrideB = 0, oStrideH = 0, oStrideS = 0;

    uint32_t batchSize = 0;
    uint32_t numHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t seqLen = 0;
    uint32_t keySeqLen = 0;
    uint32_t headDim = 0; // multiple of 32 up to 512
    float scale = 1.0f;
    uint32_t causal = 0;
    uint32_t hasRope = 0;
    int32_t windowSize = -1; // < 0 = full attention
};

/// Launches the forward kernel on `stream`. Non-blocking. 0 on success.
int forwardF16(const ForwardParams& params, void* stream);

/// Returns the reason the most recent forwardF16() call failed, or nullptr.
const char* lastError();

bool hasDevice();

} // namespace rdna
