#pragma once

// Native HIP entry point for the INT8 forward-attention kernel (both GEMMs).
// head_dim 64/128, causal, window, GQA - no RoPE.

#include <cstdint>

namespace rdna {

/// Q/K/V point at int8_t, head_dim axis contiguous; output stays _Float16.
/// qScale/kScale/vScale are per-tensor fallback scales (symmetric dequant).
struct ForwardParamsInt8QK {
    const void* q = nullptr; // int8_t
    const void* k = nullptr; // int8_t
    const void* v = nullptr; // int8_t
    void* o = nullptr;       // _Float16

    int64_t qStrideB = 0, qStrideH = 0, qStrideS = 0;
    int64_t kStrideB = 0, kStrideH = 0, kStrideS = 0;
    int64_t vStrideB = 0, vStrideH = 0, vStrideS = 0;
    int64_t oStrideB = 0, oStrideH = 0, oStrideS = 0;

    uint32_t batchSize = 0;
    uint32_t numHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t seqLen = 0;
    uint32_t keySeqLen = 0;
    uint32_t headDim = 0;    // 64 or 128
    float scale = 1.0f;      // softmax scale, applied post-dequant
    float qScale = 1.0f;     // per-tensor Q dequant scale (if qScaleVec is null)
    float kScale = 1.0f;     // per-tensor K dequant scale (if kScaleVec is null)
    float vScale = 1.0f;     // per-tensor V dequant scale
    uint32_t causal = 0;
    int32_t windowSize = -1; // < 0 = full attention

    // Optional per-token scales: qScaleVec is (batchSize, numHeads, seqLen),
    // kScaleVec (batchSize, numKvHeads, keySeqLen). V has none.
    const void* qScaleVec = nullptr; // const float*
    const void* kScaleVec = nullptr; // const float*
};

/// Launches the INT8 forward kernel on `stream`. Non-blocking. 0 on success.
int forwardInt8QK(const ForwardParamsInt8QK& params, void* stream);

/// Reason the most recent forwardInt8QK() call failed, or nullptr.
const char* lastErrorInt8QK();

} // namespace rdna
