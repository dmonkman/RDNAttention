#pragma once

// Native HIP entry point for MonarchAttention (T=1), an approximate forward
// attention path. Output is NOT softmax attention - it is the Monarch
// projection of it. Exists to measure the achievable speedup, not to ship.

#include <cstddef>
#include <cstdint>

namespace rdna {

/// Q/K/V/output are _Float16 in contiguous [batch, heads, seq_len, head_dim]
/// order. seq_len must factor as blockB * blockM.
struct MonarchParams {
    const void* q = nullptr;
    const void* k = nullptr;
    const void* v = nullptr;
    void* o = nullptr;
    void* workspace = nullptr; // monarchWorkspaceBytes(), 256-byte aligned

    uint32_t batchSize = 0;
    uint32_t numHeads = 0;
    uint32_t seqLen = 0;  // N == blockB * blockM
    uint32_t headDim = 0; // 64 or 128
    uint32_t blockB = 0;  // contiguous block size; blockM = seqLen / blockB.
                          // Any divisor of seqLen - partial tiles are masked.
    float scale = 1.0f;
};

size_t monarchWorkspaceBytes(const MonarchParams& params);

/// Launches both stages on `stream`. Non-blocking. 0 on success.
int monarchForwardF16(const MonarchParams& params, void* stream);

/// Returns the reason the most recent monarchForwardF16() call failed, or nullptr.
const char* monarchLastError();

} // namespace rdna
