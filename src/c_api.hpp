#pragma once

// Stable C ABI for rdnattention, consumed by rdnattention/__init__.py or any
// other language binding. Every function operates on caller-owned HIP device
// memory (e.g. a torch tensor's own .data_ptr()) - there is no tensor handle
// table, no allocation, no Vulkan involved anywhere in this library.

#include <cstdint>

#if defined(_WIN32)
#define RDNA_API extern "C" __declspec(dllexport)
#else
#define RDNA_API extern "C" __attribute__((visibility("default")))
#endif

RDNA_API int32_t rdna_has_device();

/// Reason the most recent call on this thread failed, or an empty string.
RDNA_API const char* rdna_get_error();

// ============================================================================
// FP16 forward attention (src/rdna/fa2_forward_f16.hip)
// ============================================================================

/// Q/K/V point at contiguous-in-head_dim _Float16 storage; *_stride_b/h/s
/// are strides in ELEMENTS for the batch/head/seq axes (head_dim itself must
/// be contiguous, stride 1 - not represented here). head_dim is any
/// multiple of 32 from 32 to 512. rot_cos/rot_sin (if has_rope) point at
/// contiguous float, shape [max(seq_len, key_seq_len), head_dim/2], shared
/// across batch and heads; pass nullptr for both to skip RoPE. num_kv_heads < num_heads
/// selects the GQA path. window_size < 0 means full attention.
///
/// hip_stream is a hipStream_t cast to void* - the launch is enqueued on it
/// and does not block the host. Returns 0 on success, nonzero (with
/// rdna_get_error() detail) on an unsupported head_dim or a launch failure.
RDNA_API int32_t rdna_attention_forward(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    const void* rot_cos,
    const void* rot_sin,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_s,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_s,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_s,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t key_seq_len,
    uint32_t head_dim,
    float scale,
    int32_t causal,
    int32_t has_rope,
    int32_t window_size,
    void* hip_stream);

// ============================================================================
// INT8 QK^T forward attention, both GEMMs (src/rdna/fa2_forward_int8qk.hip).
// head_dim 64/128, causal, window, GQA - no RoPE.
// ============================================================================

RDNA_API int32_t rdna_attention_forward_int8qk(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_s,
    int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_b, int64_t v_stride_h, int64_t v_stride_s,
    int64_t o_stride_b, int64_t o_stride_h, int64_t o_stride_s,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t key_seq_len,
    uint32_t head_dim,
    float scale,
    float q_scale,
    float k_scale,
    float v_scale,
    const void* q_scale_vec,
    const void* k_scale_vec,
    int32_t causal,
    int32_t window_size,
    void* hip_stream);
