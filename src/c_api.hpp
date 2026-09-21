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
/// null_keys counts all-zero keys the caller dropped rather than passed. A
/// zero key scores 0 against every query, so it holds softmax weight without
/// contributing any value; the kernel adds that weight back to the
/// denominator. Not valid with causal or windowed attention.
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
    uint32_t null_keys,
    void* hip_stream);

// ============================================================================
// MonarchAttention forward, T=1 (src/rdna/monarch_forward_f16.hip).
//
// APPROXIMATE: the output is the Monarch projection of softmax attention, not
// softmax attention. Sub-quadratic - Theta(N*sqrt(N)*d) rather than
// Theta(N^2*d) - and measured at 17-36x the fp16 kernel's throughput on
// gfx1030, at a rel_rms around 0.13-0.16 against exact attention.
//
// Q/K/V/output are contiguous _Float16 [batch, heads, seq_len, head_dim] -
// no strides, self-attention only (there is no key_seq_len: the algorithm has
// no Nq/Nk distinction). head_dim is 64 or 128. block_b need only divide
// seq_len - partial tiles are masked, so awkward video grids like WAN's
// 5544 = 308 x 18 work, just less efficiently than a multiple of 64.
//
// block_b is the contiguous block size. It is NOT free: it must align to the
// token grid, or accuracy collapses. Callers that know the (f,h,w) layout
// should derive it; callers that do not should not use this path.
// ============================================================================

/// Bytes of device scratch rdna_attention_forward_monarch() needs. The library
/// never allocates - the caller owns this buffer and may reuse one across
/// calls and layers.
RDNA_API uint64_t rdna_monarch_workspace_bytes(
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim);

RDNA_API int32_t rdna_attention_forward_monarch(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    void* workspace,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t block_b,
    float scale,
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
