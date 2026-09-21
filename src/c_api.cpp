// RDNA_API on the definitions as well as c_api.hpp's declarations, on
// purpose: without it a signature that drifts from the header compiles and
// links cleanly as an unrelated C++ function, and the export just disappears
// from the DLL. With it, the mismatch is a compile error.
#include "c_api.hpp"

#include "rdna/fa2_forward_f16.hpp"
#include "rdna/fa2_forward_int8qk.hpp"
#include "rdna/monarch_forward_f16.hpp"

#include <array>
#include <cstdio>

namespace {

thread_local std::array<char, 512> g_errorMessage{};

void setError(const char* fmt, const char* detail) {
    std::snprintf(g_errorMessage.data(), g_errorMessage.size(), fmt, detail != nullptr ? detail : "unknown error");
}

} // namespace

RDNA_API int32_t rdna_has_device() {
    return rdna::hasDevice() ? 1 : 0;
}

RDNA_API const char* rdna_get_error() {
    return g_errorMessage.data();
}

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
    void* hip_stream) {
    if (null_keys != 0 && (causal != 0 || window_size >= 0)) {
        setError("%s", "null_keys is not valid with causal or windowed attention: both "
                       "mask by key position, which the dropped keys no longer have");
        return -2;
    }
    rdna::ForwardParams p{};
    p.q = q;
    p.k = k;
    p.v = v;
    p.o = output;
    p.rot_cos = rot_cos;
    p.rot_sin = rot_sin;
    p.qStrideB = q_stride_b;
    p.qStrideH = q_stride_h;
    p.qStrideS = q_stride_s;
    p.kStrideB = k_stride_b;
    p.kStrideH = k_stride_h;
    p.kStrideS = k_stride_s;
    p.vStrideB = v_stride_b;
    p.vStrideH = v_stride_h;
    p.vStrideS = v_stride_s;
    p.oStrideB = o_stride_b;
    p.oStrideH = o_stride_h;
    p.oStrideS = o_stride_s;
    p.batchSize = batch_size;
    p.numHeads = num_heads;
    p.numKvHeads = num_kv_heads;
    p.seqLen = seq_len;
    p.keySeqLen = key_seq_len;
    p.headDim = head_dim;
    p.scale = scale;
    p.causal = causal != 0 ? 1u : 0u;
    p.hasRope = has_rope != 0 ? 1u : 0u;
    p.windowSize = window_size;
    p.nullKeys = null_keys;

    int rc = rdna::forwardF16(p, hip_stream);
    if (rc != 0) {
        setError("Attention (fp16) failed: %s", rdna::lastError());
        return -3;
    }
    return 0;
}

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
    void* hip_stream) {
#if defined(RDNATTENTION_NO_INT8)
    // FP16-only build (see CMakeLists.txt): the INT8 kernel needs native
    // v_dot4_i32_i8 and was excluded because this build targets hardware
    // without it. The entry point stays in the ABI and fails cleanly rather
    // than vanishing, so a caller built against the full library still links.
    (void)q;
    (void)k;
    (void)v;
    (void)output;
    (void)q_stride_b;
    (void)q_stride_h;
    (void)q_stride_s;
    (void)k_stride_b;
    (void)k_stride_h;
    (void)k_stride_s;
    (void)v_stride_b;
    (void)v_stride_h;
    (void)v_stride_s;
    (void)o_stride_b;
    (void)o_stride_h;
    (void)o_stride_s;
    (void)batch_size;
    (void)num_heads;
    (void)num_kv_heads;
    (void)seq_len;
    (void)key_seq_len;
    (void)head_dim;
    (void)scale;
    (void)q_scale;
    (void)k_scale;
    (void)v_scale;
    (void)q_scale_vec;
    (void)k_scale_vec;
    (void)causal;
    (void)window_size;
    (void)hip_stream;
    setError("Attention (int8qk) unavailable: %s",
             "this library was built without the INT8 kernel (no native v_dot4_i32_i8 on the "
             "target architecture) - use rdna_attention_forward() instead");
    return -4;
#else
    rdna::ForwardParamsInt8QK p{};
    p.q = q;
    p.k = k;
    p.v = v;
    p.o = output;
    p.qStrideB = q_stride_b;
    p.qStrideH = q_stride_h;
    p.qStrideS = q_stride_s;
    p.kStrideB = k_stride_b;
    p.kStrideH = k_stride_h;
    p.kStrideS = k_stride_s;
    p.vStrideB = v_stride_b;
    p.vStrideH = v_stride_h;
    p.vStrideS = v_stride_s;
    p.oStrideB = o_stride_b;
    p.oStrideH = o_stride_h;
    p.oStrideS = o_stride_s;
    p.batchSize = batch_size;
    p.numHeads = num_heads;
    p.numKvHeads = num_kv_heads;
    p.seqLen = seq_len;
    p.keySeqLen = key_seq_len;
    p.headDim = head_dim;
    p.scale = scale;
    p.qScale = q_scale;
    p.kScale = k_scale;
    p.vScale = v_scale;
    p.causal = causal != 0 ? 1u : 0u;
    p.windowSize = window_size;
    p.qScaleVec = q_scale_vec;
    p.kScaleVec = k_scale_vec;

    int rc = rdna::forwardInt8QK(p, hip_stream);
    if (rc != 0) {
        setError("Attention (int8qk) failed: %s", rdna::lastErrorInt8QK());
        return -3;
    }
    return 0;
#endif
}

RDNA_API uint64_t rdna_monarch_workspace_bytes(
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim) {
    rdna::MonarchParams p{};
    p.batchSize = batch_size;
    p.numHeads = num_heads;
    p.seqLen = seq_len;
    p.headDim = head_dim;
    return static_cast<uint64_t>(rdna::monarchWorkspaceBytes(p));
}

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
    void* hip_stream) {
    if (workspace == nullptr) {
        setError("Attention (monarch) failed: %s", "workspace is null");
        return -2;
    }
    rdna::MonarchParams p{};
    p.q = q;
    p.k = k;
    p.v = v;
    p.o = output;
    p.workspace = workspace;
    p.batchSize = batch_size;
    p.numHeads = num_heads;
    p.seqLen = seq_len;
    p.headDim = head_dim;
    p.blockB = block_b;
    p.scale = scale;

    int rc = rdna::monarchForwardF16(p, hip_stream);
    if (rc != 0) {
        setError("Attention (monarch) failed: %s", rdna::monarchLastError());
        return -4;
    }
    return 0;
}
