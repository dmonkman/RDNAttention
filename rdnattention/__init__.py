"""
rdnattention - native HIP FlashAttention-2-style forward attention for AMD
RDNA2 (and untested: RDNA1, late GCN, some iGPUs).

Dispatches directly against torch tensors' own HIP device memory - no
staging, no CPU round trip, no separate GPU context to initialize. Requires
torch already built against ROCm and a supported GPU.

    >>> import torch
    >>> from rdnattention import flash_attn
    >>> q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
    >>> k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
    >>> v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
    >>> out = flash_attn(q, k, v, is_causal=True)
"""

import ctypes
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

_lib = None

SUPPORTED_HEAD_DIMS = tuple(range(32, 513, 32))
SUPPORTED_HEAD_DIMS_INT8 = (64, 128)


class RDNAttentionError(Exception):
    """Raised for any rdnattention library error."""


def _find_library() -> Path:
    system = platform.system()
    if system == "Linux":
        lib_name = "librdnattention.so"
    elif system == "Windows":
        lib_name = "rdnattention.dll"
    else:
        raise RDNAttentionError(f"Unsupported platform: {system} (HIP/ROCm requires Linux or Windows)")

    override = os.environ.get("RDNATTENTION_LIB")
    if override:
        path = Path(override)
        if not path.exists():
            raise RDNAttentionError(f"RDNATTENTION_LIB points at a missing file: {path}")
        logger.debug(f"Loading library from RDNATTENTION_LIB={path}")
        return path

    packaged = Path(__file__).parent / "lib" / lib_name
    dev = Path(__file__).parent.parent / "build" / lib_name

    # In a source checkout the dev build wins, or a stale lib/ silently
    # shadows every rebuild.
    in_source_tree = (Path(__file__).parent.parent / "CMakeLists.txt").exists()
    candidates = [dev, packaged] if in_source_tree else [packaged, dev]
    for path in candidates:
        if path.exists():
            if in_source_tree and path == dev and packaged.exists():
                logger.warning(f"Ignoring the packaged {packaged}: this is a source checkout, "
                               f"so {dev} takes priority. Delete lib/ to silence this.")
            logger.debug(f"Loading library from {path}")
            return path

    raise RDNAttentionError(
        f"Could not find the rdnattention library ({lib_name}). "
        "Build it with CMake first (cmake -S . -B build && cmake --build build), "
        f"or pip install rdnattention.\nSearched: {[str(p) for p in candidates]}"
    )


def _load_library(path: Path):
    # Windows: amdhip64_*.dll is only found if already resident (torch imported
    # first) or its directory was added explicitly - PATH is not consulted.
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        if platform.system() != "Windows":
            raise
        hip_path = os.environ.get("HIP_PATH")
        if not hip_path:
            raise
        bin_dir = Path(hip_path) / "bin"
        if not bin_dir.is_dir():
            raise
        os.add_dll_directory(str(bin_dir))
        return ctypes.CDLL(str(path))


def _lib_handle():
    global _lib
    if _lib is None:
        lib = _load_library(_find_library())
        lib.rdna_has_device.argtypes = []
        lib.rdna_has_device.restype = ctypes.c_int32
        lib.rdna_get_error.argtypes = []
        lib.rdna_get_error.restype = ctypes.c_char_p
        lib.rdna_attention_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.rdna_attention_forward.restype = ctypes.c_int32
        lib.rdna_attention_forward_int8qk.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.rdna_attention_forward_int8qk.restype = ctypes.c_int32
        _lib = lib
    return _lib


def has_device() -> bool:
    """Whether a usable HIP device is present."""
    return _lib_handle().rdna_has_device() == 1


def _validate_inputs(query, key, value, expected_dtype) -> None:
    import torch

    if not isinstance(query, torch.Tensor):
        raise TypeError("requires torch.Tensor inputs (GPU tensors already on this device)")
    for name, t in (("query", query), ("key", key), ("value", value)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be on a HIP device")
        if t.dtype != expected_dtype:
            raise ValueError(f"{name} must be {expected_dtype}, got {t.dtype}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4D [batch, heads, seq, head_dim], got shape {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous - call .contiguous() first")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(f"Batch size must match: Q={query.shape}, K={key.shape}, V={value.shape}")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError(f"Q heads ({query.shape[1]}) must be a multiple of K heads ({key.shape[1]})")


def _check_head_dim(head_dim: int, who: str, allowed=SUPPORTED_HEAD_DIMS):
    if head_dim not in allowed:
        raise RDNAttentionError(
            f"{who}: head_dim {head_dim} not supported "
            f"(multiples of 32 from {allowed[0]} to {allowed[-1]})")


def _rope_ptrs(rot_cos, rot_sin, query, seq_len, key_seq_len, head_dim):
    """Validates the angle tables. Returns (cos_ptr, sin_ptr, has_rope)."""
    import torch

    if rot_cos is None and rot_sin is None:
        return None, None, 0
    if rot_cos is None or rot_sin is None:
        raise ValueError("pass both rot_cos and rot_sin, or neither")

    rows = max(seq_len, key_seq_len)
    for name, t in (("rot_cos", rot_cos), ("rot_sin", rot_sin)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.device != query.device:
            raise ValueError(f"{name} must be on the same device as query")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous - call .contiguous() first")
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2D [positions, head_dim // 2], got {tuple(t.shape)}")
        if t.shape[1] != head_dim // 2:
            raise ValueError(f"{name} must have head_dim // 2 = {head_dim // 2} columns, "
                             f"got {t.shape[1]}")
        if t.shape[0] < rows:
            raise ValueError(f"{name} needs at least max(seq_len, key_seq_len) = {rows} rows, "
                             f"got {t.shape[0]}")
    return rot_cos.data_ptr(), rot_sin.data_ptr(), 1


def flash_attn(query, key, value, is_causal: bool = False, window_size: int = -1,
               rot_cos=None, rot_sin=None):
    """FlashAttention-2-style forward. query/key/value: contiguous float16
    [batch, heads, seq, head_dim] on a HIP device, head_dim in
    SUPPORTED_HEAD_DIMS. key/value may have fewer heads than query (GQA), as
    an exact divisor. Returns a new tensor shaped like query.

    Pass rot_cos and rot_sin together to apply rotary embeddings to Q and K
    inside the kernel. Both are contiguous float32
    [max(seq_len, key_seq_len), head_dim // 2], shared across batch and heads,
    indexed by position within the tensor starting at row 0.

    Only head_dim 64 and 128 have tuned tile sizes.
    """
    import torch

    _validate_inputs(query, key, value, torch.float16)
    _check_head_dim(query.shape[-1], "flash_attn")
    if not has_device():
        raise RDNAttentionError("no usable HIP device found")

    batch, heads, seq_len, head_dim = query.shape
    _, kv_heads, key_seq_len, _ = key.shape
    cos_ptr, sin_ptr, has_rope = _rope_ptrs(
        rot_cos, rot_sin, query, seq_len, key_seq_len, head_dim)
    qs, ks, vs = query.stride(), key.stride(), value.stride()
    output = torch.empty_like(query)
    os_ = output.stride()
    scale = head_dim ** -0.5

    result = _lib_handle().rdna_attention_forward(
        query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
        cos_ptr, sin_ptr,
        qs[0], qs[1], qs[2],
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        os_[0], os_[1], os_[2],
        batch, heads, kv_heads, seq_len, key_seq_len, head_dim,
        scale, 1 if is_causal else 0, has_rope, window_size,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != 0:
        raise RDNAttentionError(f"flash_attn failed: {_lib_handle().rdna_get_error().decode()}")
    return output


def flash_attn_int8qk(query, key, value, q_scale: float, k_scale: float, v_scale: float,
                       is_causal: bool = False, window_size: int = -1):
    """INT8 attention, both GEMMs. query/key/value: contiguous int8
    [batch, heads, seq, head_dim] on a HIP device, head_dim in
    SUPPORTED_HEAD_DIMS_INT8. Output is float16. Causal, sliding window, GQA
    and cross-attention all supported; no RoPE.

    q_scale/k_scale/v_scale are symmetric dequant scales, no zero-point:
    int8_value * scale ~= original_fp_value. q_scale and k_scale also accept a
    float32 tensor of per-token scales, (batch, heads, seq_len) for Q and
    (batch, kv_heads, key_seq_len) for K.
    """
    import torch

    _validate_inputs(query, key, value, torch.int8)
    heads, kv_heads_ = query.shape[1], key.shape[1]
    if kv_heads_ == 0 or heads % kv_heads_ != 0:
        raise RDNAttentionError(
            f"flash_attn_int8qk: query heads ({heads}) must be a positive multiple of "
            f"key/value heads ({kv_heads_})")
    _check_head_dim(query.shape[-1], "flash_attn_int8qk", allowed=SUPPORTED_HEAD_DIMS_INT8)
    if not has_device():
        raise RDNAttentionError("no usable HIP device found")

    batch, heads, seq_len, head_dim = query.shape
    _, kv_heads, key_seq_len, _ = key.shape

    def _scale_arg(x, name, want):
        if not isinstance(x, torch.Tensor):
            return float(x), None
        t = x.to(dtype=torch.float32, device=query.device).contiguous()
        if tuple(t.shape) != want:
            raise RDNAttentionError(
                f"flash_attn_int8qk: {name} tensor must be {want}, got {tuple(t.shape)}")
        return 1.0, t

    q_scalar, q_vec = _scale_arg(q_scale, "q_scale", (batch, heads, seq_len))
    k_scalar, k_vec = _scale_arg(k_scale, "k_scale", (batch, kv_heads, key_seq_len))
    q_vec_ptr = q_vec.data_ptr() if q_vec is not None else None
    k_vec_ptr = k_vec.data_ptr() if k_vec is not None else None

    qs, ks, vs = query.stride(), key.stride(), value.stride()
    output = torch.empty(query.shape, dtype=torch.float16, device=query.device)
    os_ = output.stride()
    scale = head_dim ** -0.5

    result = _lib_handle().rdna_attention_forward_int8qk(
        query.data_ptr(), key.data_ptr(), value.data_ptr(), output.data_ptr(),
        qs[0], qs[1], qs[2],
        ks[0], ks[1], ks[2],
        vs[0], vs[1], vs[2],
        os_[0], os_[1], os_[2],
        batch, heads, kv_heads, seq_len, key_seq_len, head_dim,
        scale, q_scalar, k_scalar, v_scale, q_vec_ptr, k_vec_ptr,
        1 if is_causal else 0, window_size,
        torch.cuda.current_stream().cuda_stream,
    )
    if result != 0:
        raise RDNAttentionError(f"flash_attn_int8qk failed: {_lib_handle().rdna_get_error().decode()}")
    return output


def quantize_int8_perchannel_v(query, key, value):
    """Quantize fp Q/K/V for flash_attn_int8qk() with a per-channel V scale,
    returning (qi, ki, vi, q_scale, k_scale, v_scale, v_channel).

    Attention is linear in V, so a scale that depends only on the head_dim
    column factors out of the sum: the caller multiplies the kernel output by
    v_channel to undo it exactly. v_scale is therefore 1.0.
    """
    import torch

    def per_tensor(x):
        s = x.abs().max().item() / 127.0
        s = s if s != 0.0 else 1.0
        return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8), s

    qi, q_scale = per_tensor(query.float())
    ki, k_scale = per_tensor(key.float())
    v = value.float()
    v_channel = v.abs().amax(dim=-2, keepdim=True) / 127.0
    v_channel = torch.where(v_channel == 0, torch.ones_like(v_channel), v_channel)
    vi = torch.clamp(torch.round(v / v_channel), -127, 127).to(torch.int8)
    return qi, ki, vi, q_scale, k_scale, 1.0, v_channel.squeeze(-2)


def quantize_int8_qk_pertoken(query, key):
    """Quantize Q and K with one scale per (batch, head, row), returning
    (qi, ki, q_scale, k_scale). The scales are float32 tensors shaped
    (batch, heads, seq) and (batch, kv_heads, key_seq), ready to hand to
    flash_attn_int8qk().
    """
    import torch

    def per_token(x):
        x = x.float()
        s = x.abs().amax(dim=-1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.clamp(torch.round(x / s), -127, 127).to(torch.int8), s.squeeze(-1).contiguous()

    qi, q_scale = per_token(query)
    ki, k_scale = per_token(key)
    return qi, ki, q_scale, k_scale


def flash_attn_int8qk_quantized(query, key, value, is_causal: bool = False,
                                window_size: int = -1):
    """INT8 attention taking ordinary fp16/fp32 Q/K/V - the recommended entry
    point. Applies both refinements the kernel supports: per-token Q/K scales
    in-kernel, and an exact per-channel V rescale host-side. Same shape and
    dtype contract as flash_attn().
    """
    _, _, vi, _, _, v_scale, v_channel = quantize_int8_perchannel_v(query, key, value)
    qi, ki, q_scale, k_scale = quantize_int8_qk_pertoken(query, key)
    out = flash_attn_int8qk(qi.contiguous(), ki.contiguous(), vi.contiguous(),
                            q_scale, k_scale, v_scale, is_causal, window_size)
    return (out.float() * v_channel.unsqueeze(-2)).to(out.dtype)


__all__ = ["flash_attn", "flash_attn_int8qk", "flash_attn_int8qk_quantized",
           "quantize_int8_perchannel_v", "quantize_int8_qk_pertoken",
           "has_device", "RDNAttentionError", "SUPPORTED_HEAD_DIMS",
           "SUPPORTED_HEAD_DIMS_INT8", "__version__"]
