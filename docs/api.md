# Python API

`rdnattention` exposes forward-only attention kernels for AMD RDNA GPUs. The
package is a thin ctypes wrapper: it reads the pointers out of your torch
tensors and calls the HIP kernel directly, with no staging copies and no
separate GPU context.

## Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| OS | Linux or Windows |
| GPU | RDNA2 or newer, or any target with native `v_dot2_f32_f16` |
| torch | a ROCm build, imported before use |

torch is not declared as a dependency because the ROCm builds do not come from
PyPI. Install it from AMD's index first, then install this package.

## Quick start

```python
import torch
from rdnattention import flash_attn

q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")

out = flash_attn(q, k, v, is_causal=True)
```

Under ROCm, torch still spells the device `"cuda"`.

## Tensor contract

Every entry point takes the same shape. They differ in how strided a tensor
they accept.

| Rule | Value |
|---|---|
| Shape | `[batch, heads, seq_len, head_dim]` |
| Layout | `flash_attn`: `head_dim` contiguous, batch/head/seq strided as you like. Every other entry point: fully contiguous (call `.contiguous()` first). |
| Device | a HIP device |
| Returns | a new tensor, same shape as `query` - and for `flash_attn`, its layout |

Key and value carry their own head and sequence counts, so grouped-query and
cross-attention need no extra arguments:

* **GQA / MQA.** Give key and value fewer heads than query. The query head
  count must be an exact multiple of the key head count.
* **Cross-attention and decode.** Give key and value a different sequence
  length than query.

There is no backward pass. Nothing here participates in autograd.

## Functions

### `flash_attn(query, key, value, is_causal=False, window_size=-1, rot_cos=None, rot_sin=None, null_keys=0)`

FlashAttention-2 style forward attention in fp16.

| Argument | Type |
|---|---|
| `query`, `key`, `value` | `float16`, head_dim in `SUPPORTED_HEAD_DIMS` |
| `is_causal` | `bool` |
| `window_size` | `int`, negative disables |
| `rot_cos`, `rot_sin` | `float32` angle tables, or `None` |
| `null_keys` | `int`, count of all-zero keys the caller dropped |

Returns `float16`.

`null_keys` lets a caller drop all-zero keys instead of passing them. A zero
key scores 0 against every query, so it takes softmax weight while
contributing no value - dropping it outright changes the result, but declaring
how many were dropped does not, because the kernel returns exactly that weight
to the denominator. Zero-padded cross-attention context is the case this
exists for: a 512-key context holding 102 real keys runs 3.35x faster passed
as 102 keys with `null_keys=410`. The compensation is exact - it reproduces an
fp64 reference to 9.4e-16 - but the two paths reach the denominator by
different accumulations, so in fp16 they agree to rounding (measured 2.4e-04
on real captures) rather than bit-for-bit. Rejected
alongside `is_causal` and `window_size`, which mask by key position - a
position the dropped keys no longer have.

Pass `rot_cos` and `rot_sin` together to apply rotary embeddings to Q and K
inside the kernel, with no separate pass over the tensors. Both are contiguous
`float32` of shape `[max(seq_len, key_seq_len), head_dim // 2]`, on the same
device, shared across batch and heads. Passing one without the other is an
error.

Rows are indexed by position within the tensor, starting at row 0. For a
decode step, row 0 must hold the angles for the query's own position.

```python
import torch
from rdnattention import flash_attn

pos = torch.arange(seq_len, device="cuda").unsqueeze(1).float()
inv = 10000.0 ** (-torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
ang = pos * inv
cos, sin = ang.cos().contiguous(), ang.sin().contiguous()

out = flash_attn(q, k, v, is_causal=True, rot_cos=cos, rot_sin=sin)
```

The rotation pairs adjacent channels, so channel `2i` and `2i + 1` rotate
together by the angle in column `i`.

### `flash_attn_int8qk(query, key, value, q_scale, k_scale, v_scale, is_causal=False, window_size=-1)`

Both matmuls run in INT8. Measured at 1.3x to 1.7x the throughput of the fp16
kernel depending on shape, at a measurable accuracy cost.

| Argument | Type |
|---|---|
| `query`, `key`, `value` | `int8`, head_dim in `SUPPORTED_HEAD_DIMS_INT8` |
| `q_scale`, `k_scale` | `float`, or a `float32` tensor of per-token scales |
| `v_scale` | `float` |

Returns `float16`.

Scales are symmetric with no zero point, so `int8_value * scale` recovers the
original value. Pass per-token scales as `float32` tensors shaped
`(batch, heads, seq_len)` for Q and `(batch, kv_heads, key_seq_len)` for K.
V takes a single float only.

### `flash_attn_int8qk_quantized(query, key, value, is_causal=False, window_size=-1)`

The recommended INT8 entry point. Takes ordinary `float16` or `float32` input,
quantizes it, and applies both refinements the kernel supports: per-token
scales for Q and K, and a per-channel scale for V. Returns `float16`.

Prefer this over calling `flash_attn_int8qk` with per-tensor scales. The gain
is small on well-behaved input and large on real activations, where a single
per-tensor scale is set by a handful of outliers and wastes most of the range.

```python
from rdnattention import flash_attn_int8qk_quantized

out = flash_attn_int8qk_quantized(q, k, v)   # q, k, v are fp16
```

### `quantize_int8_perchannel_v(query, key, value)`

Returns `(qi, ki, vi, q_scale, k_scale, v_scale, v_channel)`.

Q and K get one scale each. V gets one scale per head_dim column, returned in
`v_channel` with shape `(batch, heads, head_dim)`. Attention is linear in V, so
multiplying the kernel output by `v_channel` undoes that scaling exactly.
`v_scale` is therefore `1.0`.

Use this only if you want to drive `flash_attn_int8qk` yourself.
`flash_attn_int8qk_quantized` already does it.

### `quantize_int8_qk_pertoken(query, key)`

Returns `(qi, ki, q_scale, k_scale)`, where the scales are `float32` tensors of
one value per row, ready to pass straight to `flash_attn_int8qk`.

### `has_device()`

`True` if a usable HIP device is present.

## Masking

`is_causal` and `window_size` are the only masking controls. There is no
argument for an arbitrary attention mask.

When query and key sequence lengths differ, causal masking aligns them to the
end of the key sequence. With `delta = key_seq_len - seq_len`, key `k` is
visible to query `q` when `k <= q + delta`. A single-token decode step against
a full cache therefore sees the entire cache.

`window_size` is interpreted differently depending on `is_causal`:

| `is_causal` | Effect of `window_size = w` |
|---|---|
| `True` | each query sees the `w` keys ending at its own position |
| `False` | each query sees a symmetric band, `w // 2` keys either side |

## Constants

| Name | Value |
|---|---|
| `SUPPORTED_HEAD_DIMS` | every multiple of 32 from 32 to 512 |
| `SUPPORTED_HEAD_DIMS_INT8` | `(64, 128)` |
| `__version__` | package version string |

Only head_dim 64 and 128 have tuned tile sizes. Other sizes are correct but
not tuned. Where you can choose, prefer a multiple of 64: at multiples of 32
that are not multiples of 64 the kernel loses an LDS read optimisation, worth
36 to 49 percent at the sizes where it applies.

## Errors

Everything raises `RDNAttentionError`, except argument checks that raise the
usual `TypeError` and `ValueError`.

```python
from rdnattention import RDNAttentionError

try:
    out = flash_attn(q, k, v)
except RDNAttentionError as e:
    ...
```

Common causes: no HIP device, an unsupported head_dim, a layout the entry point
cannot stride over, a dtype other than the one that entry point takes, or a
query head count that is not a multiple of the key head count.

## Limits

* Forward pass only.
* The softmax scale is fixed at `1 / sqrt(head_dim)` and cannot be overridden.
* No arbitrary attention mask.
* Decode shapes (`seq_len = 1` against a long cache) work but are slow. The
  kernel does not yet split work across the key axis.
* INT8 covers head_dim 64 and 128 only, and has no RoPE.

## Environment variables

| Name | Effect |
|---|---|
| `RDNATTENTION_LIB` | load this library file instead of searching |
| `HIP_PATH` | on Windows, used to locate the HIP runtime if the first load fails |

In a source checkout the build output in `build/` is preferred over a packaged
copy in `rdnattention/lib/`, so a stale packaged library cannot shadow a
rebuild.
