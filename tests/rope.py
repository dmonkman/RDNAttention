import ctypes
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

def load(path):
    lib = ctypes.CDLL(path)
    f = lib.rdna_attention_forward
    f.restype = ctypes.c_int32
    f.argtypes = ([ctypes.c_void_p]*6 + [ctypes.c_int64]*12 + [ctypes.c_uint32]*6 +
                  [ctypes.c_float] + [ctypes.c_int32]*3 + [ctypes.c_void_p])
    return lib, f

def run(f, q, k, v, cos, sin, causal):
    b, h, s, d = q.shape
    o = torch.empty_like(q)
    rc = f(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
           ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(o.data_ptr()),
           ctypes.c_void_p(cos.data_ptr()), ctypes.c_void_p(sin.data_ptr()),
           q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
           v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
           b, h, k.shape[1], s, k.shape[2], d, d ** -0.5, 1 if causal else 0, 1, -1, None)
    assert rc == 0, rc
    return o

def rope(x, cos, sin):
    """Interleaved pairs: (x0,x1) rotated by angle[d/2]. Matches the kernel."""
    y = x.double().clone()
    c = cos.double()[: x.shape[2]]
    s = sin.double()[: x.shape[2]]
    a, b = y[..., 0::2].clone(), y[..., 1::2].clone()
    y[..., 0::2] = a * c - b * s
    y[..., 1::2] = a * s + b * c
    return y

def reference(q, k, v, cos, sin, causal):
    B, H, S, D = q.shape
    KH, KS = k.shape[1], k.shape[2]
    g = H // KH
    qr, kr = rope(q, cos, sin), rope(k, cos, sin)
    out = torch.empty_like(q, dtype=torch.float64)
    delta = KS - S
    for b in range(B):
        for h in range(H):
            sc = (qr[b, h] @ kr[b, h // g].T) * (D ** -0.5)
            if causal:
                qi = torch.arange(S, device=q.device).unsqueeze(1)
                ki = torch.arange(KS, device=q.device).unsqueeze(0)
                sc = sc.masked_fill(ki > qi + delta, float("-inf"))
            out[b, h] = sc.softmax(-1) @ v[b, h // g].double()
    return out

if len(sys.argv) > 2:
    libs = [load(sys.argv[1])[1], load(sys.argv[2])[1]]
else:
    default = ROOT / "build" / ("rdnattention.dll" if sys.platform == "win32"
                                else "librdnattention.so")
    path = sys.argv[1] if len(sys.argv) > 1 else str(default)
    libs = [load(path)[1]]
    print(f"lib: {path}")
print(f"{'case':<26}{'rel_rms vs fp64':<18}{'A vs B' if len(libs) > 1 else ''}")
bad = 0
for name, B, H, KH, S, KS, D, causal in [
        ("MHA d64 rope", 1, 8, 8, 256, 256, 64, False),
        ("MHA d128 rope", 2, 8, 8, 512, 512, 128, False),
        ("causal d64 rope", 1, 8, 8, 256, 256, 64, True),
        ("causal d128 rope", 1, 8, 8, 384, 384, 128, True),
        ("GQA d64 rope", 1, 8, 2, 256, 256, 64, False),
        ("cross-attn d64 rope", 1, 8, 8, 128, 512, 64, False),
        ("tail 300 rope", 1, 8, 8, 300, 300, 64, False)]:
    torch.manual_seed(7)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, KH, KS, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, KH, KS, D, device="cuda", dtype=torch.float16)
    pos = torch.arange(max(S, KS), device="cuda").unsqueeze(1).float()
    inv = (10000.0 ** (-torch.arange(0, D, 2, device="cuda").float() / D))
    ang = pos * inv
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    ref = reference(q, k, v, cos, sin, causal)
    outs = [run(f, q, k, v, cos, sin, causal) for f in libs]
    def rr(x): return (((x.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt()).item()
    err = rr(outs[0])
    note = ""
    if len(outs) > 1:
        same = torch.equal(outs[0], outs[1])
        note = "BIT-IDENTICAL" if same else "DIFFERS"
        if not same:
            bad += 1
    # fp16 attention with rope lands around 4e-4; an order of magnitude over
    # that means the rotation itself is wrong, not accumulated rounding.
    if err > 1e-2:
        bad += 1
        note += " ERROR"
    print(f"{name:<26}{err:<18.3e}{note}")
print("\n" + ("ROPE OK" if bad == 0 else "ROPE FAILURES"))
