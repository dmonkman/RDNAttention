"""Every head_dim flash_attn() dispatches, through the Python path, against
an fp64 reference."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = [
    ("dense", 1, 300, 300, False, -1),
    ("causal", 1, 300, 300, True, -1),
    ("gqa", 4, 300, 300, False, -1),
    ("gqa_causal", 4, 300, 300, True, -1),
    ("window", 1, 300, 300, False, 64),
    ("causal_window", 1, 300, 300, True, 32),
    ("cross", 1, 200, 300, False, -1),
    ("decode", 1, 1, 300, False, -1),
    ("aligned", 1, 256, 256, True, -1),
]


def reference(q, k, v, causal, window):
    import torch

    B, H, S, D = q.shape
    KH, KS = k.shape[1], k.shape[2]
    g = H // KH
    delta = KS - S
    out = torch.empty_like(q, dtype=torch.float64)
    qi = torch.arange(S, device=q.device).unsqueeze(1)
    ki = torch.arange(KS, device=q.device).unsqueeze(0)
    for b in range(B):
        for h in range(H):
            s = (q[b, h].double() @ k[b, h // g].double().T) * (D ** -0.5)
            if causal:
                s = s.masked_fill(ki > qi + delta, float("-inf"))
            if window > 0:
                if causal:
                    s = s.masked_fill(qi + delta - ki >= window, float("-inf"))
                else:
                    s = s.masked_fill((qi + delta - ki).abs() > window // 2, float("-inf"))
            out[b, h] = s.softmax(-1) @ v[b, h // g].double()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=3e-3, help="max rel_rms vs the fp64 reference")
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    import torch

    import rdnattention

    if not rdnattention.has_device():
        print("no usable HIP device")
        return 1

    dims = rdnattention.SUPPORTED_HEAD_DIMS
    names = [v[0] for v in VARIANTS]
    print(f"{'head_dim':>8} " + " ".join(f"{n:>14}" for n in names))

    failures = []
    for d in dims:
        cells = []
        for name, kvdiv, s_len, ks_len, causal, window in VARIANTS:
            torch.manual_seed(1234)
            b, h = 1, args.heads
            kvh = max(1, h // kvdiv)
            q = torch.randn(b, h, s_len, d, device="cuda", dtype=torch.float16)
            k = torch.randn(b, kvh, ks_len, d, device="cuda", dtype=torch.float16)
            v = torch.randn(b, kvh, ks_len, d, device="cuda", dtype=torch.float16)
            got = rdnattention.flash_attn(q, k, v, is_causal=causal, window_size=window)
            if not torch.isfinite(got).all():
                failures.append((d, name, "non-finite output"))
                cells.append("NONFINITE")
                continue
            ref = reference(q, k, v, causal, window)
            err = (((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt()).item()
            if not err < args.tol:
                failures.append((d, name, f"rel_rms {err:.3e} >= {args.tol:.0e}"))
            cells.append(f"{err:.2e}" + ("" if err < args.tol else " !"))
        print(f"{d:>8} " + " ".join(f"{c:>14}" for c in cells))

    print(f"\n{len(dims) * len(VARIANTS)} cases, tolerance {args.tol:.0e}")
    if failures:
        print("FAILURES:")
        for d, name, why in failures:
            print(f"  head_dim {d} {name}: {why}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
