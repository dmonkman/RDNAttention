"""Build one library per fdot2() lowering via -DRDNATTENTION_FORCE_DOT_PATH
and execute all four on the local GPU, against an fp64 reference and against
the native path's own output.

    python tests/dot_paths.py
    python tests/dot_paths.py --isa
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PATHS = {1: "builtin", 2: "portable", 3: "mixasm", 4: "cvtfma"}
LIB_NAME = "rdnattention.dll" if sys.platform == "win32" else "librdnattention.so"


# Child mode - its own process, since the library binds at import time.
def run_checks():
    import torch

    import rdnattention

    def reference(q, k, v, causal=False, window=-1):
        B, H, S, D = q.shape
        KH, KS = k.shape[1], k.shape[2]
        g = H // KH
        out = torch.empty_like(q, dtype=torch.float64)
        delta = KS - S
        for b in range(B):
            for h in range(H):
                s = (q[b, h].double() @ k[b, h // g].double().T) * (D ** -0.5)
                qi = torch.arange(S, device=q.device).unsqueeze(1)
                ki = torch.arange(KS, device=q.device).unsqueeze(0)
                if causal:
                    s = s.masked_fill(ki > qi + delta, float("-inf"))
                if window > 0:
                    s = s.masked_fill(ki <= qi + delta - window, float("-inf"))
                out[b, h] = s.softmax(-1) @ v[b, h // g].double()
        return out

    cases = [
        ("MHA d64",       1, 8, 8, 256, 256, 64, False, -1),
        ("MHA d128",      2, 8, 8, 512, 512, 128, False, -1),
        ("causal d64",    1, 8, 8, 256, 256, 64, True, -1),
        ("causal d128",   1, 8, 8, 384, 384, 128, True, -1),
        ("GQA g4 d64",    1, 8, 2, 256, 256, 64, False, -1),
        ("GQA g4 c d128", 1, 8, 2, 256, 256, 128, True, -1),
        ("cross-attn",    1, 8, 8, 128, 512, 64, False, -1),
        ("window=64",     1, 8, 8, 256, 256, 64, True, 64),
        ("tail 300",      1, 8, 8, 300, 300, 64, False, -1),
    ]
    worst = 0.0
    for name, B, H, KH, S, KS, D, causal, win in cases:
        torch.manual_seed(1234)
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, KH, KS, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, KH, KS, D, device="cuda", dtype=torch.float16)
        got = rdnattention.flash_attn(q, k, v, is_causal=causal, window_size=win)
        if not torch.isfinite(got).all():
            print(f"NONFINITE {name}")
            return 1
        ref = reference(q, k, v, causal, win)
        err = (((got.double() - ref) ** 2).mean().sqrt() / (ref ** 2).mean().sqrt()).item()
        worst = max(worst, err)
        print(f"CASE {name}|{err:.6e}|{got.double().sum().item():.10e}")
    print(f"WORST {worst:.6e}")
    return 0


def detect_arch():
    try:
        import torch

        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return None


def toolchain_flags():
    """Pin the same ROCm build/ uses - configuring against a different one
    than the runtime resolves gives a library that launches and writes zeros."""
    cache = ROOT / "build" / "CMakeCache.txt"
    if cache.exists():
        wanted = {"hip_DIR": None, "CMAKE_HIP_COMPILER": None}
        for line in cache.read_text(errors="ignore").splitlines():
            key = line.split(":", 1)[0]
            if key in wanted and "=" in line:
                wanted[key] = line.split("=", 1)[1].strip()
        if all(wanted.values()):
            return [f"-D{k}={v}" for k, v in wanted.items()]
    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        root = Path(hip_path)
        clang = "clang++.exe" if sys.platform == "win32" else "clang++"
        return [f"-Dhip_DIR={root / 'lib' / 'cmake' / 'hip'}",
                f"-DCMAKE_HIP_COMPILER={root / 'bin' / clang}"]
    return []


def no_dot_arches():
    """RDNATTENTION_NO_DOT_ARCHS, read from CMakeLists.txt so this does not
    become a third copy of the list (arch.hpp holds the second)."""
    text = (ROOT / "CMakeLists.txt").read_text(errors="ignore")
    m = re.search(r"set\(RDNATTENTION_NO_DOT_ARCHS(.*?)\)", text, re.S)
    return set(m.group(1).split()) if m else set()


def build(path_id, arch):
    bdir = ROOT / "build" / "_dotpaths" / f"path{path_id}"
    cmd = ["cmake", "-S", str(ROOT), "-B", str(bdir), "-G", "Ninja",
           f"-DCMAKE_HIP_ARCHITECTURES={arch}",
           f"-DRDNATTENTION_FORCE_DOT_PATH={path_id}"] + toolchain_flags()
    # The INT8 kernel is unconditional native sdot4; it won't build without dot.
    if arch in no_dot_arches():
        cmd.append("-DRDNATTENTION_ENABLE_INT8=OFF")
    for c in (cmd, ["cmake", "--build", str(bdir)]):
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  BUILD FAILED (path {path_id}):")
            print("   ", (r.stderr or r.stdout).strip().splitlines()[-6:])
            return None
    return bdir / LIB_NAME


def isa_mix(path_id, arch):
    hipcc = None
    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        cand = Path(hip_path) / "bin" / ("hipcc.exe" if sys.platform == "win32" else "hipcc")
        if cand.exists():
            hipcc = str(cand)
    hipcc = hipcc or shutil.which("hipcc") or shutil.which("hipcc.exe")
    if not hipcc:
        return "(hipcc not found)"
    out = ROOT / "build" / "_dotpaths" / f"path{path_id}.s"
    r = subprocess.run(
        [hipcc, "-x", "hip", f"--offload-arch={arch}", "-O3", "-ffast-math",
         "-std=c++17", f"-I{ROOT / 'src'}", f"-I{ROOT / 'src' / 'rdna'}",
         f"-DRDNA_FORCE_DOT_PATH={path_id}", "--cuda-device-only", "-S",
         "-o", str(out), str(ROOT / "src" / "rdna" / "fa2_forward_f16.hip")],
        capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        return "(isa dump failed)"
    text = out.read_text(errors="ignore")
    n = lambda pat: sum(1 for ln in text.splitlines() if pat in ln)  # noqa: E731
    return (f"VALU={sum(1 for ln in text.splitlines() if ln.strip().startswith('v_')):<6} "
            f"dot2={n('v_dot2c_f32_f16') + n('v_dot2_f32_f16'):<6} "
            f"mix={n('v_fma_mix_f32') + n('v_mad_mix_f32'):<6} "
            f"cvt={n('v_cvt_f32_f16'):<5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--isa", action="store_true", help="also report instruction mix")
    args = ap.parse_args()
    if args.run:
        return run_checks()

    arch = detect_arch()
    if not arch:
        print("No HIP device found (need torch with ROCm).")
        return 1
    print(f"GPU: {arch}\n")

    results = {}
    for pid, name in PATHS.items():
        print(f"--- path {pid} ({name}) ---")
        lib = build(pid, arch)
        if lib is None:
            results[pid] = None
            continue
        if args.isa:
            print(f"  isa: {isa_mix(pid, arch)}")
        env = dict(os.environ, RDNATTENTION_LIB=str(lib))
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--run"],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"  FAILED:\n{r.stdout}{r.stderr}")
            results[pid] = None
            continue
        cases, worst = {}, None
        for line in r.stdout.splitlines():
            if line.startswith("CASE "):
                nm, err, checksum = line[5:].split("|")
                cases[nm] = (float(err), float(checksum))
            elif line.startswith("WORST "):
                worst = float(line.split()[1])
        results[pid] = cases
        print(f"  {len(cases)} cases ok, worst rel_rms vs fp64 reference = {worst:.2e}")

    print("\n--- cross-path agreement vs path 1 (builtin) ---")
    base = results.get(1)
    if not base:
        print("  builtin path unavailable; nothing to compare against")
        return 1
    ok = True
    for pid in (2, 3, 4):
        if not results.get(pid):
            print(f"  path {pid} ({PATHS[pid]}): NOT BUILT / FAILED")
            ok = False
            continue
        worst_rel, worst_case = 0.0, ""
        for nm, (_, chk) in results[pid].items():
            b = base[nm][1]
            rel = abs(chk - b) / max(abs(b), 1e-30)
            if rel > worst_rel:
                worst_rel, worst_case = rel, nm
        verdict = "ok" if worst_rel < 1e-3 else "SUSPICIOUS"
        if worst_rel >= 1e-3:
            ok = False
        print(f"  path {pid} ({PATHS[pid]:<8}): max checksum drift {worst_rel:.2e} "
              f"({worst_case}) {verdict}")

    print("\n" + ("ALL PATHS PASS" if ok and all(results.values()) else "FAILURES ABOVE"))
    return 0 if ok and all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
