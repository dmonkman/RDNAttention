"""Per-kernel VGPR/LDS/spill audit for every head_dim forwardF16() dispatches,
checking chooseTile()'s empirical kTileVgprCap still holds. Needs only hipcc.

    python tests/head_dim_resources.py
    python tests/head_dim_resources.py --arch gfx1100
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_SPILL = 0
MIN_OCCUPANCY = 4

# Per-head_dim spill allowance. ROCm 10.0 spills d=416's 32x32 tile by 33-37
# VGPRs, yet it measured 24% faster than ROCm 7.2's spill-free build and 28%
# faster than the 16x32 tile a lower kTileVgprCap would pick (gfx1030,
# seq 2048) - the spills sit outside the hot loop. The cap still catches growth.
SPILL_ALLOWANCE = {416: 40}


def find_hipcc():
    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        cand = Path(hip_path) / "bin" / ("hipcc.exe" if sys.platform == "win32" else "hipcc")
        if cand.exists():
            return str(cand)
    return shutil.which("hipcc") or shutil.which("hipcc.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx1030")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hipcc = find_hipcc()
    if not hipcc:
        print("hipcc not found (set HIP_PATH)")
        return 1

    r = subprocess.run(
        [hipcc, "-x", "hip", f"--offload-arch={args.arch}", "-O3", "-ffast-math",
         "-std=c++17", f"-I{ROOT / 'src'}", f"-I{ROOT / 'src' / 'rdna'}",
         "--cuda-device-only", "-S", "-o", os.devnull,
         "-Rpass-analysis=kernel-resource-usage",
         str(ROOT / "src" / "rdna" / "fa2_forward_f16.hip")],
        capture_output=True, text=True)

    fields = ("Function Name", "VGPRs", "Occupancy [waves/SIMD]",
              "VGPRs Spill", "LDS Size [bytes/block]")
    kernels, cur = [], {}
    for line in r.stderr.splitlines():
        for f in fields:
            m = re.search(re.escape(f) + r":\s*(\S+)", line)
            if m:
                cur[f] = m.group(1)
        if len(cur) == len(fields):
            kernels.append(cur)
            cur = {}

    if not kernels:
        print("no kernels reported; hipcc output was:\n" + r.stderr[-2000:])
        return 1

    bad = []
    allowed = []
    seen = {}
    for k in kernels:
        name = k["Function Name"]
        m = re.search(r"ILi(\d+)ELi(\d+)ELi(\d+)E", name)
        if not m:
            continue
        head_dim, br, bc = (int(x) for x in m.groups())
        spill, occ, lds = int(k["VGPRs Spill"]), int(k["Occupancy [waves/SIMD]"]), int(k["LDS Size [bytes/block]"])
        if spill > SPILL_ALLOWANCE.get(head_dim, MAX_SPILL):
            bad.append(f"{name}: {spill} VGPRs spilled")
        elif spill:
            allowed.append(f"{name}: {spill} VGPRs spilled (allowed up to {SPILL_ALLOWANCE[head_dim]})")
        if occ < MIN_OCCUPANCY:
            bad.append(f"{name}: occupancy {occ} < {MIN_OCCUPANCY}")
        if lds > 65536:
            bad.append(f"{name}: LDS {lds} over budget")
        seen.setdefault((head_dim, br, bc), []).append((int(k["VGPRs"]), occ, spill, lds))

    print(f"{'head_dim':>9} {'Br':>4} {'Bc':>4} {'VGPRs':>7} {'occ':>4} {'spill':>6} {'LDS':>7}")
    for (hd, br, bc) in sorted(seen):
        v = seen[(hd, br, bc)]
        print(f"{hd:>9} {br:>4} {bc:>4} {max(x[0] for x in v):>7} "
              f"{min(x[1] for x in v):>4} {max(x[2] for x in v):>6} {v[0][3]:>7}")

    print(f"\n{len(kernels)} kernels checked on {args.arch}")
    if allowed:
        print("ALLOWED SPILLS:")
        for a in allowed:
            print("  " + a)
    if bad:
        print("FAILURES:")
        for b in bad:
            print("  " + b)
        return 1
    print("ALL OK (spills within allowance, occupancy and LDS within budget)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
