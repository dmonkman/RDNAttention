"""Static checks on a built wheel: it carries exactly one native library, is
tagged for a real platform, and the library holds a code object for every
Tier 1 arch in CMakeLists.txt. Needs no GPU and no ROCm.

    python tests/wheel_check.py dist/rdnattention-*.whl
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB_NAMES = ("rdnattention/lib/librdnattention.so", "rdnattention/lib/rdnattention.dll")


def tier1_archs():
    text = (ROOT / "CMakeLists.txt").read_text()
    m = re.search(r"set\(RDNATTENTION_TIER1_ARCHS(.*?)\)", text, re.S)
    body = re.sub(r"#.*", "", m.group(1))
    return body.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", type=Path)
    args = ap.parse_args()

    bad = []
    tag = args.wheel.stem.split("-")[-1]
    if tag == "any":
        bad.append(f"platform tag is '{tag}' - the wheel ships a native library")

    with zipfile.ZipFile(args.wheel) as z:
        libs = [n for n in z.namelist() if n in LIB_NAMES]
        if len(libs) != 1:
            print(f"FAIL: expected one of {LIB_NAMES}, found {libs}")
            return 1
        data = z.read(libs[0])

    # Offload bundle entry IDs are plain strings unless the bundle is
    # compressed, in which case the arch list cannot be read this way.
    if b"CCOB" in data and b"__CLANG_OFFLOAD_BUNDLE__" not in data:
        bad.append("offload bundle is compressed; arch check cannot read it")
    found = {m.decode() for m in re.findall(rb"amdgcn-amd-amdhsa--(gfx\w+)", data)}
    missing = [a for a in tier1_archs() if a not in found]
    if missing:
        bad.append(f"no code object for {missing}")

    hip = sorted({m.decode() for m in re.findall(rb"(?:lib)?amdhip64(?:_\d+\.dll|\.so\.\d+)", data)})

    print(f"wheel:    {args.wheel.name}")
    print(f"library:  {libs[0]} ({len(data) / 1e6:.1f} MB)")
    print(f"archs:    {' '.join(sorted(found))}")
    print(f"links:    {' '.join(hip) or '?'}")
    if bad:
        print("FAILURES:")
        for b in bad:
            print("  " + b)
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
