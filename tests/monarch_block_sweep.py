"""MonarchAttention approximation error per transformer block.

Answers "which layers can this backend be used on", which is the question the
ComfyUI node's monarch_depth slider exists to act on. Every captured block is
measured against exact attention, across every aligned Monarch factorization
its token grid admits.

Needs captures in tests/.data/captured_qkv (see tests/accuracy.py's own notes on
producing them) - one file per block, which is the whole point: a single block
says nothing, because the error varies enormously with depth.

    python tests/monarch_block_sweep.py [--model anima] [--csv out.csv]
"""
import argparse
import re
import sys
from pathlib import Path

RDNATTENTION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RDNATTENTION_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
except ImportError:
    print("This script needs a ROCm PyTorch install. Run it from an environment that has one.")
    sys.exit(1)

import rdnattention
from accuracy import CAPTURED_QKV_DIR, aligned_monarch_configs, detect_grid, monarch_permutation

# The reference is fp32 SDPA on the GPU, not the fp64 CPU oracle the gates use:
# at capture shapes that oracle is hours of scalar work, and Monarch's error is
# ~1e-1 while fp32-vs-fp64 is ~1e-7, so the choice cannot move these numbers.
REFERENCE_DTYPE = torch.float32


def rel_rms(got, ref):
    diff = (got.float() - ref.float()).pow(2).sum().sqrt()
    den = ref.float().pow(2).sum().sqrt()
    return (diff / den).item() if den > 0 else diff.item()


def exact_reference(q, k, v):
    qf, kf, vf = (t.to(REFERENCE_DTYPE) for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(qf, kf, vf)


def parse_layer(stem):
    """(block_index, stage) from a capture filename. SDXL embeds the UNet level
    in the name (input4b0); DiTs carry a bare block index (b27)."""
    m = re.search(r"_([a-z]+\d+)b(\d+)_", stem)
    if m:
        return int(m.group(2)), m.group(1)
    m = re.search(r"_b(\d+)_", stem)
    if m:
        return int(m.group(1)), None
    m = re.search(r"_(\d+)_H\d+", stem)
    if m:
        return int(m.group(1)), None
    return None, None


def load_self_attention_captures(root, model_filter):
    """Self-attention captures only - Monarch has no Nq/Nk distinction, so
    cross-attention is not a thing it can approximate."""
    out = []
    for f in sorted(Path(root).rglob("*.pt")):
        model = f.parent.name
        if model_filter and model_filter != model:
            continue
        d = torch.load(f, map_location="cpu")
        q, k, v = d["q"], d["k"], d["v"]
        if q.shape[-2] != k.shape[-2]:
            continue
        block = d.get("block_index")
        stage = d.get("stage")
        if block is None:
            block, stage = parse_layer(f.stem)
        grid = d.get("grid")
        if grid is not None:
            grid = tuple(int(x) for x in grid)
            if grid[0] * grid[1] * grid[2] != q.shape[-2]:
                grid = None
        out.append({"model": model, "file": f.stem, "block": block, "stage": stage,
                    "q": q, "k": k, "v": v, "grid": grid,
                    "orig_seq": (d.get("orig_shape_q") or [None] * 3)[-2]})
    out.sort(key=lambda r: (r["model"], r["block"] if r["block"] is not None else 1 << 30))
    return out


def build_grid_cache(recs):
    """Detect once per (model, seq) using the strongest-scoring block, then
    share it. The grid belongs to the image layout, not to any one block, and
    shallow blocks routinely carry too little positional structure to detect -
    so a per-capture detection in file order would strand the early blocks."""
    cache, notes = {}, {}
    for rec in recs:
        if rec["grid"] is not None:
            continue
        key = (rec["model"], rec["q"].shape[-2])
        grid, note, score = detect_grid(rec["q"].to("cuda", torch.float16),
                                        rec["k"].to("cuda", torch.float16))
        torch.cuda.empty_cache()
        if grid is not None and score > cache.get(key, (None, -1.0))[1]:
            cache[key] = (grid, score)
            notes[key] = f"block {rec['block']}: {note}"
    return {k: v[0] for k, v in cache.items()}, notes


def resolve_grid(rec, cache, notes):
    if rec["grid"] is not None:
        return rec["grid"], "recorded"
    key = (rec["model"], rec["q"].shape[-2])
    if key in cache:
        return cache[key], f"detected, {notes[key]}"
    return None, "undetectable"


def candidate_grids(n, recorded=None):
    """Plausible 2D grids for a sequence length, widest-first. Used to check a
    detected grid rather than trust it: the correct layout should score
    distinctly better than a wrong one, since alignment is what Monarch needs."""
    out = []
    if recorded is not None:
        out.append(recorded)
    for w in sorted((w for w in range(8, n // 2 + 1) if n % w == 0), reverse=True):
        g = (1, n // w, w)
        if g not in out and min(g[1], w) >= 8:
            out.append(g)
    return out


def eval_monarch(q, k, v, grid, order, b):
    """(rel_rms, err_over_v) for one aligned factorization.

    Two metrics because they disagree, and the disagreement is the point.
    rel_rms divides by the attention output's own norm, which collapses on a
    near-uniform block (the output is then ~mean(V), and V is roughly
    zero-mean) - so a block that barely moves the residual stream can post a
    huge rel_rms. err_over_v divides by V's RMS instead, a stand-in for the
    scale this output is added into downstream."""
    ref = exact_reference(q, k, v)
    if order != "fhw":
        idx = monarch_permutation(grid, order).to(q.device)
        qp, kp, vp = (t[:, :, idx].contiguous() for t in (q, k, v))
        got = rdnattention.flash_attn_monarch(qp, kp, vp, block_b=b)
        got = got[:, :, torch.argsort(idx)]
    else:
        got = rdnattention.flash_attn_monarch(q.contiguous(), k.contiguous(),
                                              v.contiguous(), block_b=b)
    err_abs = (got.float() - ref).pow(2).mean().sqrt()
    out_rms = ref.pow(2).mean().sqrt()
    v_rms = v.float().pow(2).mean().sqrt()
    res = ((err_abs / out_rms).item() if out_rms > 0 else float("nan"),
           (err_abs / v_rms).item() if v_rms > 0 else float("nan"))
    del ref, got
    torch.cuda.empty_cache()
    return res


def probe_grid_fit(recs, cache):
    """Rank candidate grids by the Monarch error they actually produce. The
    detected grid is a hypothesis; this is the experiment that tests it."""
    by_model = {}
    for rec in recs:
        by_model.setdefault((rec["model"], rec["q"].shape[-2]), []).append(rec)
    for key, sub in by_model.items():
        model, n = key
        # The deepest block is the most positionally structured, so it
        # discriminates between grids best.
        rec = max(sub, key=lambda r: r["block"] if r["block"] is not None else -1)
        if rec["q"].shape[-1] not in rdnattention.SUPPORTED_HEAD_DIMS_MONARCH:
            continue
        q = rec["q"].to("cuda", torch.float16).contiguous()
        k = rec["k"].to("cuda", torch.float16).contiguous()
        v = rec["v"].to("cuda", torch.float16).contiguous()
        detected = cache.get(key)
        print(f"\n=== {model} seq={n}: which grid does Monarch actually fit best? "
              f"(block {rec['block']}) ===")
        print(f"{'grid':>16} {'config':>10} {'b1':>6} {'b2':>6} {'rel_rms':>10}")
        print("-" * 52)
        scored = []
        for grid in candidate_grids(n, detected):
            best = None
            for label, order, b in aligned_monarch_configs(grid):
                try:
                    err = eval_monarch(q, k, v, grid, order, b)[0]
                except Exception:
                    continue
                err = err[0] if isinstance(err, tuple) else err
                if best is None or err < best[0]:
                    best = (err, label, n // b, b)
            if best is None:
                continue
            scored.append((best[0], grid, best[1], best[2], best[3]))
            mark = "  <- detected" if grid == detected else ""
            print(f"{str(grid):>16} {best[1]:>10} {best[2]:>6} {best[3]:>6} {best[0]:>10.4f}{mark}")
        del q, k, v
        torch.cuda.empty_cache()
        if scored:
            scored.sort()
            print(f"\n  best fit: {scored[0][1]} at {scored[0][0]:.4f}"
                  + (f" (detection said {detected})" if detected and scored[0][1] != detected else ""))


def depth_band(total, depth):
    return int(round(total * depth / 200.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CAPTURED_QKV_DIR))
    ap.add_argument("--model", default=None, help="only this capture subdirectory")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--probe-grids", action="store_true",
                    help="test every plausible 2D grid per model instead of trusting "
                         "detection - the right layout should score distinctly better")
    args = ap.parse_args()

    if not rdnattention.has_device():
        print("No usable HIP device found.")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    recs = load_self_attention_captures(args.data, args.model)
    if not recs:
        print(f"No self-attention captures under {args.data}.")
        return 1

    print("detecting token grids ...")
    cache, notes = build_grid_cache(recs)
    for (model, n), g in cache.items():
        print(f"  {model} seq={n}: (f,h,w)={g}  [{notes[(model, n)]}]")

    if args.probe_grids:
        probe_grid_fit(recs, cache)
        return 0

    rows = []
    for rec in recs:
        q = rec["q"].to("cuda", torch.float16).contiguous()
        k = rec["k"].to("cuda", torch.float16).contiguous()
        v = rec["v"].to("cuda", torch.float16).contiguous()
        n, d = q.shape[-2], q.shape[-1]
        grid, how = resolve_grid(rec, cache, notes)

        if d not in rdnattention.SUPPORTED_HEAD_DIMS_MONARCH:
            rows.append({**rec, "n": n, "d": d, "grid": grid, "how": how,
                         "results": {}, "skip": f"head_dim {d} unsupported"})
        elif grid is None:
            rows.append({**rec, "n": n, "d": d, "grid": grid, "how": how,
                         "results": {}, "skip": "no grid"})
        else:
            results = {}
            for label, order, b in aligned_monarch_configs(grid):
                try:
                    e1, e2 = eval_monarch(q, k, v, grid, order, b)
                    results[label] = (e1, e2, b, n // b)
                except Exception as e:
                    results[label] = (float("nan"), float("nan"), b, n // b)
                    print(f"  [warn] {rec['file']} {label}: {type(e).__name__}: {e}")
            rows.append({**rec, "n": n, "d": d, "grid": grid, "how": how,
                         "results": results, "skip": None})
        del q, k, v
        torch.cuda.empty_cache()
        r = rows[-1]
        tag = f"{r['model']}/block {r['block']}"
        if r["skip"]:
            print(f"  {tag:28} skipped - {r['skip']}")
        else:
            best = min((e for e, _, _, _ in r["results"].values() if e == e), default=float("nan"))
            print(f"  {tag:28} N={n} grid={grid} ({how}) best rel_rms={best:.4f}")

    labels = []
    for r in rows:
        for lab in r["results"]:
            if lab not in labels:
                labels.append(lab)

    for model in dict.fromkeys(r["model"] for r in rows):
        sub = [r for r in rows if r["model"] == model]
        print(f"\n=== {model}: Monarch rel_rms vs exact attention, per block ===")
        head = (f"{'block':>7} {'N':>6} {'grid':>14}  " + "".join(f"{l:>12}" for l in labels)
                + f"{'best':>10}{'err/|V|':>10}")
        print(head)
        print("-" * len(head))
        for r in sub:
            cells = ""
            for lab in labels:
                hit = r["results"].get(lab)
                cells += f"{'-':>12}" if hit is None else f"{hit[0]:>12.4f}"
            vals = [v for v in r["results"].values() if v[0] == v[0]]
            best = min((v[0] for v in vals), default=None)
            bestv = min((v[1] for v in vals), default=None)
            blk = f"{r['block']}" + (f" {r['stage']}" if r["stage"] else "")
            bcell = f"{best:>10.4f}" if best is not None else f"{'skipped':>10}"
            vcell = f"{bestv:>10.4f}" if bestv is not None else f"{'-':>10}"
            print(f"{blk:>7} {r['n']:>6} {str(r['grid']):>14}  {cells}{bcell}{vcell}")

        measured = [r for r in sub if r["results"]]
        if len(measured) < 2:
            continue
        blocks = [r["block"] for r in measured]
        bests = {r["block"]: min(v[0] for v in r["results"].values() if v[0] == v[0])
                 for r in measured}
        bestv = {r["block"]: min(v[1] for v in r["results"].values() if v[1] == v[1])
                 for r in measured}
        worst_b = max(bests, key=lambda b: bests[b])
        best_b = min(bests, key=lambda b: bests[b])
        print(f"\n  best block  : {best_b} at {bests[best_b]:.4f}")
        print(f"  worst block : {worst_b} at {bests[worst_b]:.4f}"
              f"  ({bests[worst_b] / bests[best_b]:.1f}x spread across sampled depth)")
        wv = max(bestv, key=lambda b: bestv[b])
        bv = min(bestv, key=lambda b: bestv[b])
        print(f"  by err/|V|  : worst block {wv} at {bestv[wv]:.4f}, best block {bv} at {bestv[bv]:.4f}")
        if wv != worst_b:
            print(f"  NOTE        : the two metrics disagree on the worst block "
                  f"({worst_b} by rel_rms, {wv} by err/|V|) - see eval_monarch's docstring")
        print(f"  sampled     : blocks {blocks} - the curve between them is not measured")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "block", "stage", "seq", "head_dim", "grid", "grid_source",
                        "config", "b1", "b2", "rel_rms", "err_over_v"])
            for r in rows:
                for lab, (err, errv, b, m) in r["results"].items():
                    w.writerow([r["model"], r["block"], r["stage"] or "", r["n"], r["d"],
                                r["grid"], r["how"], lab, m, b, f"{err:.6f}", f"{errv:.6f}"])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
