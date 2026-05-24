#!/usr/bin/env python3
"""Cross-architecture anchor survival table — all available models."""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.abspath(__file__)))
import json, os

ANCHORS = [11, 12, 25, 279, 334, 382, 476, 481, 510, 715, 758]

# (label, family, params_B, hidden, n_layer, path)
SOURCES = [
    ("qwen2.5:3b",       "qwen2",          3.1,  2048, 36,
     "<cloud-server>/<path>"),
    ("qwen2.5:7b",       "qwen2",          7.6,  3584, 28,
     "<cloud-server>/<path>"),
    ("deepseek-r1:7b",   "qwen2-distill",  7.6,  3584, 28,
     "<cloud-server>/<path>"),
    ("qwen2.5:32b",      "qwen2",          32.8, 5120, 64,
     "REPO_ROOT/data/legacy-state-field/heat-summary-qwen32b_1779389549.json"),
    ("qwen2.5-coder:32b","qwen2",          32.8, 5120, 64,
     "REPO_ROOT/data/legacy-state-field/heat-summary-coder32b_1779389725.json"),
    ("deepseek-r1:32b",  "qwen2-distill",  32.8, 5120, 64,
     "REPO_ROOT/data/legacy-state-field/heat-summary-ds32b_1779390753.json"),
    ("llama3.1:8b",      "llama",          8.0,  4096, 32,
     "<cloud-server>/<path>"),
    ("llama3.3:70b",     "llama",          70.0, 8192, 80,
     "<cloud-server>/<path>"),
    ("phi3:medium",      "phi3",           14.0, 5120, 40,
     "<cloud-server>/<path>"),
    ("mistral-small3.2", "mistral-SWA",    24.0, 5120, 40,
     "<cloud-server>/<path>"),
]

def dims_ordered(path):
    d = json.load(open(path))
    raw = d.get("hottest") or d.get("hottest_top200") or d.get("hottest_top100") or []
    dims = []
    for e in raw:
        dim = e["dim"] if isinstance(e, dict) else e
        if dim not in dims:
            dims.append(dim)
    return dims, len(raw)

rows = []
per_dim = {a: {} for a in ANCHORS}

for label, family, P, H, NL, path in SOURCES:
    if not os.path.exists(path):
        rows.append((label, family, P, H, NL, 0, None, "MISSING"))
        continue
    dims, n_hot = dims_ordered(path)
    row = {}
    for K in (50, 100, 200):
        topk = dims[:K]
        hits = [a for a in ANCHORS if a in topk]
        exp = K * 11.0 / H
        ratio = len(hits) / exp if exp else 0
        row[K] = (len(hits), hits, exp, ratio)
    rows.append((label, family, P, H, NL, n_hot, row, None))
    for a in ANCHORS:
        per_dim[a][label] = (dims.index(a) + 1) if a in dims else None

# Print master table
print("=" * 120)
print("  CROSS-ARCHITECTURE ANCHOR SURVIVAL — Mercury master table (5 families, 10 models)")
print("=" * 120)
print(f"{'model':22s} {'family':16s} {'P(B)':>6s} {'H':>5s} {'L':>3s} {'n_hot':>6s}  "
      f"{'top-50':>14s} {'top-100':>14s} {'top-200':>14s}")
print("-" * 120)
for label, family, P, H, NL, n_hot, row, err in rows:
    if err:
        print(f"{label:22s} {family:16s} {P:>6.1f} {H:>5d} {NL:>3d} {err}")
        continue
    def cell(K):
        n, hits, exp, r = row[K]
        return f"{n}/11 ({r:>4.1f}x)"
    print(f"{label:22s} {family:16s} {P:>6.1f} {H:>5d} {NL:>3d} {n_hot:>6d}  "
          f"{cell(50):>14s} {cell(100):>14s} {cell(200):>14s}")

print()
print("=" * 120)
print("  PER-DIM SURVIVAL — rank in each model's hottest-dim ordering (· = not in observed top)")
print("=" * 120)
labels = [r[0] for r in rows if r[7] is None]
hdr = f"{'dim':<5} " + " ".join(f"{l[:10]:>10s}" for l in labels) + "  surv≤50"
print(hdr)
print("-" * len(hdr))
for a in ANCHORS:
    line = f"{a:<5} "
    cnt50 = 0
    for l in labels:
        rk = per_dim[a].get(l)
        if rk is not None and rk <= 50:
            cnt50 += 1
        line += f" {('·' if rk is None else str(rk)):>10s}"
    line += f"   {cnt50}/{len(labels)}"
    print(line)

print()
print("=" * 120)
print("  FAMILY-LEVEL SUMMARY")
print("=" * 120)
fams = {}
for label, family, P, H, NL, n_hot, row, err in rows:
    if err: continue
    fams.setdefault(family, []).append((label, row))

for fam, members in fams.items():
    avg_top50 = sum(m[1][50][0] for m in members) / len(members)
    avg_ratio = sum(m[1][50][3] for m in members) / len(members)
    print(f"\n[{fam}]  N={len(members)}  avg top-50 hits: {avg_top50:.1f}/11   avg ratio: {avg_ratio:.1f}x")
    for a in ANCHORS:
        hits = [m[0] for m in members if (per_dim[a].get(m[0]) or 999) <= 50]
        if len(hits) == len(members):
            mark = "⭐ ALL"
        elif len(hits) >= len(members) * 0.6:
            mark = "✓"
        elif len(hits) == 0:
            continue
        else:
            mark = " "
        print(f"  dim {a:<4} {mark}  {len(hits)}/{len(members)}")
