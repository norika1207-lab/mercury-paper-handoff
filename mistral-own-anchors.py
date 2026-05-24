#!/usr/bin/env python3
"""Discover mistral's OWN anchor structure — not measured against qwen."""
import json
from collections import Counter, defaultdict

QWEN_ANCHORS = [11, 12, 25, 279, 334, 382, 476, 481, 510, 715, 758]

MODELS = {
    "qwen2.5:7b":       "<cloud-server>/<path>",
    "llama3.1:8b":      "<cloud-server>/<path>",
    "llama3.3:70b":     "<cloud-server>/<path>",
    "phi3:medium":      "<cloud-server>/<path>",
    "mistral-small3.2": "<cloud-server>/<path>",
}

def load(path):
    d = json.load(open(path))
    raw = d.get("hottest") or d.get("hottest_top200") or d.get("hottest_top100") or []
    H = d.get("hidden") or d.get("hidden_size")
    NL = d.get("n_layer")
    return raw, H, NL

print("="*100)
print("  EACH MODEL'S OWN ANCHOR SIGNATURE — what does ITS hottest landscape look like?")
print("="*100)

profiles = {}
for name, path in MODELS.items():
    raw, H, NL = load(path)
    # dim order by first appearance (= hottest)
    dim_first = []
    for e in raw:
        d = e["dim"]
        if d not in dim_first:
            dim_first.append(d)
    own_anchors = dim_first[:11]   # top-11 hottest dims = THIS model's own anchor candidates

    # layer concentration of those anchors
    layer_of_dim = defaultdict(list)
    for e in raw:
        if e["dim"] in own_anchors:
            layer_of_dim[e["dim"]].append(e["layer"])
    # quantile distribution
    q_counter = Counter(e["quantile_idx"] if "quantile_idx" in e else e.get("q",-1) for e in raw)
    # which layers dominate hot cells
    layer_counter = Counter(e["layer"] for e in raw)

    profiles[name] = {
        "H": H, "NL": NL, "own_anchors": own_anchors,
        "layer_of_dim": dict(layer_of_dim),
        "q_dist": dict(q_counter),
        "layer_dist": dict(layer_counter),
        "n_hot": len(raw),
    }

    print(f"\n--- {name}  H={H}  L={NL}  hot-cells={len(raw)} ---")
    print(f"  OWN top-11 anchors: {own_anchors}")
    # overlap with qwen anchor set
    overlap = sorted(set(own_anchors) & set(QWEN_ANCHORS))
    print(f"  ∩ qwen anchors:    {overlap}  ({len(overlap)}/11)")
    print(f"  layer concentration (where hot cells live):")
    top_layers = sorted(layer_counter.items(), key=lambda x:-x[1])[:5]
    print(f"    top layers: {top_layers}   (total layers={NL})")
    print(f"  quantile distribution: {dict(sorted(q_counter.items()))}")
    print(f"  per-own-anchor layer footprint:")
    for d in own_anchors[:6]:
        ls = sorted(set(layer_of_dim.get(d, [])))
        print(f"    dim {d:>5}: layers {ls}")

# Cross-model: pairwise own-anchor overlap
print()
print("="*100)
print("  PAIRWISE OWN-ANCHOR OVERLAP (top-11 each) — who shares with whom?")
print("="*100)
names = list(MODELS.keys())
print(f"{'':18s}" + " ".join(f"{n[:14]:>15s}" for n in names))
for a in names:
    row = f"{a[:18]:18s}"
    for b in names:
        ovl = len(set(profiles[a]["own_anchors"]) & set(profiles[b]["own_anchors"]))
        row += f"{ovl:>15d}"
    print(row)

# Mistral spotlight
print()
print("="*100)
print("  MISTRAL SPOTLIGHT — what is special about its OWN backbone?")
print("="*100)
m = profiles["mistral-small3.2"]
q = profiles["qwen2.5:7b"]
print(f"\nmistral own top-11 anchors:  {m['own_anchors']}")
print(f"qwen   own top-11 anchors:  {q['own_anchors']}")
print(f"\nmistral exclusive (not in qwen top-11): {sorted(set(m['own_anchors']) - set(q['own_anchors']))}")
print(f"qwen   exclusive (not in mistral top-11): {sorted(set(q['own_anchors']) - set(m['own_anchors']))}")
print(f"shared:                                    {sorted(set(m['own_anchors']) & set(q['own_anchors']))}")
print(f"\nmistral layer concentration (top-10 layers):")
ml = sorted(m["layer_dist"].items(), key=lambda x:-x[1])[:10]
for L, c in ml:
    bar = "#" * (c // 5)
    print(f"  L{L:>2d}: {c:>4d}  {bar}")
print(f"\nqwen layer concentration (top-10 layers):")
ql = sorted(q["layer_dist"].items(), key=lambda x:-x[1])[:10]
for L, c in ql:
    bar = "#" * (c // 5)
    print(f"  L{L:>2d}: {c:>4d}  {bar}")

# Frankenstein angle
print()
print("="*100)
print("  FRANKENSTEIN MERGE-CANDIDATE READOUT")
print("="*100)
print(f"""
qwen anchor set:      {sorted(QWEN_ANCHORS)}
mistral anchor set:   {sorted(m['own_anchors'])}
disjoint union:       {sorted(set(QWEN_ANCHORS) | set(m['own_anchors']))}
  → {len(set(QWEN_ANCHORS) | set(m['own_anchors']))} candidate anchor dims if we built a qwen×mistral hybrid

If mistral anchors live in DIFFERENT layers than qwen anchors (check above),
that's a clean "non-overlapping functional substrate" — a strong Frankenstein
merge thesis: qwen layers carry qwen anchor function, mistral layers carry
mistral anchor function, neither interferes.

If they overlap in layers but on different dims, then merge competes for
the same residual-stream bandwidth — risky.
""")
