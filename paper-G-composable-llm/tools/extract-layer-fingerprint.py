#!/usr/bin/env python3
"""Extract per-layer functional fingerprints from Tier-B heat-summary JSON.

For each (model, layer L), produce a fingerprint:
  - top_hot_dims: top-K dims hot in this layer
  - heat_concentration: how concentrated heat is (Gini-like)
  - n_unique_hot: distinct dims appearing in hottest
  - dim_range_signature: low/mid/high dim ratio
  - per_q_distribution: how heat distributes across quantile buckets

Usage:
    python3 extract-layer-fingerprint.py <heat-summary-1.json> [...]
    → writes layer-fingerprints/{model}.json for each input
"""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os, json, math
from collections import defaultdict, Counter

OUT_DIR = "REPO_ROOT/layer-fingerprints"
os.makedirs(OUT_DIR, exist_ok=True)

def fingerprint_one(path):
    d = json.load(open(path))
    label = d.get("tag") or d.get("model", os.path.basename(path).replace("heat-summary-","").replace(".json",""))
    NL = d["n_layer"]
    H = d["hidden"]
    hottest = d.get("hottest") or d.get("hottest_top500") or d.get("hottest_top200") or d.get("hottest_top100") or []

    # group hottest entries by layer
    by_layer = defaultdict(list)
    for e in hottest:
        by_layer[e["layer"]].append(e)

    fingerprints = []
    for L in range(NL):
        entries = by_layer.get(L, [])
        if not entries:
            fingerprints.append({
                "layer": L, "n_hot": 0, "top_hot_dims": [],
                "heat_concentration": 0.0, "n_unique_hot": 0,
                "dim_range_ratio": {"low": 0, "mid": 0, "high": 0},
                "q_distribution": {}, "total_heat": 0,
            })
            continue

        # top hot dims ordered by heat
        sorted_entries = sorted(entries, key=lambda x: -x["heat"])
        seen_dims = []
        for e in sorted_entries:
            if e["dim"] not in seen_dims:
                seen_dims.append(e["dim"])
        top_dims = seen_dims[:20]

        # heat concentration: top-3 share of total heat
        total_heat = sum(e["heat"] for e in entries)
        top3_heat = sum(e["heat"] for e in sorted_entries[:3])
        concentration = top3_heat / total_heat if total_heat else 0

        # dim range distribution (low: <H/3, mid: H/3..2H/3, high: >2H/3)
        ranges = {"low": 0, "mid": 0, "high": 0}
        for e in entries:
            r = e["dim"] / H
            if r < 1/3: ranges["low"] += e["heat"]
            elif r < 2/3: ranges["mid"] += e["heat"]
            else: ranges["high"] += e["heat"]
        rsum = sum(ranges.values())
        dim_range_ratio = {k: round(v/rsum, 3) if rsum else 0 for k,v in ranges.items()}

        # quantile distribution
        q_dist = Counter(e["q"] for e in entries)

        fingerprints.append({
            "layer": L,
            "n_hot": len(entries),
            "n_unique_hot": len(seen_dims),
            "top_hot_dims": top_dims,
            "total_heat": total_heat,
            "heat_concentration": round(concentration, 3),
            "dim_range_ratio": dim_range_ratio,
            "q_distribution": dict(q_dist),
        })

    result = {
        "label": label, "n_layer": NL, "hidden": H,
        "tier": "B", "source": os.path.basename(path),
        "fingerprints": fingerprints,
    }
    out_path = f"{OUT_DIR}/{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result, out_path

def summarize(result):
    """Print a quick visual summary of per-layer character."""
    label = result["label"]
    NL = result["n_layer"]
    H = result["hidden"]
    print(f"\n{'='*80}\n  {label}  ({NL} layers × {H} hidden)\n{'='*80}")
    print(f"{'L':>3} {'n_hot':>6} {'uniq':>5} {'conc':>5}  {'low':>5} {'mid':>5} {'high':>5}  top-5 dims")
    print("-"*80)
    for fp in result["fingerprints"]:
        if fp["n_hot"] == 0:
            print(f"{fp['layer']:>3}  --- empty layer")
            continue
        r = fp["dim_range_ratio"]
        print(f"{fp['layer']:>3} {fp['n_hot']:>6} {fp['n_unique_hot']:>5} {fp['heat_concentration']:>5.2f}  "
              f"{r['low']:>5.2f} {r['mid']:>5.2f} {r['high']:>5.2f}  {fp['top_hot_dims'][:5]}")

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for p in sys.argv[1:]:
        result, out_path = fingerprint_one(p)
        print(f"→ {out_path}")
        summarize(result)

if __name__ == "__main__":
    main()
