#!/usr/bin/env python3
"""Cross-model attention head alignment, v2: Euclidean distance + percentile ranking.

Cosine similarity saturated at 1.0 because the 5D fingerprints all point
roughly the same direction in 5D space. Switch to Euclidean distance
which preserves magnitude differences, and report top-K LOW-distance
matches (the actual "twin heads" across models).
"""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, glob, os
from collections import defaultdict
import numpy as np

DIR = "REPO_ROOT/tier-c-data"

def load_model(p):
    d = json.load(open(p))
    label = d["tag"]; NL = d["n_layer"]; NH = d["n_head"]; arch = d.get("arch", "?")
    fp = np.zeros((NL, NH, 5), dtype=np.float32)
    for L_entry in d["per_layer_head_stats"]:
        L = L_entry["layer"]
        for h_entry in L_entry["heads"]:
            h = h_entry["head"]
            fp[L, h] = [h_entry["entropy"], h_entry["focus_pos"], h_entry["self_attn"],
                        h_entry["prev1_attn"], h_entry["bos_attn"]]
    return {"label": label, "arch": arch, "NL": NL, "NH": NH, "fp": fp}

models = {}
for p in sorted(glob.glob(f"{DIR}/heat-summary-*.json")):
    m = load_model(p)
    models[m["label"]] = m

labels = sorted(models.keys())
print(f"loaded {len(labels)} models")

# Pair every head against every other-model head, Euclidean in 5D
# Smaller dist = better match
all_matches = []
for A in labels:
    mA = models[A]
    fpA = mA["fp"].reshape(-1, 5)
    for B in labels:
        if A == B: continue
        mB = models[B]
        fpB = mB["fp"].reshape(-1, 5)
        # Euclidean distance matrix
        diff = fpA[:, None, :] - fpB[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))  # (NA_heads, NB_heads)
        # For each A head, find best (min dist) B head
        best_idx = dist.argmin(axis=1)
        best_dist = dist.min(axis=1)
        for a_flat, (b_flat, d) in enumerate(zip(best_idx, best_dist)):
            LA, HA = divmod(a_flat, mA["NH"])
            LB, HB = divmod(int(b_flat), mB["NH"])
            depthA = LA / max(1, mA["NL"] - 1)
            depthB = LB / max(1, mB["NL"] - 1)
            all_matches.append((float(d), A, LA, HA, depthA, B, LB, HB, depthB))

print(f"total cross-model best-match pairs: {len(all_matches)}")
all_matches.sort()  # ascending: smallest distance first

print("\n" + "="*100)
print("  TOP-30 STRONGEST CROSS-MODEL HEAD MATCHES (lowest Euclidean dist in 5D space)")
print("="*100)
print(f"{'dist':<7}  {'modelA':<22} {'L':>3} {'H':>3} {'dep%':>4}  {'modelB':<22} {'L':>3} {'H':>3} {'dep%':>4}  {'Δdep':>4}")
print("-"*100)
seen = set()
shown = 0
for d, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    # Skip if reverse direction already shown (symmetry)
    key = tuple(sorted([(A, LA, HA), (B, LB, HB)]))
    if key in seen: continue
    seen.add(key)
    print(f"{d:.5f}  {A[:20]:<22} {LA:>3} {HA:>3} {int(dA*100):>3}%   {B[:20]:<22} {LB:>3} {HB:>3} {int(dB*100):>3}%   {int(abs(dA-dB)*100):>3}%")
    shown += 1
    if shown >= 30: break

# Distance distribution
dists = np.array([m[0] for m in all_matches])
print("\n" + "="*100)
print(f"  Distance distribution: min={dists.min():.6f}  median={np.median(dists):.4f}  max={dists.max():.4f}")
print("="*100)
for pct in [0.1, 1, 5, 10, 25, 50]:
    th = np.percentile(dists, pct)
    print(f"  {pct:>5.1f}-th percentile threshold = {th:.5f}")

# Depth bucket with absolute distance threshold (top 1% closest matches)
top1pct_threshold = np.percentile(dists, 1)
print("\n" + "="*100)
print(f"  DEPTH BUCKET: where do TOP-1% CLOSEST matches (dist<={top1pct_threshold:.4f}) concentrate?")
print("="*100)
bins = defaultdict(int)
for d, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    if d > top1pct_threshold: continue
    mean_depth = (dA + dB) / 2
    b = min(9, int(mean_depth * 10))
    bins[b] += 1
print(f"{'depth%':<10}  count")
for b in range(10):
    bar = '#' * (bins[b] // 30)
    print(f"  {b*10:>3}-{(b+1)*10:<3}%  {bins[b]:>5}  {bar}")

# Cross-arch (not same family) emphasis
print("\n" + "="*100)
print("  TOP-20 CROSS-ARCHITECTURE matches (excluding qwen-family internal matches)")
print("="*100)
qwen_family = {"qwen3b", "qwen7b", "qwen14b", "qwen-coder-7b"}
shown = 0
seen_xa = set()
for d, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    if A in qwen_family and B in qwen_family: continue
    key = tuple(sorted([(A, LA, HA), (B, LB, HB)]))
    if key in seen_xa: continue
    seen_xa.add(key)
    print(f"{d:.5f}  {A[:20]:<22} {LA:>3} {HA:>3} {int(dA*100):>3}%   {B[:20]:<22} {LB:>3} {HB:>3} {int(dB*100):>3}%   {int(abs(dA-dB)*100):>3}%")
    shown += 1
    if shown >= 20: break

# Save
top1000 = [{"dist": d, "modelA": A, "LA": LA, "HA": HA, "depthA": dA,
            "modelB": B, "LB": LB, "HB": HB, "depthB": dB}
           for d, A, LA, HA, dA, B, LB, HB, dB in all_matches[:1000]]
with open("REPO_ROOT/tier-c-head-alignment-v2.json", "w") as f:
    json.dump({"models": labels, "top1000_matches": top1000,
               "top1pct_depth_distribution": dict(bins)}, f, indent=2)
print(f"\n  → saved top-1000 matches")
