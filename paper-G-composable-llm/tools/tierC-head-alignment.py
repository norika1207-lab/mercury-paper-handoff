#!/usr/bin/env python3
"""Cross-model attention head alignment using Tier-C 5D fingerprints.

For each (modelA, layerA, headA) find closest (modelB, layerB, headB)
across all other models using Euclidean distance in 5D feature space:
  [entropy_norm, focus_pos_rel, self_attn, prev1_attn, bos_attn]

Output:
  - top N strongest cross-model head matches
  - per-model head export statistics
  - depth-bucketed analysis (where do the best matches concentrate)
"""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, glob, os
from collections import defaultdict
import numpy as np

DIR = "REPO_ROOT/tier-c-data"

def load_model(p):
    d = json.load(open(p))
    label = d["tag"]
    NL = d["n_layer"]
    NH = d["n_head"]
    arch = d.get("arch", "?")
    # Build (NL, NH, 5) numpy array
    fp = np.zeros((NL, NH, 5), dtype=np.float32)
    for L_entry in d["per_layer_head_stats"]:
        L = L_entry["layer"]
        for h_entry in L_entry["heads"]:
            h = h_entry["head"]
            fp[L, h, 0] = h_entry["entropy"]
            fp[L, h, 1] = h_entry["focus_pos"]
            fp[L, h, 2] = h_entry["self_attn"]
            fp[L, h, 3] = h_entry["prev1_attn"]
            fp[L, h, 4] = h_entry["bos_attn"]
    return {"label": label, "arch": arch, "NL": NL, "NH": NH, "fp": fp}

models = {}
for p in sorted(glob.glob(f"{DIR}/heat-summary-*.json")):
    m = load_model(p)
    models[m["label"]] = m
    print(f"  loaded {m['label']:>20s}  arch={m['arch']:<10}  L={m['NL']:>2}  H={m['NH']:>2}")

print(f"\nloaded {len(models)} models")
total_heads = sum(m["NL"] * m["NH"] for m in models.values())
print(f"total heads: {total_heads}")

# ============================================================
# For each (modelA, layerA, headA), find best match in each other model
# ============================================================
all_matches = []  # list of (sim, modelA, LA, HA, depthA, modelB, LB, HB, depthB)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# Pre-flatten each model into (n_head_total, 5) + index back
labels = sorted(models.keys())
for i, A in enumerate(labels):
    mA = models[A]
    fpA = mA["fp"].reshape(-1, 5)  # (NL*NH, 5)
    for j, B in enumerate(labels):
        if A == B: continue
        mB = models[B]
        fpB = mB["fp"].reshape(-1, 5)
        # Compute cosine sim matrix: (NHA*NLA, NHB*NLB)
        # Normalize
        fpA_norm = fpA / (np.linalg.norm(fpA, axis=1, keepdims=True) + 1e-12)
        fpB_norm = fpB / (np.linalg.norm(fpB, axis=1, keepdims=True) + 1e-12)
        sim_matrix = fpA_norm @ fpB_norm.T  # (NA_heads, NB_heads)
        # For each A head, find best B head
        best_idx = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)
        # Iterate
        for a_flat, (b_flat, s) in enumerate(zip(best_idx, best_sim)):
            LA, HA = divmod(a_flat, mA["NH"])
            LB, HB = divmod(int(b_flat), mB["NH"])
            depthA = LA / max(1, mA["NL"] - 1)
            depthB = LB / max(1, mB["NL"] - 1)
            all_matches.append((float(s), A, LA, HA, depthA, B, LB, HB, depthB))

print(f"\ntotal cross-model best-match pairs: {len(all_matches)}")

# ============================================================
# Top N strongest cross-model matches
# ============================================================
all_matches.sort(reverse=True)
seen_pairs = set()
top_unique = []
for s, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    key = (A, LA, HA, B)
    if key in seen_pairs: continue
    seen_pairs.add(key)
    top_unique.append((s, A, LA, HA, dA, B, LB, HB, dB))
    if len(top_unique) >= 30: break

print("\n" + "="*100)
print("  TOP-30 STRONGEST CROSS-MODEL ATTENTION HEAD ALIGNMENTS (5D fingerprint cosine sim)")
print("="*100)
print(f"{'sim':<6} {'modelA':<22} {'L':>3} {'H':>3} {'dep%':>4}  {'modelB':<22} {'L':>3} {'H':>3} {'dep%':>4}  {'Δdep':>4}")
print("-"*100)
for s, A, LA, HA, dA, B, LB, HB, dB in top_unique:
    print(f"{s:.4f} {A[:20]:<22} {LA:>3} {HA:>3} {int(dA*100):>3}%   {B[:20]:<22} {LB:>3} {HB:>3} {int(dB*100):>3}%   {int(abs(dA-dB)*100):>3}%")

# ============================================================
# Depth-bucket analysis: where do best matches concentrate?
# ============================================================
print("\n" + "="*100)
print("  DEPTH BUCKET: where do best (sim>=0.99) cross-model heads concentrate?")
print("="*100)
bins = defaultdict(int)
for s, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    if s >= 0.99:
        mean_depth = (dA + dB) / 2
        b = min(9, int(mean_depth * 10))
        bins[b] += 1
print(f"{'depth%':<10}  count")
for b in range(10):
    bar = '#' * (bins[b] // 20)
    print(f"  {b*10:>3}-{(b+1)*10:<3}%  {bins[b]:>5}  {bar}")

# ============================================================
# Per-model: which heads are MOST cross-arch matchable (vs MOST unique)
# ============================================================
print("\n" + "="*100)
print("  Per-model: most cross-arch matchable heads (top-3 each)")
print("="*100)
per_model_best = defaultdict(list)
for s, A, LA, HA, dA, B, LB, HB, dB in all_matches:
    per_model_best[A].append((s, LA, HA, dA, B, LB, HB))
for m_label in sorted(per_model_best.keys()):
    matches = per_model_best[m_label]
    # Find this model's heads with highest average cross-model sim
    by_head = defaultdict(list)
    for s, LA, HA, dA, B, LB, HB in matches:
        by_head[(LA, HA, dA)].append(s)
    head_avg = [(np.mean(sims), L, H, d) for (L, H, d), sims in by_head.items()]
    head_avg.sort(reverse=True)
    top3 = head_avg[:3]
    print(f"\n  {m_label}  ({models[m_label]['NL']}L × {models[m_label]['NH']}H):")
    for avg_s, L, H, d in top3:
        print(f"    L{L:>2} H{H:>2} (depth {int(d*100)}%)  avg cross-arch sim = {avg_s:.4f}")

# ============================================================
# Save match matrix to JSON for later use
# ============================================================
out_path = "REPO_ROOT/tier-c-head-alignment.json"
top1000 = [
    {"sim": s, "modelA": A, "LA": LA, "HA": HA, "depthA": dA,
     "modelB": B, "LB": LB, "HB": HB, "depthB": dB}
    for s, A, LA, HA, dA, B, LB, HB, dB in all_matches[:1000]
]
with open(out_path, "w") as f:
    json.dump({"models": labels, "top1000_matches": top1000,
               "depth_distribution_high_sim": dict(bins)}, f, indent=2)
print(f"\n  → saved top-1000 matches to {out_path}")
