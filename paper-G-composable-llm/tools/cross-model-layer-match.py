#!/usr/bin/env python3
"""Cross-model layer matching matrix.

For each pair of (model_A layer i, model_B layer j), compute fingerprint similarity.
Output: similarity matrix + best-match pairings.

Similarity metrics:
  - dim_range_cosine: cosine of (low, mid, high) ratio vectors
  - top_dim_jaccard: jaccard of normalized top-dim sets (normalized = dim/hidden bucketed to 100 bins)
  - q_distribution_cosine: cosine of quantile bucket vectors
  Combined = weighted average

Usage: python3 cross-model-layer-match.py <fp1.json> <fp2.json> [...]
"""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, os, json, math
from itertools import combinations

OUT_DIR = "REPO_ROOT/layer-fingerprints"

def vec_cos(a, b):
    da = sum(x*x for x in a)**0.5
    db = sum(x*x for x in b)**0.5
    if da == 0 or db == 0: return 0
    return sum(x*y for x,y in zip(a,b)) / (da*db)

def normalize_dims(dims, hidden, bins=100):
    """Bucket dims into normalized position bins so different hidden sizes are comparable."""
    return set(int(d / hidden * bins) for d in dims)

def fingerprint_sim(fp_a, fp_b, hidden_a, hidden_b):
    if fp_a["n_hot"] == 0 or fp_b["n_hot"] == 0:
        return 0.0, {}
    # dim_range cosine
    ra = fp_a["dim_range_ratio"]
    rb = fp_b["dim_range_ratio"]
    drs = vec_cos([ra["low"], ra["mid"], ra["high"]],
                  [rb["low"], rb["mid"], rb["high"]])
    # top-dim jaccard (normalized to relative position)
    sa = normalize_dims(fp_a["top_hot_dims"], hidden_a)
    sb = normalize_dims(fp_b["top_hot_dims"], hidden_b)
    jac = len(sa & sb) / max(1, len(sa | sb))
    # q-dist cosine
    qa_vec = [fp_a["q_distribution"].get(str(q), fp_a["q_distribution"].get(q, 0)) for q in range(10)]
    qb_vec = [fp_b["q_distribution"].get(str(q), fp_b["q_distribution"].get(q, 0)) for q in range(10)]
    qsim = vec_cos(qa_vec, qb_vec)
    # heat concentration similarity (1 - abs diff)
    csim = 1 - abs(fp_a["heat_concentration"] - fp_b["heat_concentration"])
    combined = 0.4*drs + 0.3*jac + 0.2*qsim + 0.1*csim
    return combined, {"dim_range": drs, "jaccard": jac, "q_dist": qsim, "conc": csim}

def match_pair(a_data, b_data):
    A = a_data["fingerprints"]
    B = b_data["fingerprints"]
    Ha, Hb = a_data["hidden"], b_data["hidden"]
    NA, NB = len(A), len(B)
    # similarity matrix
    M = [[0.0]*NB for _ in range(NA)]
    for i in range(NA):
        for j in range(NB):
            sim, _ = fingerprint_sim(A[i], B[j], Ha, Hb)
            M[i][j] = sim
    return M

def print_matrix(M, label_a, label_b, na, nb):
    print(f"\n{'='*80}")
    print(f"  LAYER MATCHING: {label_a} ({na} layers) ↔ {label_b} ({nb} layers)")
    print(f"  cell = similarity (0=no, 1=identical)  · = either layer empty")
    print(f"{'='*80}")
    # header
    hdr_tag = "A\\B"
    print(f"{hdr_tag:>4}", end="")
    for j in range(nb):
        print(f"{j:>4}", end="")
    print()
    print("-"*(4 + 4*nb))
    for i in range(na):
        print(f"{i:>4}", end="")
        for j in range(nb):
            v = M[i][j]
            if v == 0:
                print(f"   ·", end="")
            else:
                # color heat: high=★, mid=●, low=·
                pct = int(v*99)
                print(f"{pct:>4}", end="")
        print()

def best_matches(M, label_a, label_b, na, nb, top=8):
    pairs = []
    for i in range(na):
        for j in range(nb):
            if M[i][j] > 0:
                pairs.append((M[i][j], i, j))
    pairs.sort(reverse=True)
    print(f"\n  TOP-{top} BEST LAYER MATCHES ({label_a} ↔ {label_b}):")
    seen_a = set(); seen_b = set()
    shown = 0
    for sim, i, j in pairs:
        if i in seen_a or j in seen_b:
            continue
        seen_a.add(i); seen_b.add(j)
        print(f"    {label_a} L{i:>2} ↔ {label_b} L{j:>2}   sim={sim:.3f}")
        shown += 1
        if shown >= top: break

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    datas = []
    for p in sys.argv[1:]:
        d = json.load(open(p))
        datas.append(d)
        print(f"loaded: {d['label']}  ({d['n_layer']}L × {d['hidden']}H)")

    for a, b in combinations(datas, 2):
        M = match_pair(a, b)
        # save matrix
        out_path = f"{OUT_DIR}/match-{a['label']}-vs-{b['label']}.json"
        json.dump({
            "a": a["label"], "b": b["label"],
            "n_layer_a": a["n_layer"], "n_layer_b": b["n_layer"],
            "matrix": M,
        }, open(out_path, "w"), indent=2)
        print_matrix(M, a["label"], b["label"], a["n_layer"], b["n_layer"])
        best_matches(M, a["label"], b["label"], a["n_layer"], b["n_layer"])
        print(f"\n  → matrix saved: {out_path}")

if __name__ == "__main__":
    main()
