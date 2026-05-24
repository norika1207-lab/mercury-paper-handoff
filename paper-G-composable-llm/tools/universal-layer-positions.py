#!/usr/bin/env python3
"""Find universal functional layer positions across ALL models.

For each layer in each model, normalize position to depth%.
Then find positions where cross-model layer fingerprint similarity is consistently high.

Output: ranked list of "universal layer positions" — depth% values where most
models have functionally-equivalent layers.
"""
import os
REPO_ROOT = os.environ.get("MERCURY_HANDOFF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json, glob
from collections import defaultdict
from itertools import combinations

FP_DIR = "REPO_ROOT/layer-fingerprints"

def load_all():
    models = {}
    for p in glob.glob(f"{FP_DIR}/*.json"):
        if "match-" in p: continue
        d = json.load(open(p))
        models[d["label"]] = d
    return models

def main():
    models = load_all()
    print(f"loaded {len(models)} models: {list(models.keys())}")

    # Load all pairwise matrices and find best matches at each depth
    pair_matches = {}
    for p in glob.glob(f"{FP_DIR}/match-*-vs-*.json"):
        d = json.load(open(p))
        a, b = d["a"], d["b"]
        NA, NB = d["n_layer_a"], d["n_layer_b"]
        M = d["matrix"]
        pair_matches[(a, b)] = (M, NA, NB)

    # For each model, what depth% does each layer correspond to?
    def depth(layer, n_layer):
        return layer / max(1, n_layer - 1)

    # Find "universal positions": for depth bin (0-10%, 10-20%, ..., 90-100%),
    # count how many cross-model pairs have a high-sim match in that bin
    DEPTH_BINS = 10
    bin_alignments = defaultdict(lambda: defaultdict(int))  # bin -> {(modelA, modelB): count of matches > 0.7}

    for (a, b), (M, NA, NB) in pair_matches.items():
        for i in range(NA):
            for j in range(NB):
                if M[i][j] >= 0.7:
                    dA = depth(i, NA)
                    dB = depth(j, NB)
                    # use mean depth bin
                    bin_idx = min(DEPTH_BINS-1, int((dA + dB) / 2 * DEPTH_BINS))
                    bin_alignments[bin_idx][(a, b)] = max(
                        bin_alignments[bin_idx].get((a, b), 0), M[i][j]
                    )

    print("\n" + "="*80)
    print("  UNIVERSAL LAYER DEPTH SLOTS — where do cross-model matches concentrate?")
    print("="*80)
    print(f"{'depth%':<10} {'pair_count':<12} {'avg_sim':<10} {'pairs_aligned'}")
    print("-"*80)
    for bin_idx in range(DEPTH_BINS):
        pairs = bin_alignments[bin_idx]
        if not pairs: continue
        avg_sim = sum(pairs.values()) / len(pairs)
        depth_lo = bin_idx * 10
        depth_hi = (bin_idx + 1) * 10
        print(f"{depth_lo:>3}-{depth_hi:<3}%   {len(pairs):>10}   {avg_sim:>8.3f}   {len(pairs)}/{len(pair_matches)} model-pairs aligned")

    # Find the strongest "stand-out" alignments — pairs that align AT same depth
    print("\n" + "="*80)
    print("  ★ STRONGEST SAME-DEPTH CROSS-MODEL ALIGNMENTS (Δdepth < 5%)")
    print("="*80)
    candidates = []
    for (a, b), (M, NA, NB) in pair_matches.items():
        for i in range(NA):
            for j in range(NB):
                if M[i][j] < 0.75: continue
                dA, dB = depth(i, NA), depth(j, NB)
                if abs(dA - dB) < 0.05:
                    candidates.append((M[i][j], a, i, NA, dA, b, j, NB, dB))
    candidates.sort(reverse=True)
    print(f"{'sim':<6}  {'modelA':<22} {'L':>3} {'/':>3} {'%':>4}   {'modelB':<22} {'L':>3} {'/':>3} {'%':>4}")
    print("-"*100)
    for sim, a, i, na, da, b, j, nb, db in candidates[:20]:
        print(f"{sim:.3f}  {a[:20]:<22} {i:>3} /{na:>3} {int(da*100):>3}%   "
              f"{b[:20]:<22} {j:>3} /{nb:>3} {int(db*100):>3}%")

    # For frankenstein: which model has the most "donatable" layers (high cross-arch sim)?
    print("\n" + "="*80)
    print("  ⚒ FRANKENSTEIN-READY LAYERS — each model's most cross-compatible layers")
    print("="*80)
    model_layer_max_sim = defaultdict(lambda: defaultdict(float))  # model -> {layer: best_cross_sim}
    for (a, b), (M, NA, NB) in pair_matches.items():
        for i in range(NA):
            best_for_i = max(M[i]) if M[i] else 0
            if best_for_i > model_layer_max_sim[a][i]:
                model_layer_max_sim[a][i] = best_for_i
        # also reverse direction
        for j in range(NB):
            best_for_j = max(M[i][j] for i in range(NA))
            if best_for_j > model_layer_max_sim[b][j]:
                model_layer_max_sim[b][j] = best_for_j

    for model, layer_sims in sorted(model_layer_max_sim.items()):
        sorted_layers = sorted(layer_sims.items(), key=lambda x: -x[1])
        top3 = sorted_layers[:5]
        n_layer = models[model]["n_layer"]
        print(f"\n  {model} ({n_layer}L):")
        for L, sim in top3:
            pct = int(L/(n_layer-1)*100)
            print(f"    L{L:>2} ({pct:>3}% depth)  best cross-arch sim = {sim:.3f}")

if __name__ == "__main__":
    main()
