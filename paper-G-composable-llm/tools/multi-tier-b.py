#!/usr/bin/env python3
"""
Multi-model Tier-B observation via HuggingFace transformers output_hidden_states.

Unlike Tier-A (llama_cpp logits hook → only last layer), Tier-B captures
EVERY layer's residual stream activation. This is the data source for
layer fingerprint extraction (Paper G).

Usage:
    python3 multi-tier-b.py qwen2.5-7b
    python3 multi-tier-b.py mistral-7b llama-8b phi3-medium
    python3 multi-tier-b.py all     # run all configured models sequentially

Each model writes:
    ~/mercury-tier-b/{label}/heat-summary-{label}-{ts}.json
    ~/mercury-tier-b/{label}/heat-{label}-{ts}.bin  (mmap'able cell heat)
    ~/mercury-tier-b/{label}/layer-activations-{label}-{ts}.npz  (per-layer stats)
"""
import sys, os, json, time
import numpy as np
from collections import defaultdict

# (label, hf_repo, expected_n_layer, expected_hidden, dtype_pref)
MODELS = {
    "qwen2.5-3b":     ("Qwen/Qwen2.5-3B-Instruct",   36, 2048, "bfloat16"),
    "qwen2.5-7b":     ("Qwen/Qwen2.5-7B-Instruct",   28, 3584, "bfloat16"),
    "qwen2.5-14b":    ("Qwen/Qwen2.5-14B-Instruct",  48, 5120, "bfloat16"),
    "llama-3.1-8b":   ("meta-llama/Llama-3.1-8B-Instruct", 32, 4096, "bfloat16"),
    "mistral-7b":     ("mistralai/Mistral-7B-Instruct-v0.3", 32, 4096, "bfloat16"),
    "mistral-small":  ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", 40, 5120, "bfloat16"),
    "phi3-medium":    ("microsoft/Phi-3-medium-4k-instruct", 40, 5120, "bfloat16"),
    "falcon3-7b":     ("tiiuae/Falcon3-7B-Instruct", 28, 3072, "bfloat16"),
    "internlm2-7b":   ("internlm/internlm2_5-7b-chat", 32, 4096, "bfloat16"),
    "ds-r1-7b":       ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 28, 3584, "bfloat16"),
}

PROMPTS = [
    "Explain in Chinese: what is quantum entanglement and why is it important?",
    "用中文寫一段關於龍蝦的科學說明，三段論述風格",
    "Write a Python function to detect race conditions in concurrent code",
    "Refactor this code for clarity: def f(x): return [i*2 for i in x if i>0]",
    "用中文解釋 race condition 並給程式範例",
    "What is the capital of France? Reply in one sentence.",
    "Prove that the sum of two even numbers is even, step by step.",
    "Translate to formal Chinese: 'The cat sat on the mat'",
    "List three benefits of exercise, formatted as bullet points",
    "Write a haiku about programming",
]

QUANTS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995]
N_QUANT = len(QUANTS)
OUT_BASE = os.path.expanduser("~/mercury-tier-b")

def run_one(label, repo, n_layer_exp, hidden_exp, dtype_pref):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_dir = f"{OUT_BASE}/{label}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\n  TIER-B observation: {label}  ({repo})\n{'='*70}")
    t0 = time.time()
    dtype = torch.bfloat16 if dtype_pref == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    NL = getattr(cfg, "num_hidden_layers", n_layer_exp)
    H  = getattr(cfg, "hidden_size", hidden_exp)
    arch = getattr(cfg, "model_type", "unknown")
    print(f"  arch={arch}  n_layer={NL}  hidden={H}")

    # Cell grid: (L, D, Q) for Tier-B (no seq position)
    TOTAL = NL * H * N_QUANT
    heat = np.zeros(TOTAL, dtype=np.uint32)
    # Also keep per-layer stats: top-K hot dims, selectivity
    per_layer_top_dims = [defaultdict(int) for _ in range(NL)]   # layer -> {dim: heat_count}
    per_layer_per_prompt_heat = np.zeros((NL, len(PROMPTS)), dtype=np.float32)

    ts = int(time.time())
    obs_path = f"{out_dir}/observation-{label}-{ts}.jsonl"
    with open(obs_path, "w") as obs:
        for pi, prompt in enumerate(PROMPTS):
            t1 = time.time()
            inputs = tok(prompt, return_tensors="pt").to("cpu")
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, return_dict=True)
            # hidden_states is tuple of (NL+1) tensors of shape (1, seq_len, H)
            # skip embedding layer (index 0), use layer outputs 1..NL
            hs = out.hidden_states[1:]
            assert len(hs) == NL, f"expected {NL} hidden states, got {len(hs)}"

            # for each layer, compute per-dim "heat" — use mean(abs(activation)) over seq positions
            for L_idx, layer_act in enumerate(hs):
                act = layer_act[0].float().cpu().numpy()  # (seq_len, H)
                # per-dim activation magnitude
                dim_mag = np.abs(act).mean(axis=0)  # (H,)
                # rank dims and assign quantile buckets
                ranks = np.argsort(-dim_mag)        # hottest dim first
                top_k = min(64, H)
                top_idx = ranks[:top_k]
                top_vals = dim_mag[top_idx]
                vmin, vmax = float(top_vals.min()), float(top_vals.max())
                rng = max(vmax - vmin, 1e-9)
                for dim, val in zip(top_idx.tolist(), top_vals.tolist()):
                    norm = (val - vmin) / rng
                    qi = 0
                    for i, q in enumerate(QUANTS):
                        if norm >= q: qi = i
                    nid = (L_idx * H + dim) * N_QUANT + qi
                    if 0 <= nid < TOTAL:
                        heat[nid] += 1
                    per_layer_top_dims[L_idx][dim] += 1
                per_layer_per_prompt_heat[L_idx, pi] = float(dim_mag.mean())

            ms = int((time.time()-t1)*1000)
            print(f"  p{pi+1:>2}: {ms:>6}ms  seq_len={inputs['input_ids'].shape[1]}")
            obs.write(json.dumps({"p":pi+1, "ms":ms,
                                  "seq_len":int(inputs['input_ids'].shape[1])})+"\n")

    # write heat binary
    heat_path = f"{out_dir}/heat-{label}-{ts}.bin"
    heat.tofile(heat_path)

    # top hottest cells global
    nonzero = np.nonzero(heat)[0]
    hottest = []
    if len(nonzero):
        order = nonzero[np.argsort(-heat[nonzero])][:500]
        for nid in order.tolist():
            qi = nid % N_QUANT
            dim = (nid // N_QUANT) % H
            layer = nid // (N_QUANT * H)
            hottest.append({"nid":int(nid),"layer":int(layer),"dim":int(dim),
                           "q":int(qi),"heat":int(heat[nid])})

    # per-layer top dims
    per_layer_summary = []
    for L_idx in range(NL):
        top = sorted(per_layer_top_dims[L_idx].items(), key=lambda x:-x[1])[:50]
        per_layer_summary.append({
            "layer": L_idx,
            "top_dims": [d for d,_ in top],
            "top_dim_heat": {str(d):int(c) for d,c in top},
            "per_prompt_mean_act": per_layer_per_prompt_heat[L_idx].tolist(),
        })

    summary = {
        "ts": ts, "label": label, "model": repo, "arch": arch,
        "tier": "B", "n_layer": NL, "hidden": H, "n_quant": N_QUANT,
        "total_cells": TOTAL, "cells_with_heat": int(len(nonzero)),
        "n_prompts": len(PROMPTS),
        "hottest_top500": hottest,
        "per_layer": per_layer_summary,
    }
    sp = f"{out_dir}/heat-summary-{label}-{ts}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  → {sp}")
    print(f"  → cells_with_heat = {len(nonzero):,}/{TOTAL:,}  ({100*len(nonzero)/TOTAL:.2f}%)")

    # save per-layer activation matrix as npz for downstream fingerprinting
    np.savez_compressed(
        f"{out_dir}/layer-activations-{label}-{ts}.npz",
        per_layer_per_prompt=per_layer_per_prompt_heat,
        per_layer_top_dims=np.array(
            [[per_layer_top_dims[L].get(d,0) for d in range(H)] for L in range(NL)],
            dtype=np.uint32
        ),
    )
    print(f"  → layer-activations-{label}-{ts}.npz")
    del model
    import gc; gc.collect()

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    targets = sys.argv[1:]
    if targets == ["all"]:
        targets = list(MODELS.keys())
    for label in targets:
        if label not in MODELS:
            print(f"unknown label: {label} (available: {list(MODELS.keys())})")
            continue
        try:
            run_one(label, *MODELS[label])
        except Exception as e:
            print(f"FAILED {label}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
