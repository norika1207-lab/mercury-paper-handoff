# Paper C — What Doesn't Work in Cross-Fine-Tune Merge

**Suggested title:** *Negative Results in Cross-Fine-Tune LLM Merge: A Systematic Failure-Mode Catalog of SLERP, TIES, DARE, and Observation-Driven Rescue*

**Target venue:** NeurIPS 2026 Mech Interp Workshop / BlackboxNLP / ICLR Tiny Papers (負面結果在 workshop 很受歡迎)
**狀態:** **可立即動筆**

---

## Headline claim

We systematically compare 8 cross-fine-tune merge methods on the Qwen2.5-Instruct × Qwen2.5-Coder pair, evaluated on a 5-prompt test set covering multilingual reasoning, structured argumentation, and code tasks. We identify four reproducible failure modes that the model-merging community has not previously catalogued: (1) **conservative-SLERP collapse** (low t = 0.2 fails worse than aggressive t = 0.9, counter to common practice); (2) **DARE-TIES gibberish** (random drop + magnitude voting produces token noise across all prompts); (3) **observation-driven TIES regression** (using observation-derived density profiles introduces new bugs); (4) **scalar-only anchor rescue breaks integration** (patching layer-norm gain at anchor dim positions without patching downstream attention/MLP read paths causes generation collapse). Among the 8 methods, **simple passthrough Frankenmerge with conservative Coder injection (24 Instruct + 4 Coder layers, "v7b")** is the empirical winner — outperforming all blend methods on coherence preservation.

---

## 核心 figure 數據

### Figure 1 — 8 method comparison table

| Rank | Method | Config | 5-prompt 結果 | Notes |
|---|---|---|---|---|
| 🥇 | **v7b** | passthrough 24 Instruct + 4 Coder layers | 3 strong + 1 best + 1 partial | Winner |
| 🥈 | **v3** | TIES uniform (density 0.55, weight 65/35) | 4 strong + 1 partial | Stable baseline |
| BASE | qwen2.5-7B-Instruct | – | 2 ✓ + 2 △ + 1 ✓ | Reference |
| 4 | v7c | Sandwich graft (Inst-Coder-Inst) | 3 ✓ + 2 ✗ | Mixed |
| 5 | v4 | Observation-driven layer-banded TIES | 2 strong + 1 same + 2 regression | New failure mode |
| 6 | v1 | SLERP aggressive (t=0.1→0.9) | 5 fail (language drift, repetition, capability loss) | Known issue |
| 7 | v2 | SLERP conservative (t≈0.2 全層) | **5 complete collapse** | **Counter-intuitive new finding** |
| **8** | **v5** | **DARE-TIES** | **5 全部 token gibberish** | **Complete failure** |

### Figure 2 — Per-prompt outcome detail

Show side-by-side outputs of BASE vs v1 vs v2 vs v5 vs v7b on prompt 5 (race condition) and prompt 4 (code refactor). Visually compelling failure illustrations:
- v1: 繁體 → 簡體漂移, "謝謝你的耐心傾聽" loop ×5
- v2: A/B/C/D/E/F/G/... 多項選擇式輸出
- v5: 0武℠ bénéfic1-masteruantkea Pendant_equalTo... token gibberish
- v7b: clean 三段式 + 完整反駁 + 回應反駁

### Figure 3 — Scalar vs subspace rescue (the integration constraint)

| Method | Layer norm patched | Attention/MLP patched | Result |
|---|---|---|---|
| v8 LN-only | ✓ (99 scalars) | ✗ | Generation collapse on 2/5 prompts |
| v9 full subspace | ✓ | ✓ (550 slices) | Coherence preserved |

**Interpretation**: Coder fine-tune adjusts layer-norm gain at anchor dims as *integrated adaptation* to its modified read/write paths. Anchor rescue that only patches LN scalar breaks this integration.

### Figure 4 — Observation-driven TIES failure (v4 deep dive)

v4 used Mercury observation to set per-layer TIES density/weight profile:
- lower layers (0-9): high Instruct weight (0.78)
- middle layers (10-19): 50/50 with high density (0.65)
- upper layers (20-27): high Coder weight (0.62)

Result: prompt 3 (code-bug) regressed — model called dict.update() "a set operation". Demonstration that **naive observation → merge parameter mapping** doesn't preserve correctness.

---

## Method 章節

### C.1 Setup
- Base models: Qwen2.5-7B-Instruct + Qwen2.5-Coder-7B-Instruct (same architecture, same tokenizer, different fine-tune objectives)
- Hardware: Mac mini M4 Pro 24GB unified memory
- mergekit 0.1.4+ from git main (pydantic 2.10 compat patch)
- CPU bfloat16 inference (Metal hit OOM on full 14GB model load)

### C.2 8 merge configurations
- v1: SLERP, layer-wise t profile [0.1, 0.2, 0.4, 0.6, 0.8] for attn; [0.1, 0.2, 0.5, 0.7, 0.9] for mlp
- v2: SLERP, uniform t = 0.20 (attn), 0.25 (mlp)
- v3: TIES uniform density 0.55, weight 65/35
- v4: Observation-driven TIES (layer-banded density profile from Mercury observation)
- v5: DARE-TIES (density 0.55, weight 65/35)
- v6: Passthrough 20 Instruct + 8 Coder layers
- v7a/b/c: Passthrough variants (18+10, 24+4, sandwich Instruct-Coder-Instruct)
- v8/v9: Post-merge anchor rescue (LN-only, full subspace)

### C.3 Evaluation
5-prompt test set + per-prompt failure mode classification.

---

## 資料路徑

```
<workstation-A>/frankenstein/
├── compare-results.json              (v1 baseline)
├── compare-all-results.json          (BASE/v1/v2/v3 ablation 4-way)
├── compare-v7-results.json           (BASE/v6/v7a/v7b/v7c passthrough ablation)
├── compare-v8-results.json           (LN-only rescue regression)
├── compare-v9-results.json           (full subspace rescue recovery)
├── merge-v{1,2,3,4,5,6,7a,7b,7c}.yml (mergekit configs)
└── *.log                              (full mergekit + inference logs)
```

---

## 寫稿 outline

1. **Abstract** (150 字): 8 method comparison, 4 failure modes, passthrough wins, subspace constraint
2. **Introduction**: model merging community uses SLERP/TIES/DARE as standards; no systematic failure analysis on cross-fine-tune pairs
3. **Method**: 8 configurations, eval protocol
4. **Results**: 
   - 4.1 SLERP-low-t collapse (counter-intuitive)
   - 4.2 DARE-TIES gibberish (cross-fine-tune incompatible)
   - 4.3 Observation-driven TIES regression
   - 4.4 Passthrough family wins
   - 4.5 Subspace consistency constraint
5. **Discussion**:
   - Fine-tunes adapt as integrated subspace ↔ blend methods that average weights destroy integration
   - Passthrough success implies sliceable functional block hypothesis
6. **Limitations**: single pair (Instruct + Coder), 5-prompt evaluation, single arch family
7. **Reproducibility**: open scripts, single Mac mini

---

## Limitations 必寫

1. **Single model pair**: 只測 Qwen Instruct × Coder。應加 Llama Instruct × Llama Code, Mistral × Mistral-Code 等
2. **5-prompt small**: 擴到 20-100 prompts，加 hypothesis test
3. **No quantitative MMLU/HumanEval**: 沒跑標準 benchmark。加 HumanEval / MMLU 比較 v7b vs BASE 數字會更有 reviewer 信賴度
4. **Architecture-specific**: 只 qwen2 arch；不同 arch (llama, mistral) 可能 passthrough 不一定贏
5. **Inference precision**: CPU bf16 偶發空白輸出 (v6 P4, v7a multiple)，無法 100% 排除非 merge 原因

---

## Related work 引用

- **mergekit**: Goddard et al. (2024) — main toolkit
- **TIES**: Yadav et al. (2023) "Resolving Interference When Merging Models"
- **DARE**: Yu et al. (2024) "Language Models are Super Mario"
- **SLERP**: Shoemake (1985) original; popularized by mergekit
- **Model Soups**: Wortsman et al. (2022)
- **Frankenmerge** / passthrough: community knowledge, less formal lit
- Mercury 自己: 10.5281/zenodo.20313154 (Method)

---

## Onboard 指令給寫者

```
你接手寫 Mercury Paper C: Negative Results in Cross-Fine-Tune Merge。

讀 ~/Desktop/mercury-paper-handoff/paper-C-merge-failures/handoff.md

主任務：寫 4-6 頁 negative-results 形式短文。投 NeurIPS Mech Interp Workshop / BlackboxNLP。

關鍵：你寫的是 systematic failure catalog 這條。是負面結果不是炫耀。
聚焦：8 method comparison + 4 new failure modes + passthrough winner + subspace constraint

風格：norika 風格 + 強調 "we tried, here's why it failed"
所有數字：從 <workstation-A>/frankenstein/ 原始 JSON 重算
引用：每 bibitem 驗 DOI
paper-check.py 8 關全過

先列 outline 給我看再開寫。
```
