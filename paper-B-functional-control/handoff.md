# Paper B — Single-Dim Functional Control & Pathway Inhibition

**Target venue:** NeurIPS 2026 Mech Interp Workshop / BlackboxNLP
**狀態:** **可立即動筆**

---

## Headline claim

Subspace-level ablation of single hidden dimensions within a passthrough-merged hybrid LLM can produce reproducible single-dim control of visible model behavior. Specifically:
1. Patching only dim 758 in the last 4 layers triggers a Chinese-narrative-to-codeblock style switch on technical-explanation prompts
2. Adding dim 715 to the patch cancels triggers from dim 279/476 but not from dim 382/758, revealing pathway-specific inhibition structure
3. Layer-norm-only patching (v8) destabilizes generation; only full-subspace rescue (v9) preserves coherence — the "subspace consistency principle"

These results constitute the first single-dim behavioral controller identified via observation-driven ablation (without SAE training, dictionary learning, or gradient-based feature attribution).

---

## 核心 figure 數據

### Figure 1 — 11-dim functional taxonomy

| Category | Dims | Effect on 5-prompt test set |
|---|---|---|
| **Clean controllable trigger** | 279, 382, 476, **758** | Triggers 🦀+codeblock style on prompt 5 (race condition, 中文敘事+技術), no side effects on other prompts |
| **Style trigger + destabilizer** | 11, 334, 510 | Same trigger PLUS prompt 2 (three-段論 述) repetition loop |
| **Pure destabilizer** | 12, 25, 481 | No visible trigger; causes prompt 2 repetition AND prompt 5 instruction-list collapse |
| **Hidden stabilizer / inhibitor** | **715** | Single-dim rescue: no visible change. Combined with trigger dim: pathway-specific inhibition |

### Figure 2 — Single-dim controller demo (dim 758)

| Prompt | v7b baseline (no rescue) | v17 (dim 758 only rescue) | Δ |
|---|---|---|---|
| 1. 龍蝦反駁三句 | 龍蝶 plain CN | identical | none |
| 2. 三段式論述 | plain CN | identical | none |
| 3. code-bug | EN + ``` | identical | none |
| 4. code refactor | EN + ``` | identical | none |
| **5. race condition 中文解釋** | **龍蝶 + plain CN dialogue** | **🦀 emoji + ``` codeblock + bilingual** | **✓ TRIGGERED** |

**Interpretation**: dim 758 mediates a context-conditional behavior — Chinese natural language + technical concept → reach-for-English-formatting. Other prompts don't satisfy the activation context.

### Figure 3 — Inhibitor pathway (dim 715)

| Pair (dim X + dim 715) | Prompt 5 result | Inhibition |
|---|---|---|
| {279, 715} | 龍蝶 plain CN, **no 🦀, no ``` ** | ✅ Full inhibition |
| {382, 715} | 🦀 still appears | ❌ No inhibition |
| {476, 715} | 龍蝶 plain CN, **no 🦀, no ``` ** | ✅ Full inhibition |
| {758, 715} | weird output, partial 🦀 | ⚠ Partial / destabilizing |

**Interpretation**: anchor dims fall into at least 2 subgroups by pathway:
- Pathway A: {279, 476} — suppressible by dim 715
- Pathway B: {382, 758} — immune to dim 715

This is the first published structural relationship between anchor dimensions.

### Figure 4 — Subspace consistency (v8 vs v9 ablation)

| Method | What was patched | Result |
|---|---|---|
| v8 (LN-only rescue) | 9 layer-norm scalars × 11 anchors = 99 overrides | **Generation collapse on 2/5 prompts (repetition loop on P2, instruction-list collapse on P5)** |
| v9 (full subspace rescue) | LN + q/k/v/o + gate/up/down + lm_head = 550+ slice overrides | **No collapse, behavior ≈ baseline on all 5 prompts** |

**Interpretation**: Coder fine-tune's adjustment to anchor dim layer-norm gain is functional adaptation to its modified attention/MLP read pathway, not noise. Scalar patching breaks integration; only holistic subspace patching preserves coherence.

---

## Method 章節要寫的東西

### B.1 Base model setup
- Start from Frankenstein v7b: passthrough merge of 24 layers Qwen2.5-7B-Instruct + 4 layers Qwen2.5-Coder-7B
- Why v7b: empirically determined best passthrough config (see Paper C)

### B.2 Full-subspace single-dim rescue procedure
- Identify target dim D and layer range L
- Override the following tensors at dim D (row or column depending on axis):
  - `model.layers.L.input_layernorm.weight[D]`
  - `model.layers.L.post_attention_layernorm.weight[D]`
  - `model.layers.L.self_attn.{q,k,v}_proj.weight[D, :]`
  - `model.layers.L.self_attn.{q,k,v}_proj.bias[D]`
  - `model.layers.L.self_attn.o_proj.weight[:, D]`
  - `model.layers.L.mlp.{gate,up}_proj.weight[D, :]`
  - `model.layers.L.mlp.down_proj.weight[:, D]`
  - global: `model.norm.weight[D]`, `lm_head.weight[:, D]`
- Per dim × 4 target layers ≈ 50 weight slice copies

### B.3 Evaluation protocol
- 5-prompt test suite (lobster-tone, lobster-structure, code-bug, code-refactor, mixed/race-condition)
- All inference: CPU bfloat16, max_tokens=200, greedy (do_sample=False)
- Each prompt classified by:
  - Language ratio (English / Chinese character proportion)
  - Style markers: 🦀 emoji present, ``` codeblock present, 龍蝶 character present
  - Failure modes: repetition loop, instruction list, gibberish

---

## 資料路徑

```
<workstation-A>/frankenstein/
├── compare-v7-results.json              (BASE/v6/v7a/v7b/v7c)
├── compare-v8-results.json              (LN-only rescue → collapse)
├── compare-v9-results.json              (full subspace rescue → no collapse)
├── compare-v{10-17}-results.json        (selective rescue + single-dim bisection)
├── compare-singledim-{11,12,25,279,334,382,476,510,715}.json   (9-dim sweep)
├── compare-{279_715,382_715,476_715,758_715}-results.json       (inhibitor test)
└── merge-v7.yml etc.

scripts:
<workstation-A>:<path>
├── make-v8-anchor-rescue.py             (v8 LN-only)
├── make-v9-full-subspace-rescue.py      (v9 full subspace)
├── make-v10-selective.py + make-v14-bisect.py + ...
├── run-single-dim.py + sweep-anchor-dims.sh
└── inhibitor-test.sh
```

---

## 寫稿 outline

1. **Abstract** (150 字): single-dim controller demo + inhibitor finding + subspace consistency
2. **Introduction**:
   - Activation steering literature lacks "where to steer" guidance
   - Mercury observation provides anchor candidates
   - This paper validates them via ablation
3. **Method**:
   - Frankenstein v7b setup
   - Full-subspace rescue protocol
   - Evaluation: 5-prompt test + feature markers
4. **Results**:
   - Figure 1 functional taxonomy
   - Figure 2 dim 758 demo (concrete outputs side-by-side)
   - Figure 3 inhibitor pathway
   - Figure 4 subspace consistency
5. **Discussion**:
   - dim 758 as context-conditional feature (vs naive "always-on switch")
   - Pathway-specific inhibition implies sub-cluster structure within anchors
   - Subspace consistency = integration constraint on any patch-based intervention
6. **Limitations**:
   - 5 prompt evaluation set small — could just be noise
   - Single hybrid (v7b) base — other passthrough configs untested
   - dim 758 effect not yet demonstrated to be model-portable (only v7b verified)
7. **Reproducibility**: scripts open source, single Mac mini ~30 min runtime

---

## Limitations 必寫

1. **5-prompt sample**: 太小，rebuttor 會打。對策：擴展到 20+ prompt with 固定 rubric
2. **Single base model**: 只在 v7b 驗證。應在多個 hybrid baseline 上重複（v6, v3 TIES, etc.）
3. **Single-arch validation**: 只在 qwen2.5 family。需在 llama hybrid 上重複（如 llama2-base + llama2-coder）才能宣稱 dim 758 角色是跨架構的
4. **Manual classification**: 5 prompt 的 feature markers 是肉眼判斷。可用 string matching 自動化
5. **「Single-dim」claim 的範圍**: dim 758 alone in last 4 layers triggers something. Doesn't mean dim 758 is THE controller globally — Coder injection at layers 24-27 created the substrate

---

## Related work 引用

- **Activation steering**: Turner et al. (2023) "Activation Addition"; Panickssery et al. (2024) "Steering Llama 2 via CAA"; Subramani et al. "Extracting Latent Steering Vectors"
- **Single-feature interventions**: Hewitt & Manning (2019) probing; Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories"
- **Model merging**: Yadav et al. "TIES" (2023); Yu et al. "DARE"; Wortsman et al. "Model Soups"
- **Mechanistic interp**: Anthropic transformer-circuits 系列
- **Mercury 自己**: 10.5281/zenodo.20313154 等

---

## Onboard 指令給寫者

```
你接手寫 Mercury Paper B: Single-Dim Functional Control & Pathway Inhibition。

讀 ~/Desktop/mercury-paper-handoff/paper-B-functional-control/handoff.md

主任務：寫 4-6 頁 NeurIPS Mech Interp Workshop short paper。

關鍵：你寫的是「functional ablation」這條，不是 cross-arch（那是 Paper A）也不是 merge failures（那是 Paper C）。
聚焦：dim 758 + dim 715 + subspace consistency 三件事。

風格：norika 風格（讀 Mercury-HN-post.md）
所有數字：從 <workstation-A>/frankenstein/ 原始 JSON 重算
引用：每 bibitem 驗 DOI
paper-check.py 8 關全過

先列 outline 給我看再開寫。
```
