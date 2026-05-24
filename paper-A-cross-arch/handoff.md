# Paper A — Cross-Architecture Hot-Dim Geometry

**Suggested title:** *Cross-Architecture Hot-Dim Geometry: Conservation Within GPT-style Decoders, Divergence in Sliding-Window Attention*

**Target venue:** NeurIPS 2026 Mech Interp Workshop (短文 4-6 頁)
**狀態:** **可立即動筆**，cross-arch sweep 結果陸續進來會強化 figure

⚠ **REFRAME (2026-05-22)**：原本只用 qwen anchor 套到別家 → mistral 看起來「0/11」像反例。改框架：**每個 model 都先量自己的 hot-dim geometry，再做 family-level 比較**。Mistral 不是沒有基底，是骨幹結構完全不同（分散全層 + 高位 dim），這是正面發現，且開了 cross-architecture Frankenstein 合併的入口。詳細數字見 `~/Desktop/mercury-paper-handoff/mistral-own-anchors-OUTPUT.txt`。

🆕 **UPDATE (2026-05-23)**：再 +5 模型（ds-r1:70b / olmo2 / gemma2 / granite / yi）→ **18 model / 13 家族**。最大發現：**OLMo2:7b (AllenAI 全公開獨立訓練) 也有 4/11 anchor at top-50 (29.8×)** — 把 anchor 從「家族遺傳」升級為「結構性 universal」。另外 **ds-r1:70b 雖然 base 是 llama 但仍有 3/11 anchor (44.7×) — 蒸餾保留 qwen 結構**。最新表見下面 v3 headline。

🚨 **CRITICAL METHODOLOGY CAVEAT (2026-05-23 PM)** — 必須改寫 framing 才能投：

當同一個模型（OLMo2:7b）的 Tier-A vs Tier-B 結果**完全相反**：
- Tier-A: 4/11 qwen-anchor [11, 25, 279, 382] at 29.8×
- Tier-B: **0/11**，OLMo2 真正熱的 residual stream dim 是 [514, 2273, 2594, 2983, 808, 1066]

→ **Tier-A 的「universal anchor」很可能是 vocab-frequency artifact，不是真的 residual stream feature**。
   原因：Tier-A 用 `dim = tok_id % hidden_size` 把 vocab token 投影到 dim。如果不同模型用相似 tokenizer（共享高頻 BPE token），這些 token 會撞到同樣的 dim 位置，造成假性「跨架構守恆」。

   真正 residual stream feature 必須用 Tier-B (HF `output_hidden_states`) 才能測。

**Paper A 必須改 framing 為兩層次**：

| 層次 | Finding | 強度 |
|---|---|---|
| **Tier-A (vocab-output projection)** | 「LLM 跨家族在 vocab→hidden_size hash 空間有共享熱區，反映 tokenizer 訓練語料的結構性重疊」| 弱 paper，但仍是 paper |
| **Tier-B (residual stream)** | 「LLM 跨家族在中段層 (50-60% 深度) residual stream 有可量化功能對齊，獨立於 tokenizer」| **強 paper** — 真正的 mech interp finding |

**必加對照實驗**：用 random token id 跑 Tier-A protocol，驗證 dim 11 是 vocab artifact 而非真結構。如果 random 也撞到 dim 11，證實 artifact。

**Paper G 核心 thesis 不受影響** — 因為 Paper G 完全建在 Tier-B per-layer fingerprint 上，不依賴 Tier-A anchor。

---

## Headline claim (v3, 2026-05-23)

11 anchor hidden dimensions identified within Qwen2.5 (qwen-family baseline) continue to appear in the top-K hottest dimensions across **18 LLMs from 13 distinct architecture families**, at up to **65× above hypergeometric random baseline**. The strongest cross-architecture conservation comes from **OLMo2 (AllenAI fully-open, completely independent lineage) at 29.8×** and **DeepSeek-R1:70b (distilled into llama-base) at 44.7×**. Conservation also breaks cleanly on **3 specific architectures (mistral-7b/small, yi-1.5)** that exhibit alternative residual stream geometry. **This is the first 18-model cross-architecture hot-dim survey in the mechanistic interpretability literature, all obtained on consumer hardware (Mac mini + DGX Spark).**

---

## 核心 figure 數據

### Figure 1 — Master cross-architecture table (18 models, 13 families)

| Model | Family | params(B) | hidden | top-50 命中 | dims found | 倍數 |
|---|---|---|---|---|---|---|
| qwen2.5:3b | qwen2 | 3.1 | 2048 | 6/11 | – | 22.3× |
| qwen2.5:7b ⭐ | qwen2 | 7.6 | 3584 | **10/11** | – | **65.2×** |
| qwen2.5:32b | qwen2 | 32.8 | 5120 | 6/11 | – | 55.9× |
| qwen2.5-coder:32b | qwen2 | 32.8 | 5120 | 5/11 | – | 46.5× |
| deepseek-r1:7b | qwen2-distill | 7.6 | 3584 | 6/11 | – | 39.1× |
| deepseek-r1:32b | qwen2-distill | 32.8 | 5120 | 5/11 | – | 46.5× |
| **deepseek-r1:70b** ⭐⭐ | **qwen2-distill (on llama base)** | **70.0** | **8192** | **3/11 [11,25,279]** | – | **44.7×** ← scale 王者 |
| llama3.1:8b | llama | 8.0 | 4096 | 3/11 [11,12,25] | – | 22.3× |
| llama3.3:70b | llama | 70.0 | 8192 | 2/11 [11,12] | top-200=4/11 | 29.8× |
| phi3:medium | phi3 | 14.0 | 5120 | 3/11 [12,334,758] | – | 27.9× |
| falcon3:7b | falcon3 | 7.5 | 3072 | 2/11 [11,12] | – | 11.2× |
| **olmo2:7b** ⭐ | **olmo2 (AllenAI fully open)** | 7.0 | 4096 | **4/11 [11,25,279,382]** | top-100=5, top-200=7 | **29.8×** ← 獨立 lineage 鐵證 |
| internlm2:7b | internlm2 | 7.7 | 4096 | 1/11 [334] | – | 7.4× |
| gemma2:9b | gemma2 (Google) | 9.0 | 3584 | 1/11 [476] | – | 6.5× |
| granite3.1:8b | granite (IBM) | 8.0 | 4096 | 0/11 → top-100=2 [25,334] | – | 0× → 7.4× |
| **mistral:7b** | **mistral** | 7.2 | 4096 | **0/11** | none | 0× — boundary |
| **mistral-small3.2** | **mistral (SWA)** | 24.0 | 5120 | **0/11** | none | 0× — boundary |
| **yi-1.5:9b** | **yi-llama** | 9.0 | 4096 | **0/11** | none | 0× — boundary (despite llama-arch tag) |

---

### Per-dim universal anchor ranking

| dim | 跨幾個 model top-50 (/18) | 出現家族 |
|---|---|---|
| **11** | **11/18 (61%)** ⭐ | qwen, qwen-distill, llama, falcon, **olmo2** + 多家 |
| 12 | 8/18 (44%) | qwen, llama, phi3, falcon, **gemma2** |
| 25 | 7/18 (39%) | qwen + ds-distill + **olmo2** |
| 334 | 5/18 (28%) | qwen, phi3, **internlm2**, **granite** — 4 家族 |
| 715 | 5/18 (28%) | qwen-family core |
| 476 | 5/18 (28%) | qwen + **gemma2** |
| 279 | 4/18 (22%) | qwen + ds-distill + **olmo2** |
| 758 | 3/18 (17%) | qwen + phi3 |

→ **dim 11 + dim 12 + dim 25 = 三個跨最廣家族的 universal anchor**，跨 GPT-style 全部家族（qwen / llama / phi3 / falcon / olmo2）— 這是 paper 的核心主張。

---

### ⭐ Mistral 自身 geometry — 不是反例，是另一條設計點（Figure 3 主角）

實際量化 mistral 自己的 hot-dim landscape 後發現它跟 GPT-style decoder 在三個維度都不一樣：

| 性質 | qwen / llama / phi3 (GPT-style) | mistral-small3.2 (SWA) |
|---|---|---|
| 熱層分佈 | 100% 集中在最後一層 | 散落 L0–L31 全 32 層 (L16=27, L1=27, L11=24, L3=24...) |
| anchor dim 範圍 | 低位 11-758 (+ 散落 1483-3055) | 高位 705-3701 |
| per-anchor 層足跡 | 1 層 | 多層 trajectory (dim 3701 ∈ {1,3,5,8,13}) |
| 量化分佈 | 96%+ in q=0 | 67% q=0，剩下散到 q=1..9 |
| own top-11 anchors | {11, 12, 25, 116, 279, 549, 1483, 1548, 2741, 3055, 14} | {705, 1113, 1115, 2043, 2232, 2810, 2812, 3039, 3071, 3072, 3701} |

**Pairwise own-anchor overlap (top-11 each)**：
| | qwen | llama8B | llama70B | phi3 | mistral |
|---|---|---|---|---|---|
| qwen | – | 1 | 0 | 0 | **0** |
| llama8B | 1 | – | 3 | 0 | **0** |
| llama70B | 0 | 3 | – | 1 | **0** |
| phi3 | 0 | 0 | 1 | – | **0** |
| mistral | 0 | 0 | 0 | 0 | – |

→ mistral 跟所有 GPT-style decoder zero overlap，但 GPT-style 彼此之間有微弱 overlap (1-3)

**Interpretation**：
- GPT-style decoder 把 task-relevant 表徵壓縮到「**輸出層 + 低位 dim**」這個窄狹 geometry
- Sliding-window attention (mistral) 不需要這種壓縮 — anchor function 分散在多層多 dim，是「**深 + 寬**」的 trajectory geometry
- 兩種 geometry 都成立，反映不同 design priority

### 🧬 Frankenstein 接口（Discussion 末段點題、留給後續論文）

qwen anchor 占的 (layer × dim) 子空間 = {L_last} × {11..758}
mistral anchor 占的子空間 = {L_0..L_31 分散} × {705..3701}

→ **dim 軸不重疊、layer 軸也不重疊** → 理論上可以建構 qwen × mistral hybrid 而兩家骨幹**互不踩線**。
→ 22 個 disjoint anchor dim union，跟同 arch 的 qwen × qwen-coder (Paper C) 只能在同 11 dim 內互搶，是質性差異。
→ 後續 Paper F 候選：cross-architecture passthrough merge 的可行性研究

### Figure 2 — Dim-by-dim 跨 7 qwen models 存活

| dim | 3b | 7b | 14b | ds-r1:7b | q32b | coder32b | ds32b | 存活/7 |
|---|---|---|---|---|---|---|---|---|
| **11** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **7/7 ⭐** |
| **715** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **7/7 ⭐** |
| 12 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | 6/7 |
| 476 | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ | 6/7 |
| 25 | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | 5/7 |
| 334 | ✓ | ✓ | · | ✓ | ✓ | ✓ | · | 5/7 |
| 481 | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | 5/7 |
| 758 | ✓ | ✓ | · | ✓ | ✓ | · | ✓ | 5/7 |
| 382 | ✓ | ✓ | · | ✓ | · | · | ✓ | 4/7 |
| 279 | ✓ | ✓ | · | ✓ | · | · | · | 3/7 |
| 510 | ✓ | ✓ | · | ✓ | · | · | · | 3/7 |

56/77 = 73% conservation rate. Random baseline: 11 × 100/3584 ≈ 0.31 expected hits per model.

---

## Method 章節要寫的東西

1. **Mercury 觀測 framework**:
   - addressable cell grid `(L, D, Q[, S])`, mmap binary
   - llama_cpp.Llama logits_processor hook
   - Tier-A (output layer only) vs Tier-B (per-layer via HF transformers output_hidden_states)
   - Cell ID 公式

2. **Anchor identification**:
   - Per-model: 10 multilingual prompts × 60 tokens each
   - Top-K hottest cells → projection to dim → hot_dims set
   - 跨 model intersection: anchors = ⋂_models hot_dims[top_K]
   - Triple-confirmed subset: ∩ within-model universal lane

3. **Cross-architecture comparison**:
   - 對每個 architecture 在自己的 hidden_size 內找 top-K hot dims
   - 檢查 qwen-derived 11 anchors 是否在該 architecture 的 top-K 中
   - hypergeometric random baseline: `K × anchor_count / hidden_size`
   - ratio = actual_hits / random_expected

---

## 資料路徑

### Qwen 7 models（已完成）
```
<cloud-server>:<path>
├── heat-summary-1m_1779278023.json       (qwen 7B)
├── heat-summary-3b_1779277545.json       (qwen 3B)
├── heat-summary-deepseek7b_1779385549.json (deepseek-r1 7B)
└── fired-cells-*_p[01-10].bin

<cloud-server>:<path>
├── heat-summary-qwen32b_1779389549.json
├── heat-summary-coder32b_1779389725.json
├── heat-summary-ds32b_1779390753.json
└── fired-cells-*_p[01-10].bin
```

### Anchor-fingerprint matrix (precomputed)
```
<cloud-server>:<path>
  ↑ contains per-(model, prompt) anchor-fire boolean — 直接 import 進 paper
```

### Extraction-meta (49 dim cross-2-size overlap，原始)
```
<cloud-server>:<path>
└── extraction-meta-fixed_1779295092.json
  ↑ 含 hot_dims_7b, hot_dims_3b, 49-dim intersection, 11-dim triple-confirmed
```

### Cross-architecture（持續 update）
```
<cloud-server>:<path>
├── mercury-llama/heat-summary-llama-1779439430.json  (llama3.1:8b)
├── mercury-phi3/heat-summary-phi3-1779453535.json    (phi3:medium)
├── mercury-gemma/   ← pending (gemma 3n via HF)
├── mercury-mistral/ ← pending
├── mercury-llama70b/ ← pending
├── mercury-ds70b/   ← pending
└── mercury-{mixtral,yi,internlm,falcon,qwen-vl,qwen72b}/ ← pending
```

---

## 寫稿 outline 建議

1. **Abstract** (150 字): introduce Mercury observation, claim cross-arch anchor conservation, key numbers (3 archs confirmed at 22-28× random)
2. **Introduction** (1 頁): 
   - Mechanistic interpretability landscape (SAE expensive, narrow)
   - Mercury as low-cost addressable observation
   - This paper's contribution: cross-architecture anchor finding
3. **Method** (1 頁):
   - Mercury cell scheme + observation hook
   - Anchor identification protocol
   - Cross-architecture comparison protocol
4. **Results** (2 頁):
   - Figure 1 cross-arch table
   - Figure 2 dim-by-dim heatmap
   - Statistical significance (hypergeometric test)
   - Qualitative: dim 12 universal across 3 architectures
5. **Discussion** (0.5 頁):
   - Interpretation: anchors as structural transformer feature
   - Relation to feature universality hypothesis (Olah et al.)
   - Practical implications: model lineage detection, anchor-based steering
6. **Limitations** (0.5 頁):
   - 10-prompt sample, Tier-A coverage, post-hoc anchor identification, cross-arch sample still small
7. **Reproducibility**:
   - All code MIT, data CC-BY-4.0 on Zenodo (link Mercury 1-3 DOIs)
   - 4 hours on consumer laptop (RTX 3060 12GB)

---

## Limitations 必寫

1. **Sample size**: 10 prompts per model. 對策：列出 hypergeometric p-value + permutation test
2. **Tier-A coverage**: 只看 output layer。Paper D 補 Tier-B per-layer 觀察
3. **Cross-architecture sample**: 當前 N=3 confirmed (qwen/llama/phi3), pending N≥5
4. **Post-hoc anchor identification**: 11 dim 是事後從 qwen family 找出來，不是 pre-registered。下一篇可 pre-register 在 gemma + mistral 驗證
5. **Hidden-dim alignment assumption**: 假設不同 architecture 的 dim D 是「同一個位置」。沒有 a priori 證據，要在 discussion 講

---

## Related work 引用

- **Sparse autoencoders**: Bricken et al. (2023) "Towards Monosemanticity"; Templeton et al. (2024)
- **Feature universality**: Olah et al. "Circuits" (2020); Gurnee & Tegmark (2023)
- **Activation steering**: Turner et al. "Activation Addition"; Panickssery et al.
- **Cross-model representation alignment**: Kornblith et al.; Bansal et al. "Model Stitching"
- **Mercury 自己**: 10.5281/zenodo.20313154 (Method), 20313748 (Discovery), 20325676 (Scaling), 20313150 (Viewer)

---

## Onboard 指令給寫者

```
你接手寫 Mercury Paper A: Cross-Architecture Anchor Conservation。

讀 ~/Desktop/mercury-paper-handoff/paper-A-cross-arch/handoff.md

主任務：寫 4-6 頁 NeurIPS Mech Interp Workshop short paper。

風格：norika 寫作風格（讀 <cloud-server>:<path>）
所有數字：必須從 <cloud-server>:<path> 原始 JSON 重算，不可從本檔複製
引用：每個 \bibitem 要驗 DOI（feedback_citation_verification_disaster.md 規矩）
查 paper-check.py 8 關全過才能交

先列 outline 給我看再開寫。
```
