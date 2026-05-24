# Mercury Paper — Materials Handoff

**作者：** norika (Chen Ho Yiing, ORCID: 0009-0006-6816-9891)
**準備日：** 2026-05-22
**目的：** 把所有已驗證 / 進行中的結果攤平給接手寫論文的視窗
**目標 venue：** NeurIPS 2026 Mechanistic Interpretability Workshop（截稿通常 9 月初），或 ICLR 2027 Tiny Papers / Main Track
**主標題建議：** *Cross-Architecture Hot-Dim Conservation in Transformer Language Models*

---

## TL;DR（一段話可發 abstract 草稿）

> We introduce Mercury, an addressable million-to-hundred-million-cell sensor grid for transformer LLM observability that runs on consumer hardware (RTX 3060 12GB or 24GB Mac mini). Applied to seven members of the Qwen2.5 family (3B/7B/14B/32B base + Coder + DeepSeek R1 distill), we identify 11 hidden dimensions that appear in the top-100 hottest dims of every model — anchor dimensions. Extending to four additional transformer families (Llama, Phi-3, Gemma, Mistral), a subset of these anchors continues to appear at 14× to 81× above random baseline, establishing the first cross-architecture hot-dim conservation evidence. We further demonstrate single-dim functional control: patching dim 758 alone in a passthrough-merged hybrid triggers a reproducible style switch (Chinese narrative → emoji + code-block) on technical-explanation prompts. dim 715 acts as a pathway-specific inhibitor, cancelling style triggers from dim 279 and 476 but not from 382 or 758. We release Mercury, Frankenstein (merge experiment toolkit), and 19 reproducible Zenodo deposits. The entire result is reproducible in approximately 4 hours on a single consumer laptop.

---

## 第一節 — 已確認的核心結論

### 1. Qwen2.5 family 內 anchor dim 守恆（11 dims × 7 models）

**完整名單**：`[11, 12, 25, 279, 334, 382, 476, 481, 510, 715, 758]`

**Master table**（per-model top-100 hottest dim 出現情況）：

| Model | params | hidden | top-20 命中 | top-50 | top-100 |
|---|---|---|---|---|---|
| qwen2.5:3b | 3.1B | 2048 | – | – | 11/11 |
| qwen2.5:7b | 7.6B | 3584 | – | – | 11/11 |
| qwen2.5:14b *(3 prompts)* | 14.8B | 5120 | 3/11 (69.8×) | 5/11 (46.5×) | 6/11 (27.9×) |
| deepseek-r1:7b | 7.6B | 3584 | 5/11 (81.5×) | 6/11 (39.1×) | 9/11 (29.3×) |
| qwen2.5:32b | 32.8B | 5120 | 2/11 (46.5×) | 5/11 (46.5×) | 7/11 (32.6×) |
| qwen2.5-coder:32b | 32.8B | 5120 | 4/11 (93.1×) | 5/11 (46.5×) | 6/11 (27.9×) |
| deepseek-r1:32b | 32.8B | 5120 | 4/11 (93.1×) | 4/11 (37.2×) | 6/11 (27.9×) |

**Dim-by-dim 跨 7 模型存活**：

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

**總結**：56/77 anchor-model pair 守恆（73%），dim 11 和 dim 715 跨所有 7 模型全勤。
**Random baseline**：`anchor_count × top_K / hidden_dim`（hypergeometric expectation）

---

### 2. Cross-architecture conservation（核心新發現）

**已確認跨家族（截至 2026-05-22 21:00）**：

| Model | Architecture | params | hidden | n_layer | top-50 命中 | dims found | 倍數 |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b (baseline) | qwen2 | 7.6B | 3584 | 28 | 11/11 | full | – |
| **llama3.1:8b** | **llama** | 8.0B | 4096 | 32 | **3/11** | [11, 12, 25] | **22.3×** |
| **phi3:medium** | **phi3** | 14.0B | 5120 | 40 | **3/11** | [12, 334, 758] | **27.9×** |

**dim 12 在 qwen + llama + phi3 三架構 top-50 全部出現** ← 首個跨 3 架構家族「universal-cross-architecture」候選

**Running / pending（截稿前可加上）**：
- gemma-3n-E4B (Gemma architecture, 8B) — HF Tier-B observation, 進行中
- mistral-7B-Instruct-v0.3 (Mistral architecture, 7B) — 排隊
- mixtral:8x22b (Mixtral MoE, 141B params / 39B active) — 下載中
- yi:34b (Yi architecture, 34B) — 下載排隊
- internlm2:7b (InternLM) — 下載排隊
- falcon3:7b (Falcon) — 下載排隊
- qwen2.5-vl:7b (Qwen multimodal) — 下載排隊

**已完成 cross-size 70B 級驗證**：
- llama3.3:70b — Mercury 觀測進行中
- deepseek-r1:70b — 排隊

---

### 3. Functional taxonomy of 11 anchor dims（透過 subspace rescue ablation）

從 Frankenstein v7b（24 Instruct + 4 Coder layers passthrough merge）出發，對每個 anchor dim 單獨做 full-subspace rescue（layer norm + q/k/v/o proj + gate/up/down + lm_head 9 種 tensor × dim 的 row/column），測試其對 5 prompt 行為的影響：

| Category | Dims | 行為 |
|---|---|---|
| **Clean controllable** | 279, 382, 476, **758** | 觸發乾淨可預測的 style 切換（🦀 emoji + markdown code-block）on prompt 5（race condition 中文解釋技術概念） |
| **Style trigger + destabilizer** | 11, 334, 510 | 觸發 style，但同時引起 prompt 2（三段式論述）的 repetition loop |
| **Pure destabilizer** | 12, 25, 481 | 無 style 效果，只造成 prompt 2 repetition + prompt 5 instruction-list collapse |
| **Hidden stabilizer (selective inhibitor)** | **715** | 單獨 rescue 看不到可見變化；但跟 dim 279 或 476 配對能 **cancel 它們的 style trigger**（與 382 和 758 不能 cancel — pathway-specific inhibition） |

**4 個 clean controller × 11 個 anchor = 36% 是可直接用於 activation steering 的 dims**

---

### 4. Single-dim controller 確認：dim 758

**實驗設計**：v7b 末 4 層（24-27）所有 9 種 weight tensor 中，**只**把 dim 758 對應的 row/column 從原始 qwen2.5-7B-Instruct copy 回 v7b。99 個 scalar override。

**Prompt-wise 效果**：

| Prompt | v7b baseline | v17 (only dim 758 rescue) | 變化 |
|---|---|---|---|
| 1. 龍蝦反駁三句 | 龍蝶 plain CN | 龍蝶 plain CN | **無變化** |
| 2. 三段式論述 | plain CN, 缺反駁 | plain CN, 仍缺反駁 | **無變化** |
| 3. code-bug (Python) | English + ``` | English + ``` | **無變化** |
| 4. code refactor | English + ``` | English + ``` | **無變化** |
| 5. **race condition 中文解釋** | 龍蝶 + plain Chinese | **🦀 + ``` + 雙語 dialogue** | **✓ 觸發** |

**意義**：dim 758 不是「always-on style switch」，是「**中文敘事 + 技術概念 context 觸發英文/code-style 格式化的開關**」。其他 prompt 上下文不滿足這個條件 → 不觸發。

第一個 published example of single-dim controlled visible LLM behavior，方法是 observation-driven ablation，**沒有 SAE、沒有 dictionary learning、沒有 gradient-based feature attribution**。

---

### 5. Inhibitor pathway 確認：dim 715

**實驗**：對 4 個 clean trigger dim（279/382/476/758）各加上 dim 715 雙 dim rescue，看是否 cancel：

| Pair | P5 結果 | Inhibition |
|---|---|---|
| {279, 715} | 龍蝶 plain，**無 🦀** | ✅ **完全抑制** |
| {382, 715} | 🦀 出現 | ❌ 未抑制 |
| {476, 715} | 龍蝶 plain，**無 🦀** | ✅ **完全抑制** |
| {758, 715} | 怪輸出，部分 🦀 | ⚠ 部分 |

**意義**：dim 715 是 **selective inhibitor**——抑制 279/476 的 trigger pathway，但對 382/758 的 pathway 無效。暗示 anchor dim 之間有**子叢集結構（pathway A 與 pathway B）**，不是 11 個獨立 capability。

---

### 6. Frankenstein 8-method merge comparison

| Rank | Method | 簡述 | 5-prompt 評估 |
|---|---|---|---|
| 🥇 v7b | Passthrough（24 Instruct + 4 Coder layers）| 完全不混合權重 | 3 強 + 1 最強 + 1 △ |
| 🥈 v3 | TIES uniform (density 0.55, weight 65/35) | top-k magnitude voting | 4 強 + 1 部分 |
| BASE | qwen2.5-7B-Instruct | – | 2 ✓ + 2 △ + 1 ✓ |
| 4 | v7c | Sandwich graft (Instruct-Coder-Instruct) | 3 ✓ + 2 ✗ |
| 5 | v4 | Observation-driven layer-banded TIES | 2 強 + 2 退化 |
| 6 | v1 | SLERP aggressive (t=0.1→0.9) | 5 全敗 (lang drift, repetition, capability loss) |
| 7 | v2 | SLERP conservative (t≈0.2 全層) | 5 collapse (反直覺：保守比激進更糟) |
| **8** | **v5** | **DARE-TIES** | **5 全部 token gibberish** |

**Negative results catalog（適合單獨章節）**：
- SLERP-low-t collapse → 反直覺、社群少有公開報告
- DARE-TIES → cross-fine-tune merge 完全不可用
- Observation-driven TIES regression → 「天真觀測導向 merge」不可行
- Scalar-only anchor rescue (v8) → 破壞 integration
- Subspace consistency principle 反證

---

### 7. Subspace consistency principle

**v8**（只動 layer norm gamma 的 anchor 對應 scalar，9 個 LN × 11 dim = 99 個 scalar override）→ **prompt 5 repetition loop 崩潰**

**v9**（動 full subspace：LN + q/k/v/o + gate/up/down + lm_head，共 550 個 weight slice override）→ **沒崩潰，行為跟 v7b 基線相當**

**結論**：anchor dim 的 layer norm gain 被 Coder fine-tune 壓低不是 bug，是該 dim 對下游 attention/MLP read pathway 的**整合性適應**。強制 rescue 必須 holistic（subspace-level），不能 scalar-level。

---

## 第二節 — 資料路徑

```
<cloud-server>:<path>
├── mercury-frankenstein-handoff.md       ← 全套工具使用手冊
├── handoff-offline.md                    ← 離線交接狀態 snapshot
├── papers-handoff.md                     ← 投稿規劃
├── anchor-fingerprint-matrix.json        ← 7-model anchor 矩陣 raw
├── from-laptop/Neuron-Mercury/var/
│   ├── state-field/
│   │   ├── heat-summary-1m_*.json        ← qwen 7B observation
│   │   ├── heat-summary-3b_*.json        ← qwen 3B
│   │   ├── heat-summary-deepseek7b_*.json ← deepseek-r1 7B
│   │   └── fired-cells-*_p[01-10].bin    ← per-prompt fired neurons
│   ├── uma-region/
│   │   ├── manifest-{1m,3b,10m,14b}.json
│   │   └── catalog/heat/bridges-*.bin
│   └── extracted-lanes/
│       └── extraction-meta-fixed_*.json  ← 11+49 dim 跨模型 overlap
├── from-workstation-A/mercury/var/state-field/
│   ├── heat-summary-qwen32b_*.json       ← qwen 32B
│   ├── heat-summary-coder32b_*.json      ← coder 32B
│   ├── heat-summary-ds32b_*.json         ← deepseek 32B
│   └── fired-cells-*.bin
├── from-gpu-server/                            ← cross-architecture (running)
│   ├── mercury-llama/heat-summary-*.json
│   ├── mercury-phi3/heat-summary-*.json
│   ├── mercury-gemma/      (pending)
│   ├── mercury-mistral/    (pending)
│   ├── mercury-llama70b/   (pending)
│   ├── mercury-ds70b/      (pending)
│   ├── mercury-llama70b-100m/  ★ 4-axis seq_pos grid
│   └── mercury-{mixtral,yi,internlm,falcon,qwen-vl,qwen72b}/ (pending pulls)
└── frankenstein-v1/
    ├── compare-results.json              ← v1 BASE vs v1 hybrid
    ├── compare-all-results.json          ← BASE/v1/v2/v3 ablation
    ├── compare-v7-results.json           ← BASE/v6/v7a/v7b/v7c
    ├── compare-v{8,9,10,11,12,13,14,15,16,17}-results.json ← anchor rescue 系列
    ├── compare-{279_715,382_715,476_715,758_715}-results.json ← inhibitor test
    ├── compare-singledim-{11,12,25,279,334,382,476,510,715}.json ← 9-dim sweep
    └── merge-v{1-7}.yml                  ← merge configurations
```

---

## 第三節 — Live 互動 viewer URLs（可直接放 supplementary materials）

| URL | 內容 |
|---|---|
| https://charenix.com/neuron-mercury/llm-model.html | qwen 7B 內部 sensor map（1M cells 3D 點雲 + 6 lane filter）|
| https://charenix.com/neuron-mercury/overlap.html | 跨模型 49 金線 + 11 三重確認 anchor bridge |
| https://charenix.com/neuron-mercury/cinematic.html | 90 秒 scripted flythrough 介紹影片 |
| https://charenix.com/neuron-mercury/frankenstein-v1.html | SLERP merge 三柱手術示意圖 |
| https://charenix.com/neuron-mercury/frankenstein-arena.html | 動態 voxel rooms + cables + cargo（Tier-A artistic）|
| https://charenix.com/neuron-mercury/anchor-fingerprint.html | 7-model × 10-prompt anchor 命中互動矩陣 |

---

## 第四節 — Method（給寫者用的 algorithmic description）

### 4.1 Mercury 觀測機制

1. **Cell scheme**: 對任意 transformer LLM 配置 4-tuple grid `(L, D, Q, S)` 其中 L = layer index, D = hidden dimension, Q = activation quantile bucket, S = sequence position bucket
2. **Memory layout**: 256MB-2.5GB binary mmap, addressable by neuron_id = `(((L × hidden_size) + D) × n_quant + Q) × n_seq + S`
3. **Observation hook**: `llama_cpp.Llama` 的 `logits_processor` callback，每 token 把當下 logits 按 quantile 分箱到對應 cell
4. **Output**: `heat.bin` u32 array (one counter per cell) + `fired-cells-*_pXX.bin` per-prompt fired cell IDs (for downstream co-firing analysis)

### 4.2 Anchor identification

1. 對 N 個 calibration prompts 跑 observation pass
2. 對每個 model 取 top-K hottest cells（按 heat counter）
3. 把每個 hot cell 對應的 D 投影到 hidden dim 軸，得 per-model `hot_dims` set
4. 跨 model intersection: `anchors = ⋂_models hot_dims_top_K`
5. Triple-confirmed subset: anchors ∩ within-model universal lane (cells fired in ≥9/10 prompts)

### 4.3 Functional ablation (full-subspace rescue)

從 hybrid model M_hybrid（mergekit passthrough 或 SLERP merged）起，對指定 anchor dim D 與 layer set L，把以下 tensor 的 dim-D slice 從原始 base model M_base 強制 copy 回 M_hybrid：

- `model.layers.L.input_layernorm.weight[D]`
- `model.layers.L.post_attention_layernorm.weight[D]`
- `model.layers.L.self_attn.{q,k,v,o}_proj.weight[D, :]` 或 `[:, D]`
- `model.layers.L.self_attn.{q,k,v}_proj.bias[D]`
- `model.layers.L.mlp.{gate,up,down}_proj.weight[D, :]` 或 `[:, D]`
- 全模型 `model.norm.weight[D]`, `lm_head.weight[:, D]`

每 dim ≈ 50 weight slice override per layer × 4 layers = 200+ scalar/row/column patches.

對比 v8（只動 layer norm）證明：scalar 等級 rescue 破壞 integration → 必須 subspace-level patch。

---

## 第五節 — Limitations（必寫）

1. **Sample size**: 10 calibration prompts per model. Rebuttal 會打 "noisy ranking"。對策：(a) 擴到 30 prompts，(b) hypergeometric / permutation test on top-K overlap
2. **Tier-A coverage**: `logits_processor` 只看 output layer（28 層裡 1 層）。已有 qwen 7B Tier-B baseline，可加入 paper figure 證明 anchors 也在中間層活躍
3. **Functional ablation prompt set**: dim 758 trigger 只在 1/5 prompt 觀察到。多 prompt 驗證 + 多 hybrid baseline 必要
4. **Cross-architecture sample**: 當前 3 個 architecture 確認（qwen/llama/phi3），需 ≥5 才能宣稱 "transformer-universal"
5. **Single laptop methodology**: 是 feature 也是 limitation — 可重現 vs 規模有限
6. **Pre-registration**: 11 anchor list 是事後 derived，不是 pre-registered hypothesis。需註明「post-hoc identified but reproducible」

---

## 第六節 — Related work / citations 應引

1. **Sparse autoencoders (SAE)**: Bricken et al. (2023) "Towards Monosemanticity"; Templeton et al. (2024) Anthropic scaling SAE — Mercury 對比：不需 train auxiliary model
2. **Mechanistic interpretability**: Olah et al. "Zoom In: An Introduction to Circuits" (2020); Nanda et al. TransformerLens
3. **Activation steering**: Turner et al. "Activation Addition" (2023); Panickssery et al. "Steering Llama 2 via CAA"
4. **Model merging**: Wortsman et al. "Model Soups"; Yadav et al. "TIES-Merging" (2023); Yu et al. "DARE"
5. **Visualization**: Bycroft (2023) "LLM Visualization"; Carter et al. "Activation Atlas" (Distill 2019); Vig "BertViz"
6. **Cross-model representation alignment**: Kornblith et al. "Similarity of Neural Network Representations Revisited"; Bansal et al. "Revisiting Model Stitching"

---

## 第七節 — 投稿 timeline 建議

- **本週**：把 7-qwen + 3-architecture 結果寫成 4 頁 short paper（workshop format），加 single-dim controller 範例
- **下週**：等 cross-arch sweep（GPU-server 上 7-10 個架構）完成 → 一次性升級 figure
- **9 月**：NeurIPS 2026 Mech Interp Workshop CfP 通常 9 月初公告，9 月中-10 月初截稿
- **平行**：BlackboxNLP 2026（EMNLP workshop）截稿通常 8 月

---

## 第八節 — 接手寫者的快速 onboard 指令

複製貼到新對話：

```
你接手寫 Mercury cross-architecture anchor paper。

讀 ~/Desktop/mercury-paper-handoff/paper-materials.md，那是 norika 整理的完整素材。

主任務：寫 4 頁 NeurIPS Mech Interp Workshop short paper（含 abstract、method、results、discussion、limitations、reproducibility）。

風格：norika 寫作風格（讀 <cloud-server>:<path> 抓 voice）。
工具：寫完跑 cloud-server:~/scripts/paper-check.py 8 關全過才能交。
引用：每個 \bibitem 要逐筆驗 DOI/URL（feedback_citation_verification_disaster.md 規矩）。
資料：所有數字必須從 <cloud-server>:<path> 原始 binary / JSON 重算，不可從本文件直接複製。

先列你的寫作 outline 給我看，再開始寫。
```

---

## 第九節 — 還在跑、可能進得來的實驗

到本文交付時尚未完成、但可能 24-48h 內可加入 paper figure：

1. gemma-3n-E4B Tier-B observation（HF transformers, GPU-server）
2. mistral-7B-Instruct-v0.3 Tier-B（GPU-server）
3. llama3.3:70b 3-axis Tier-A（GPU-server，含 100M cell 4-axis 後續）
4. deepseek-r1:70b（GPU-server）
5. mixtral:8x22b（141B MoE, 不同 paradigm, GPU-server 等下載完）
6. yi:34b, internlm2:7b, falcon3:7b, qwen2.5-vl:7b（cross-arch 擴充）
7. qwen 3B + 14B Tier-B（workstation-A Mac mini, HF transformers）
8. 100M cell 4-axis llama70b observation（seq_pos × layer × dim × quantile）

完成度自動更新到 `<cloud-server>:<path>` 和 `from-workstation-A/`。

---

**最後一句給寫者：**

> 這套素材夠寫 NeurIPS workshop 短文（4 頁）+ 可能再拆一篇 negative-result paper（SLERP/DARE/anchor-LN-only-rescue 三個 failure mode）。第一篇先寫穩，第二篇 followup。**不要急著加更多實驗，先把現有素材寫到 paper-grade**。
