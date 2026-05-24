# Paper D — Tier-B Per-Layer Anchor Evolution Across Model Scale

**Target venue:** NeurIPS 2026 Mech Interp Workshop
**狀態:** **PENDING — Tier-B sweep 在 workstation-A 上跑中（qwen 3B + 14B），等齊後可動筆**

---

## Headline claim (provisional)

Using HuggingFace transformers `output_hidden_states` to capture per-layer residual stream activations during inference, we map the layer-wise distribution of 11 anchor dimensions across the Qwen2.5 model size scale (3B / 7B / 14B / 32B). Anchor dims are not uniformly distributed across layers; they show characteristic concentration patterns that shift with model depth. dim 11 and dim 715 (the two cross-7-model universal anchors) appear at <claim TBD after sweep finishes> layer ranges.

---

## 等齊的資料

| Model | Tier-B 狀態 | 完成預計 |
|---|---|---|
| qwen2.5:7b | ✅ 完成 (workstation-A) | already done |
| qwen2.5:3b | 🔄 workstation-A sweep 進行中 | ~1-2 hr |
| qwen2.5:14b | 🔄 workstation-A sweep 排隊 | ~2-3 hr |
| qwen2.5:32b | ⚠ 32B HF fp16 = 64GB，workstation-A 24GB unified 太緊。可能要 bnb 4bit 才行 | uncertain |
| qwen2.5-coder:32b | 同上 | uncertain |
| deepseek-r1:32b | 同上 | uncertain |

---

## 核心 figure 候選

### Figure 1 — Per-layer anchor presence heatmap

Y-axis: layer index (0 to n_layer)
X-axis: 11 anchor dim positions
Color: cell heat (per-layer hot intensity at that dim)
Panel-per-model: 3B (36 layers) / 7B (28 layers) / 14B (48 layers) / 32B (64 layers)

→ Visually compelling: do anchors live in same relative layer position across sizes?

### Figure 2 — Anchor layer trajectory

For each of the 11 anchor dims, plot the layer where it first becomes "hot" (heat > threshold) across the 4 sizes.

Hypothesis: universal anchors (dim 11, 715) become hot at relatively shallow layers; specialized anchors (dim 758) only at deeper layers.

### Figure 3 — Anchor → functional dimension link

Combine with Paper B's dim 758 finding: show that dim 758 layer-wise activation profile matches its functional role (Chinese-tech-context style trigger lives in upper layers).

---

## 資料路徑

```
<workstation-A>:<path>        ← already done
├── heat-summary-qwen7b-tierB-*.json
├── heat-qwen7b-tierB-*.bin
└── fired-cells-qwen7b-tierB-*.bin

<workstation-A>:<path>         ← pending
├── mercury-qwen3b-tierB/  ← 跑中
├── mercury-qwen14b-tierB/ ← 排隊
└── (32B variants if memory allows)
```

---

## 寫稿 outline (provisional)

1. **Abstract** (150 字): Tier-B observation reveals anchor layer-distribution structure
2. **Introduction**: 
   - Tier-A (output layer only) misses where anchors actually fire across the network
   - Tier-B via HF transformers `output_hidden_states` exposes all layers
3. **Method**:
   - HF transformers hook protocol
   - 4-tuple cell grid (L, D, Q, S) with full layer coverage
   - Hot-cell ranking criterion
4. **Results**:
   - Per-layer anchor heatmap (figure 1)
   - Anchor layer trajectory across model sizes (figure 2)
   - Functional anchor (dim 758) layer profile matches its observed behavior (figure 3)
5. **Discussion**: 
   - Anchors as deep-layer features (post-attention adjustment)
   - Comparison to Anthropic's SAE feature layer mapping
6. **Limitations**: 
   - Limited to ≤ 14B (memory constraint on consumer hardware)
   - Single-architecture (qwen2 family only)
7. **Reproducibility**: scripts open source, workstation-A Mac mini + HF token

---

## Onboard 指令給寫者（等資料齊再啟用）

```
你接手寫 Mercury Paper D: Tier-B Per-Layer Anchor Evolution。

⚠ 先檢查 <workstation-A>:<path> 是否齊（至少 qwen 3B + 14B 完成）。
   未齊則等齊再開始。

讀 ~/Desktop/mercury-paper-handoff/paper-D-tier-b-layer-map/handoff.md

聚焦：per-layer anchor distribution，搭配 Paper B 的功能解釋

風格：norika 風格
所有數字：從 <workstation-A>:<path>*-tierB/ 原始 JSON 重算
引用：每 bibitem 驗 DOI
paper-check.py 8 關全過

等資料齊先列 outline 給我看再開寫。
```
