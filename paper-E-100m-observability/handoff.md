# Paper E — 100M-Cell 4-Axis Observability on Consumer Hardware

**Suggested title:** *Hundred-Million-Cell Addressable Activation Observability for Large Language Models on Consumer Hardware*

**Target venue:** NeurIPS 2026 Mech Interp Workshop (tools track) / IEEE VIS / IUI demo
**狀態:** **PENDING — 100M grid built, observation 排在 GX10 ds70b 完成之後**

---

## Headline claim (provisional)

Mercury observability scales from million-cell to hundred-million-cell addressable grids without specialized hardware. We construct a 104,857,600-cell 4-axis grid (80 layers × 8192 hidden × 10 quantile × 16 sequence positions) for Llama 3.3 70B, observable on a consumer-grade NVIDIA DGX Spark (GB10, 121GB unified memory) in <X minutes. The sequence-position axis exposes anchor dim activation patterns that vary by token position within the generation — a temporal dimension absent from existing interpretability tooling.

---

## 等齊的資料

| Task | 狀態 | 完成預計 |
|---|---|---|
| 100M grid build | ✅ done (840MB, 3.5s) | – |
| 100M observation on llama 3.3:70B | 🔄 排在 ds70b 之後 | ~3-4 hr |

---

## 核心 figure 候選

### Figure 1 — Grid scale comparison

| Grid | Cells | Storage | Build time |
|---|---|---|---|
| qwen 7B 1M (3-axis) | 1,003,520 | 24MB | 2.5s |
| qwen 7B 10M (4-axis) | 10,035,200 | 240MB | 23s |
| qwen 14B 24M (4-axis) | 24,576,000 | 562MB | 73s |
| **llama 70B 100M (4-axis)** | **104,857,600** | **840MB** | **3.5s** |

### Figure 2 — Seq-pos × dim heatmap

For each anchor dim D, plot heat distribution across 16 sequence position buckets.

Hypothesis: some anchors fire only at early tokens (context-setting); others only at late tokens (generation-time decisions).

### Figure 3 — Coverage analysis

100M grid Tier-A observation fills approximately 1.3% of cells (output-layer slice with seq_pos differentiation). Demonstrate that this 1.3% contains 11 qwen anchors → cross-arch + cross-size + cross-sequence-position validation.

---

## 資料路徑

```
GX10:~/mercury-llama70b-100m/
├── manifest.json
├── catalog.bin (400MB)
├── heat.bin (400MB)
├── observation-{ts}.jsonl  ← pending
├── fired-cells-{ts}_p[01-10].bin  ← pending
└── heat-summary-{ts}.json  ← pending

scripts:
GX10:/tmp/build-grid-100m-llama70b.py
GX10:/tmp/observe-100m-llama70b.py
GX10:/tmp/wait-and-100m.sh  ← queue waiting for ds70b
```

---

## 寫稿 outline (provisional)

1. **Abstract** (150 字): 100M cell grid, 4-axis observability, consumer hardware, novel seq-pos axis
2. **Introduction**: scale of mech interp tooling currently limited; SAE training takes weeks; Mercury enables addressable observation at this scale
3. **Method**:
   - Grid design: (L, D, Q, S) addressing
   - mmap-based storage for petascale (potentially)
   - Llama_cpp `logits_processor` hook + per-token seq_pos tracking
4. **Results**:
   - Grid build / observation timing benchmarks
   - Coverage analysis
   - Anchor seq_pos preference (early vs late token firing)
5. **Discussion**:
   - Implications for online interpretability tooling
   - Scaling: how big can we go? (limit = disk + RAM, not algorithm)
6. **Limitations**:
   - Tier-A only (output layer)
   - Single architecture (llama 70B)
7. **Reproducibility**: full scripts open source, ~3-4 hour runtime on GB10

---

## Onboard 指令給寫者（等資料齊再啟用）

```
你接手寫 Mercury Paper E: 100M-Cell 4-Axis Observability。

⚠ 先檢查 GX10:~/mercury-llama70b-100m/heat-summary-*.json 是否存在。
   未存在則等 100M observation 完成再開始。

讀 ~/Desktop/mercury-paper-handoff/paper-E-100m-observability/handoff.md

聚焦：scale claim + 4-axis novelty + seq_pos finding（如果有的話）。
這篇較 tools/demo 形式。

風格：norika 風格 + 工具實用性
所有數字：從 GX10:~/mercury-llama70b-100m/ 原始 JSON 重算
引用：每 bibitem 驗 DOI
paper-check.py 8 關全過

等資料齊先列 outline 給我看再開寫。
```
