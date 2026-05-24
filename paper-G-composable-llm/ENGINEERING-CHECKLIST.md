# Paper G 工程清單 — 第一階段（為 minimal hybrid 鋪路）

**目標：3-4 週做出第一個 cross-arch hybrid demo**

---

## Phase 1: Tier-B 全層觀測補齊（最大瓶頸）

當前狀態：只有 qwen 7B Tier-B 完成。其他全部 Tier-A only（只有輸出層）。

要做的事：把以下模型用 HF transformers `output_hidden_states=True` 跑一遍 Tier-B：

| Model | 機器 | 預估時間 | 狀態 |
|---|---|---|---|
| qwen2.5:7b | workstation-A | done | ✅ |
| qwen2.5:3b | workstation-A | ~1hr | 🔄 跑中 |
| qwen2.5:14b | workstation-A | ~2hr | 🔄 排隊 |
| llama3.1:8b | workstation-A / gpu-server | ~2hr | ⏳ |
| llama3.3:70b | gpu-server | ~6hr (大模型) | ⏳ |
| phi3:medium | workstation-A / gpu-server | ~3hr | ⏳ |
| mistral-small3.2 | gpu-server | ~4hr | ⏳ |
| mistral:7b | gpu-server | ~2hr | ⏳ |
| falcon3:7b | gpu-server | ~2hr | ⏳ |
| internlm2:7b | gpu-server | ~2hr | ⏳ |
| deepseek-r1:7b | workstation-A | ~2hr | ⏳ |
| deepseek-r1:32b | gpu-server | ~4hr | ⏳ |
| gpt-oss:20b | gpu-server | ~3hr | ⏳ (一旦 pull 完) |

**Total ~30 機器小時**，gpu-server + workstation-A 並行做大概 5-7 天牆鐘時間。

腳本：擴 `<local-tmp>/<script>`（workstation-A 上已有）成 `multi-tier-b.py`，支援批次掃多模型。

---

## Phase 2: Per-layer fingerprint 抽取

對每個模型每一層，從 Tier-B raw output 抽出：

```python
{
  "model": "qwen2.5:7b",
  "layer": 18,
  "fingerprint": {
    "top_hot_dims": [11, 12, 25, 758, ...],   # 該層 top-50 hot dims
    "anchor_presence": [11, 12, 758],           # 11 個 universal anchor 中該層有的
    "selectivity": {
      "chinese": 0.73,                          # 該層對中文輸入的 activation 強度
      "english": 0.41,
      "code": 0.22,
      "math": 0.18,
    },
    "heat_concentration": 0.84,                # 該層 heat 多集中（高=少數 dim 主導，低=散）
    "multi_layer_anchors": [758],              # 跨多層的 anchor 在這層出現
  }
}
```

工程量：寫 `extract-layer-fingerprint.py`，對所有 Tier-B 結果生成 fingerprint database。
產出：`~/Desktop/mercury-paper-handoff/layer-fingerprints/{model}.json`

---

## Phase 3: Cross-model layer matching 矩陣

對所有 (model_A, layer_i) × (model_B, layer_j)，計算 fingerprint similarity (cosine / Jaccard)。

產出 N × N 矩陣（N = 全部模型總層數），找出：
- 「功能等價層族」（不同模型的某幾層 fingerprint 高度相似）
- 「獨家功能層」（某個模型某層 fingerprint 完全孤立）

視覺化：heatmap（橫軸全模型所有層，縱軸同上，色深 = similarity）
→ 這張圖直接變 Paper F 的 main figure。

---

## Phase 4: First minimal hybrid

最保守設定（一定要先成功）：
- 取 qwen 7B 全 28 層當 base
- 偷偷置換 L20-23 為 mistral-small3.2 的 L13-16（基於 fingerprint 相似度選）
- 加 4 個 projection (3584 ↔ 5120) + 4 個 rank-32 adapter
- 凍結所有原 weight，只訓 adapter (5K token 微調)
- 看 5-prompt test 是否 coherent

如果 work：擴到 8 層置換
如果不 work：debug per-layer activation distribution

工具：HuggingFace + PEFT (LoRA) + 自訂 layer-swap monkey patch。

---

## Phase 5: 「按鈕 MVP」介面

最小化版本：
- 一個 Gradio / Streamlit 頁面
- 2 個按鈕：「中文模式」 / 「程式模式」
- 按下 → swap layer set → reload model → 接新對話
- 30 秒 demo 影片放 X / HN

---

## 立刻可以動的事（不用等 SSH 回來）

1. **寫 `extract-layer-fingerprint.py`**（離線就能寫，等 Tier-B 資料來了直接跑）
2. **寫 `cross-model-similarity.py`**（同上）
3. **寫 Paper F outline**（cross-arch detailed comparison）
4. **設計按鈕 UI mockup**（draw.io / figma）

---

## 風險清單

| 風險 | 緩解 |
|---|---|
| Procrustes alignment 在跨 arch 上效果差 | 改用 linear probe + Hungarian matching |
| Adapter 不夠 rank 收不住 | 升 rank 到 64 / 128，必要時做 full MLP adapter |
| Layer swap 後 generation 直接 collapse | 從相鄰層 swap 開始，逐步擴大；參考 Paper C v9 經驗 |
| Tier-B 觀測過慢 | 用 sub-sampling prompts（10 → 3）先看趨勢 |
| GPU-server / workstation-A 不穩定（SSH drop） | 寫 retry 包覆所有遠端任務 |

---

## 第一週具體執行步驟（明天起跑）

1. 寫 `multi-tier-b.py`，部署到 gpu-server + workstation-A
2. workstation-A 補 qwen 3B/14B/llama8B Tier-B
3. gpu-server 補 llama70b / mistral / phi3 / mistral-small Tier-B
4. 等資料同時，本機寫 fingerprint extraction + similarity matrix
5. 第一張 cross-model layer matching heatmap 出爐 → Paper F 第一圖

---

最後一句：vision 不會餓死你，**資料會餵活你**。先補資料。
