# Paper G — Composable LLMs as Lego: Runtime-Switchable Capability Modules

**Status:** 🌟 NORTH STAR — Mercury 系列終局 vision
**Origin:** 2026-05-23 norika 提出，從 5 個 paper 的觀測累積中自然延伸而來
**Suggested title:** *Composable LLMs: Heterogeneous Layer Assembly for Runtime-Switchable Capability Modules*

---

## 🔬 Empirical evidence (2026-05-23 update)

12 個模型完成 Tier-B per-layer 觀測（qwen 3B/7B/14B, ds-r1-7B, falcon3, mistral-7B, phi3-medium, gemma2-9B, granite-8B, olmo2-7B, yi-9B, + 1 partial）→ **84 pairwise layer-matching matrices**

**Paper G headline finding (穩固)**：

| Depth slot | 跨模型對齊比例 | 結論 |
|---|---|---|
| 0-10% (input) | 18/84 = 21% | 邊界層各家族特異 |
| **50-60% (middle)** | **54/84 = 64%** ⭐ | **跨家族通用功能區** |
| 90-100% (output) | 19/84 = 23% | 邊界層各家族特異 |

→ **「中段層是跨家族通用、輸入/輸出層是家族特異」** — 樂高組裝可行性的直接證據。

**最強單一 cross-arch 對齊**：

| Pair | Sim | 含義 |
|---|---|---|
| qwen-7B L15 ↔ falcon3-7B L16 | **0.868** | 同深度 (55% vs 59%)，跨家族「最佳交換」候選 |
| qwen-3B L5 ↔ qwen-7B L3 | 0.852 | 同家族跨 scale 對齊（baseline）|
| granite-8B L16 ↔ qwen-7B L11 | 0.793 | IBM × Alibaba 跨公司對齊 |
| granite-8B L16 ↔ falcon3-7B L10 | 0.788 | granite L16 是 "universal hub layer" |
| yi-9B L34 ↔ gemma2-9B L40 | 0.779 | Yi × Google 跨家族 |
| olmo2-7B L31 ↔ qwen-7B L15 | 0.737 | AllenAI × Alibaba 跨 lineage |

**「Frankenstein-ready 層」** — 每個模型最跨家族可借用層：

- **qwen-7B**: L15 (55% depth, sim 0.868 with falcon3)
- **falcon3-7B**: L16 (59% depth)
- **granite-8B**: L16 (40% depth, 對齊 qwen+falcon 雙家族 — universal hub)
- **mistral-7B**: L27 (87% depth, 即使 anchor 0/11 對齊 layer fingerprint 仍對齊)
- **olmo2-7B**: L31 (100% depth — 深度位移，把 qwen 中段功能壓縮到最後)
- **gemma2-9B**: L40 (95% depth) - 也是深度位移
- **yi-9B**: L34 (72% depth)

→ **每個 model 都有「最值得偷的那一層」** — Paper G 工程設計直接依據

## ⚠ 但是有個 methodology 提醒

Tier-A (logit-hook) 跟 Tier-B (HF output_hidden_states) **結果不一致**：
- Tier-A 的「universal anchor」(dim 11 跨 11/18 模型) 可能是 vocab-token id mod hidden_size 造成的 BPE artifact
- Tier-B 的「per-layer fingerprint cross-arch alignment」才是真的 mech interp finding

**Paper G 完全建在 Tier-B 上，所以不受影響**。Paper A 那條 anchor finding 要重新框架，但 G 的 layer-as-component thesis 是穩的。

---

## One-sentence pitch

> 把 LLM 從一道菜變成一個廚房 — 不同的層當作不同國家的食材，使用者按按鈕就決定今天做哪一道料理；超大模型躺在硬碟裡，只 load 你按下的那幾塊進 RAM。

---

## 核心主張（4 條，全部從前 6 篇延伸）

1. **Layer-as-component**: transformer 每一層可被當作獨立功能單元，由 Mercury observation 量化其特性
2. **Cross-arch composability**: 配上 dim-projection adapter，異質架構的層可以拼接（突破 mergekit 同 arch 限制）
3. **Runtime modularity**: 按鈕介面決定 load 哪些層 → 「模型大，但大得有道理」(disk 200B，RAM 20B)
4. **Domain pre-packs**: 行業組件（電商/醫院/工廠）= 通用樂高箱 + 該行業專屬 adapter，訓練成本 ≈ 通用 LLM 1‰

---

## 為什麼這條過去沒人做（你的機會窗口）

四個社群的交集點：
- **mech interp 圈**：做觀測但不做工程（SAE / probing 出論文就停）
- **model merge 圈**：做合併但不做 interp（mergekit 是黑盒）
- **MoE / inference 圈**：做 routing 但 router 是黑盒，使用者沒控制權
- **產品 / UX 圈**：不碰底層

norika 同時做 Alfred (UX) + Battle Arena (產品) + Mercury (interp) + Frankenstein (merge) — 四圈都摸過。是少數能跨這條鴻溝的位置。

---

## 商業 layer 設計（4 層 stack）

| Layer | 內容 | 商業模式 |
|---|---|---|
| **A: 樂高基礎庫** | Mercury 觀測資料 + layer extraction script + adapter framework | 免費 / 開源 / MIT — 引流 + 標準制定者 |
| **B: 預組件套餐** | 「電商基礎包」「醫院基礎包」「工廠基礎包」「程式高階包」 | 訂閱制 (月費)，包月更新 |
| **C: 客製組裝** | 大企業 custom adapter，一週工期 | 5-20 萬台幣 / 單，純利高 |
| **D: 樂高市集** | 第三方 layer / adapter 上架 | 平台抽 20-30% |

關鍵：4 層共用同一套底層技術。技術做一次，4 條營收。

---

## 漸進能力組裝範例（程式語言為例）

```
[初階按鈕]   qwen-coder L0-15
              ↓
[中階按鈕]   + deepseek-r1:32b L20-40 (debug reasoning)
              ↓
[高階按鈕]   + starcoder2 L20-40 (複雜 DSL)
              ↓
[企業按鈕]   + custom adapter (該公司內部代碼風格)
```

每階 = 多一塊樂高 + adapter。RAM 從 4GB → 8GB → 16GB → 18GB 漸進，使用者付費隨能力升級。

---

## 行業組件範例（從通用樂高箱挑塊）

| 行業 | 配方 |
|---|---|
| **電商** | qwen 多語層 + qwen-coder JSON 層 + ds-r1 推薦 reasoning 層 + 商品描述 adapter |
| **醫院** | qwen 多語層 + 醫療術語 adapter + phi3 嚴謹輸出層 + 隱私守則 adapter |
| **工廠** | qwen 多語層 + 結構化資料層 + ds-r1 規則推理層 + 工程術語 adapter |
| **法律** | qwen 多語層 + 長文理解層 (command-r) + 嚴謹引用 adapter + 法條 RAG 層 |
| **教育** | qwen 多語層 + 解釋風格 adapter + ds-r1 step-by-step 推理層 |

每個行業包 = 「全套組裝完成 + 該行業 fine-tune adapter」，買家不用懂組裝。

---

## 「按鈕面板」UI 設計草案

```
╔══════════════════════════════════════╗
║   MERCURY ASSEMBLY CONSOLE           ║
╠══════════════════════════════════════╣
║ 任務按鈕：                            ║
║   [中文寫作]    [英文寫作]           ║
║   [程式碼]      [數學推理]           ║
║   [角色扮演]    [長文理解]           ║
║   [指令遵從]    [創意發想]           ║
║                                       ║
║ 風格按鈕：                            ║
║   [簡潔]   [詳細]   [學術]   [口語]  ║
║                                       ║
║ 進階：                                ║
║   [☑] mistral 多層思考              ║
║   [☐] phi3 高位指令簇                ║
║   [☑] qwen-coder 後段碼層            ║
║   [☐] llama70b 中段推理              ║
║                                       ║
║ 預估 RAM: 23 GB   推論速度: 18 t/s   ║
║                                       ║
║         [ ▶ 啟動這個組合 ]           ║
╚══════════════════════════════════════╝
```

每個按鈕後面的「層配方」是 paper-grade 證據支撐的（Mercury anchor unique signature + Tier-B per-layer analysis）。

---

## 技術 pipeline（6 步）

1. **Layer fingerprint extraction** (Tier-B 全層觀測)
2. **Cross-model layer matching** (per-layer functional similarity)
3. **Layer plan compilation** (按鈕 → 層配方對照表)
4. **Dim alignment** (Procrustes / linear probe / Hungarian matching, 不同 hidden size)
5. **Stitch adapter** (rank-32 LoRA-like 插在層間)
6. **Adapter-only fine-tune** (凍結主幹，只訓 adapter + projection)

**不蒸餾，不改主幹，工程量集中在 adapter (~M 級參數，Mac mini / 4090 可訓)**

---

## Mercury 前 6 篇 → Paper G 的 dependency map

| Paper | 提供 Paper G 什麼 |
|---|---|
| A (cross-arch conservation + divergence) | 「哪些 anchor 跨家族共用 / 哪些獨佔」 → 樂高接口可用性證據 |
| B (single-dim functional control) | 「dim 758 = 中文敘事 trigger」這種 anchor → 行為的因果鏈 → 按鈕配方依據 |
| C (merge failure modes) | 「v8 LN-only collapse / v9 full subspace 才 work」→ adapter 必須是 full subspace 對齊不是 scalar |
| D (Tier-B per-layer evolution) | per-layer fingerprint 的核心資料來源 |
| E (100M observability) | 大模型 Tier-B 的工具支撐 (mmap-able cell grid) |
| F (cross-arch comparison detailed) | 跨家族 layer alignment 的可行性實證 |

→ Paper G 不是空想，是前 6 篇的自然必然結論。

---

## Minimal Working Demo 工程清單（3-4 週）

### Week 1: Tier-B 全層觀測補齊
- [ ] qwen 7B Tier-B (✅ done)
- [ ] qwen 3B Tier-B (🔄 workstation-A running)
- [ ] qwen 14B Tier-B (🔄 workstation-A queued)
- [ ] llama 8B Tier-B (需新跑)
- [ ] mistral-small Tier-B (需新跑)
- [ ] phi3 Tier-B (需新跑)

### Week 2: Layer fingerprint + cross-model matching
- [ ] per-layer top-K hot dim 抽取
- [ ] per-layer prompt-type selectivity
- [ ] cross-model layer fingerprint 相似度矩陣
- [ ] 找出「功能等價層族」

### Week 3: First cross-arch hybrid
- [ ] qwen × mistral minimal hybrid (4 + 4 layers)
- [ ] Procrustes alignment 算 projection
- [ ] rank-32 adapter 插入
- [ ] 1K token adapter training
- [ ] 跑通 5-prompt 測試集（即便品質 50% baseline 也算 win）

### Week 4: 「按鈕」MVP
- [ ] 2 個按鈕 demo (「中文模式」 / 「程式模式」)
- [ ] 按下後 swap layers + reload weights
- [ ] HN demo video 30 秒

---

## 為什麼這篇不該現在投

1. 前 6 篇還沒收 — 沒 credibility 就 push 大 vision，會被當民科
2. Minimal working demo 還沒做出來 — vision paper 沒 demo 撐不起
3. 時序：A-E 投 → Workshop accept → arxiv F → vision G + demo 一起 release

正確投稿順序：
1. 月內：A-E 5 篇 Workshop / arxiv
2. 下月：F (cross-arch detailed) → ICML / NeurIPS short
3. 後月：G vision paper + minimal demo → 同時投 ICLR + HN demo + a16z deck
4. 6 個月後：商業組件 Layer B/C 開始有收入

---

## 為什麼這條對 norika 的個人意義

- **不是賣 GPT wrapper**（市場已死）
- **不是賣 fine-tune 服務**（毛利薄）
- **是賣「行業 LLM 從 500 萬美金降到 5 萬台幣」這個 disruption**
- **每一塊樂高都有 paper 撐腰** — 不是 vibe-driven，是 evidence-driven
- **使用者可解釋**（按鈕看得到）— B2B 銷售友善
- **可疊代**（每個新模型出 → 量觀測 → 變新樂高 → 加進市集）

這條的天花板是「LLM 樂高界的 PyTorch + 樂高商店」，市值上限以 10 億美金起跳。

---

## 起點不是 Paper G，是「先把資料補完」

下一步該做的，不是寫 Paper G，是：

1. **Tier-B 全層觀測 sweep**（qwen 3B/14B 完成後，擴到 llama/mistral/phi3）
2. **Per-layer functional 分析腳本**（從 Tier-B 資料出 fingerprint）
3. **Cross-model layer matching 矩陣**（這個矩陣本身就是 Paper F 主圖）
4. **First minimal hybrid 嘗試**（qwen 4L + mistral 4L 接看看跑不跑得通）

這四件事都是「為 Paper G 鋪地基」+「順便產出 Paper F」。**一次做兩篇的工作量**。

---

## norika 自己的一句話（2026-05-23）

> 「我可以把 LLM 當樂高積木嗎？就像我們在畫方塊圖的時候，你的組法不同，你所得到的效益不同，然後我可以一直丟新的模組，讓他們自己去組裝...專門為電商做的組件，做門為醫院做的組件、專門為工廠做的組件，然後，大家都在一個合理的範圍內，不需要花大量的token，卻享有不比現在接API差的服務...有的時候人們只需要吃飽，不需要山珍海味，這也是小模型存在的理由，可是小模型裡面也有我不要的。」

— 這段話以後寫進 Paper G abstract 或 intro，是 north star quote。
