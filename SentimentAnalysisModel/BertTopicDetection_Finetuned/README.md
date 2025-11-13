## 話題分類（BERT 中文基座）

本目錄提供一個使用 `google-bert/bert-base-chinese` 的中文話題分類實現：
- 自動處理本地/緩存/遠程三段式加載邏輯；
- `train.py` 進行微調訓練；`predict.py` 進行單條或交互式預測；
- 所有模型與權重統一保存至本目錄的 `model/`。

參考模型卡片： [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

### 數據集亮點

- 約 **410 萬**條預過濾高質量問題與回覆；
- 每個問題對應一個“【話題】”，覆蓋 **約 2.8 萬**個多樣主題；
- 從 **1400 萬**原始問答中篩選，保留至少 **3 個點贊以上**的答案，確保內容質量與有趣度；
- 除了問題、話題與一個或多個回覆外，每個回覆還帶有點贊數、回覆 ID、回覆者標籤；
- 數據清洗去重後劃分三部分：示例劃分訓練集約 **412 萬**、驗證/測試若干（可按需調整）。

> 實際訓練時，請以 `dataset/` 下的 CSV 爲準；腳本會自動識別常見列名或允許通過命令參數顯式指定。

### 目錄結構

```
BertTopicDetection_Finetuned/
  ├─ dataset/                   # 已放置數據
  ├─ model/                     # 訓練生成；亦緩存基礎 BERT
  ├─ train.py
  ├─ predict.py
  └─ README.md
```

### 環境

```
pip install torch transformers scikit-learn pandas
```

或使用你既有的 Conda 環境。

### 數據格式

CSV 至少包含文本列與標籤列，腳本會嘗試自動識別：
- 文本列候選：`text`/`content`/`sentence`/`title`/`desc`/`question`
- 標籤列候選：`label`/`labels`/`category`/`topic`/`class`

如需顯式指定，請使用 `--text_col` 與 `--label_col`。

### 訓練

```
python train.py \
  --train_file ./dataset/web_text_zh_train.csv \
  --valid_file ./dataset/web_text_zh_valid.csv \
  --text_col auto \
  --label_col auto \
  --model_root ./model \
  --save_subdir bert-chinese-classifier \
  --num_epochs 10 --batch_size 16 --learning_rate 2e-5 --fp16
```

要點：
- 首次運行會檢查 `model/bert-base-chinese`；若無則嘗試本機緩存，再不行則自動下載並保存；
- 訓練過程按步評估與保存（默認每 1/4 個 epoch），最多保留 5 個最近 checkpoint（可通過環境變量 `SAVE_TOTAL_LIMIT` 調整）；
- 支持早停（默認耐心 5 次評估），並在評估/保存策略一致時自動回滾到最佳模型；
- 分詞器、權重與 `label_map.json` 保存到 `model/bert-chinese-classifier/`。

### 可選中文基座模型（訓練前交互選擇）

默認基座：`google-bert/bert-base-chinese`。啓動訓練時，若終端可交互，程序會提示從下列選項中選擇（或輸入任意 Hugging Face 模型 ID）：

1) `google-bert/bert-base-chinese`
2) `hfl/chinese-roberta-wwm-ext-large`
3) `hfl/chinese-macbert-large`
4) `IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese`
5) `IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese`
6) `Langboat/mengzi-bert-base`
7) `BAAI/bge-base-zh`（更適合檢索式/對比學習範式）
8) `nghuyong/ernie-3.0-base-zh`

說明：
- 非交互環境（如調度系統）或設置 `NON_INTERACTIVE=1` 時，會直接使用命令行參數 `--pretrained_name` 指定的模型（默認爲 `google-bert/bert-base-chinese`）。
- 選擇後，基礎模型將下載/緩存至 `model/` 目錄，統一管理。

### 預測

單條：
```
python predict.py --text "這條微博討論的是哪個話題？" --model_root ./model --finetuned_subdir bert-chinese-classifier
```

交互：
```
python predict.py --interactive --model_root ./model --finetuned_subdir bert-chinese-classifier
```

示例輸出：
```
預測結果: 體育-足球 (置信度: 0.9412)
```

### 說明

- 訓練與預測均內置簡易中文文本清洗。
- 標籤集合以訓練集爲準，腳本自動生成並保存 `label_map.json`。

### 訓練策略（簡述）

- 基座：`google-bert/bert-base-chinese`；分類頭維度=訓練集唯一標籤數。
- 學習率與正則：`lr=2e-5`，`weight_decay=0.01`，可在大型數據上微調到 `1e-5~3e-5`。
- 序列長度與批量：`max_length=128`，`batch_size=16`；若截斷嚴重可升至 256（成本上升）。
- Warmup：若環境支持，使用 `warmup_ratio=0.1`；否則回退 `warmup_steps=0`。
- 評估/保存：按 `--eval_fraction` 折算步數（默認 0.25），`save_total_limit=5` 限制磁盤佔用。
- 早停：監控加權 F1（越大越好），默認耐心 5、改善閾值 0.0。
- 單卡穩定運行：默認僅使用一張 GPU，可通過 `--gpu` 指定；腳本會清理分佈式環境變量。


### 作者說明（關於超大規模多分類）

- 當話題類別達到上萬級時，直接在編碼器後接單一線性分類頭（大 softmax）往往受限：長尾類別難學、語義稀疏、新增話題無法增量適配、上線後需頻繁重訓。
- 改進思路（推薦優先級）：
  - 檢索式/雙塔範式（文本 vs. 話題名稱/描述 對比學習）+ 近鄰檢索 + 小頭重排，天然支持增量擴類與快速更新；
  - 分層分類（先粗分再細分），顯著降低單頭難度與計算；
  - 文本-標籤聯合建模（使用標籤描述），提升近義話題的可遷移性；
  - 訓練細節：class-balanced/focal/label smoothing、sampled softmax、對比預訓練等。
- 重要聲明：本目錄使用的“靜態分類頭微調”僅作爲備選與學習參考。對於英文/多語微短文場景，話題變化極快，傳統靜態分類器難以及時覆蓋，我們的工作重點在 `TopicGPT` 等生成式/自監督話題發現與動態體系構建方向；本實現旨在提供一個可運行的基線與工程示例。


