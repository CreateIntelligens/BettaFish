# BettaFish · 多智能體輿情分析系統（繁體中文增強版）

<div align="center">
  <img src="static/image/logo_compressed.png" alt="BettaFish Logo" width="420" />
  <p>Break the echo chamber · Restore real public sentiment</p>
  <a href="https://github.com/CreateIntelligens/BettaFish">Fork 自 666ghj/BettaFish</a>
</div>

---

## 目錄

- [專案簡介](#專案簡介)
- [核心能力地圖](#核心能力地圖)
- [系統運作流程](#系統運作流程)
- [專案結構](#專案結構)
- [快速開始](#快速開始)
- [環境與機密設定](#環境與機密設定)
- [常見操作與工具](#常見操作與工具)
- [疑難排解](#疑難排解)
- [授權與致謝](#授權與致謝)

---

## 專案簡介

**BettaFish** 是一套由多個 AI Agent 協作完成的輿情洞察系統。透過 Query / Media / Insight / Report 四大引擎與 Forum 協作框架，可自動巡檢 30+ 國內外主流社羣平臺與媒體站點，整合文字、圖片、影音與結構化資料，輸出決策級報告。

> Betta 代表「小而強韌、無懼挑戰」，象徵本系統即使部署在個人或企業私有環境，也能具備企業級的洞察戰鬥力。

閱讀系統示例報告：`final_reports/final_report__20250827_131630.html`

---

## 核心能力地圖

| Agent | 主要任務 | 特色工具 | 典型輸出 |
| --- | --- | --- | --- |
| **QueryEngine** | 以關鍵字策略搜尋國內外熱點 | 多搜尋供應商、結果去重、反思迭代 | 話題脈絡、媒體觀點、時間線 |
| **MediaEngine** | 多模態深度理解短影片與圖文 | 視訊轉寫、畫面標註、結構化卡片抽取 | 影音摘要、情緒極化指標 |
| **InsightEngine** | 私有資料庫探勘與交叉分析 | Qwen 關鍵字強化、情感模型、SQL 搜尋 | 深度洞察、私域 FAQ |
| **ReportEngine** | 多輪報告生成與排版 | 範本庫、模板選擇器、HTML Render | 最終 HTML 報告 (支援客製風格) |
| **ForumEngine** | Agent 辯論/協調機制 | 主持人 LLM、節點監看、log diff | 回合摘要、策略調整建議 |
| **MindSpider** | 微博等社羣爬蟲 | 熱榜提取、深度情感、平臺驅動 | 結構化貼文/評論資料 |
| **SentimentAnalysisModel** | 多種情感模型集合 | BERT、GPT-2、Qwen、傳統 ML | 置信度標註、語言自動判斷 |

---

## 系統運作流程

1. **使用者提問**：Flask 主應用接收需求與查詢條件。
2. **三引擎並行**：Query / Media / Insight 同步啟動，先行產出初步觀測。
3. **策略會議**：透過 ForumEngine 主持人，依回應品質調整搜尋策略。
4. **多輪研究**：各 Agent 以 think-reflect 模式進行深挖，必要時調用私有數據或外部爬蟲。
5. **資訊彙整**：ReportEngine 拉取所有 Agent 輸出、論壇紀要與外部附件。
6. **報告生成**：多輪挑選適合模板、先生成大綱，再輸出完整 HTML 與媒體資產。
7. **交付決策**：最終報告寫入 `final_reports/` 並透過前端介面呈現。

---

## 專案結構

```
BettaFish/
├── app.py                      # Flask 主系統入口
├── config.py                   # 全域設定（含 .env 載入）
├── QueryEngine/                # 國際新聞/論壇蒐集 Agent
├── MediaEngine/                # 多模態內容理解 Agent
├── InsightEngine/              # 私域資料庫探勘 Agent
├── ReportEngine/               # 報告產生與模板系統
├── ForumEngine/                # Agent 論壇協作模組
├── MindSpider/                 # 爬蟲系統（微博等）
├── SentimentAnalysisModel/     # 情感分析模型集合
├── SingleEngineApp/            # 各 Agent 的 Streamlit Demo
├── templates/ & static/        # Flask 前端資源
├── final_reports/              # HTML 成品報告輸出
├── scripts/                    # 維運腳本 & 工具
└── requirements.txt            # 依賴清單
```

---

## 快速開始

### 環境需求

- OS：Linux / macOS / Windows 均可
- Python：3.10 以上 (建議 3.11)
- DB：PostgreSQL 15 (預設)；亦支援 MySQL
- 其他：Docker 24+（可選）、Playwright、Conda 或 uv、2GB+ RAM

### A. Docker Compose（推薦）

```bash
git clone https://github.com/CreateIntelligens/BettaFish.git
cd BettaFish
cp .env.example .env      # 填入 LLM / DB / Proxy 設定
docker compose up -d      # 首次啟動會自動建置映像
```

- 鏡像拉取過慢可改用 `docker-compose.yml` 內已註解的鏡像地址。
- `http://localhost:8903` 為預設入口，由 nginx 反向代理 Flask 與各 Agent 服務；健康檢查與代理設定自動套用 env。

## 環境與機密設定

### 必填 .env 參數速覽

| 類別 | 主要參數 | 說明 |
| --- | --- | --- |
| 資料庫 | `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_DIALECT` | 預設 PostgreSQL，若改 MySQL 請同步調整 port 與字元集 |
| Insight Agent | `INSIGHT_ENGINE_API_KEY`, `INSIGHT_ENGINE_BASE_URL`, `INSIGHT_ENGINE_MODEL_NAME` | 支援任何 OpenAI 格式 API，如 Kimi、DeepSeek、SiliconFlow |
| Media Agent | `MEDIA_ENGINE_API_KEY`, `MEDIA_ENGINE_BASE_URL`, `MEDIA_ENGINE_MODEL_NAME` | 影片/圖片理解建議使用多模態模型 |
| Query Agent | `QUERY_ENGINE_API_KEY`, `QUERY_ENGINE_BASE_URL`, `QUERY_ENGINE_MODEL_NAME` | 可切換至最擅長搜尋解讀的模型 |
| Report Agent | `REPORT_ENGINE_API_KEY`, `REPORT_ENGINE_BASE_URL`, `REPORT_ENGINE_MODEL_NAME` | 決策報告建議使用長文本推理模型 |
| 其他 | `PROXY`, `REQUEST_TIMEOUT`, `ENABLE_STREAMING`, `FORUM_HOST_MODEL` | 可依內網環境與成本調整 |


### 進階調校

- `QueryEngine/utils/config.py`：調整反思輪數 (`max_reflections`)、搜尋結果數 (`max_search_results`)。
- `MediaEngine/utils/config.py`：設定多模態搜尋範圍 (`comprehensive_search_limit`)。
- `InsightEngine/tools/sentiment_analyzer.py`：切換情感模型、`confidence_threshold` 等。
- `ForumEngine/monitor.py`：可依團隊需求調整論壇回合上限與空閒時間。

---

## 常見操作與工具

### MindSpider 爬蟲

```bash
cd MindSpider
python main.py --setup                     # 初始化資料庫
python main.py --broad-topic               # 提取熱榜與關鍵詞
python main.py --complete --date 2024-01-20
python main.py --deep-sentiment --platforms xhs dy wb
```

### 報告輸出

- 報告預設儲存於 `final_reports/`，檔名含時間戳。
- 可透過 `ReportEngine/report_template/` 自訂中文或產業專屬模板。
- 若需自動寄送，可在 `ReportEngine/flask_interface.py` 加入 webhook 或郵件邏輯。

### 日誌與監控

- 所有 Agent log 依模組分流存於 `logs/`。
- ForumEngine 會將每輪會議摘要寫入同目錄下的 `forum` log，方便追溯。
- 若需集中式監控，可將 log 目錄掛載到 ELK / Loki。 

---

## 授權與致謝

- 本專案延續原始倉庫 [LICENSE](./LICENSE) 規範。
- 特別感謝原作者 [@666ghj](https://github.com/666ghj) 以及所有貢獻者打造出強大的 BettaFish 生態。

<div align="center">
  <strong>原倉庫連結：</strong> https://github.com/666ghj/BettaFish
</div>
