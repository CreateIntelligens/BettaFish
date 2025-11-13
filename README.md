<div align="center">

<img src="static/image/logo_compressed.png" alt="Weibo Public Opinion Analysis System Logo" width="100%">

<a href="https://trendshift.io/repositories/15286" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15286" alt="666ghj%2FBettaFish | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<a href="https://aihubmix.com/?aff=8Ds9" target="_blank"><img src="./static/image/logo_aihubmix.png" alt="666ghj%2FBettaFish | Trendshift" height="40"/></a>&ensp;
<a href="https://lioncc.ai/" target="_blank"><img src="./static/image/logo_loincc.png" alt="666ghj%2FBettaFish | Trendshift" height="40"/></a>

[![GitHub Stars](https://img.shields.io/github/stars/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/stargazers)
[![GitHub Watchers](https://img.shields.io/github/watchers/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/watchers)
[![GitHub Forks](https://img.shields.io/github/forks/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/network)
[![GitHub Issues](https://img.shields.io/github/issues/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/pulls)

[![GitHub License](https://img.shields.io/github/license/666ghj/Weibo_PublicOpinion_AnalysisSystem?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-v1.0.0-green.svg?style=flat-square)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)



[English](./README-EN.md) | [中文文檔](./README.md)

</div>

## ⚡ 項目概述

“**微輿**” 是一個從0實現的創新型 多智能體 輿情分析系統，幫助大家破除信息繭房，還原輿情原貌，預測未來走向，輔助決策。用戶只需像聊天一樣提出分析需求，智能體開始全自動分析 國內外30+主流社媒 與 數百萬條大衆評論。

> “微輿”諧音“微魚”，BettaFish是一種體型很小但非常好鬥、漂亮的魚，它象徵着“小而強大，不畏挑戰”

查看系統以“武漢大學輿情”爲例，生成的研究報告：[武漢大學品牌聲譽深度分析報告](./final_reports/final_report__20250827_131630.html)

查看系統以“武漢大學輿情”爲例，一次完整運行的視頻：[視頻-武漢大學品牌聲譽深度分析報告](https://www.bilibili.com/video/BV1TH1WBxEWN/?vd_source=da3512187e242ce17dceee4c537ec7a6#reply279744466833)

不僅僅體現在報告質量上，相比同類產品，我們擁有🚀六大優勢：

1. **AI驅動的全域監控**：AI爬蟲集羣7x24小時不間斷作業，全面覆蓋微博、小紅書、抖音、快手等10+國內外關鍵社媒。不僅實時捕獲熱點內容，更能下鑽至海量用戶評論，讓您聽到最真實、最廣泛的大衆聲音。

2. **超越LLM的複合分析引擎**：我們不僅依賴設計的5類專業Agent，更融合了微調模型、統計模型等中間件。通過多模型協同工作，確保了分析結果的深度、準度與多維視角。

3. **強大的多模態能力**：突破圖文限制，能深度解析抖音、快手等短視頻內容，並精準提取現代搜索引擎中的天氣、日曆、股票等結構化多模態信息卡片，讓您全面掌握輿情動態。

4. **Agent“論壇”協作機制**：爲不同Agent賦予獨特的工具集與思維模式，引入辯論主持人模型，通過“論壇”機制進行鏈式思維碰撞與辯論。這不僅避免了單一模型的思維侷限與交流導致的同質化，更催生出更高質量的集體智能與決策支持。

5. **公私域數據無縫融合**：平臺不僅分析公開輿情，還提供高安全性的接口，支持您將內部業務數據庫與輿情數據無縫集成。打通數據壁壘，爲垂直業務提供“外部趨勢+內部洞察”的強大分析能力。

6. **輕量化與高擴展性框架**：基於純Python模塊化設計，實現輕量化、一鍵式部署。代碼結構清晰，開發者可輕鬆集成自定義模型與業務邏輯，實現平臺的快速擴展與深度定製。

**始於輿情，而不止於輿情**。“微輿”的目標，是成爲驅動一切業務場景的簡潔通用的數據分析引擎。

> 舉個例子. 你只需簡單修改Agent工具集的api參數與prompt，就可以把他變成一個金融領域的市場分析系統
>
> 附一個比較活躍的L站項目討論帖：https://linux.do/t/topic/1009280

<div align="center">
<img src="static/image/system_schematic.png" alt="banner" width="800">

告別傳統的數據看板，在“微輿”，一切由一個簡單的問題開始，您只需像對話一樣，提出您的分析需求
</div>

## 🪄 贊助商

LLM模型API贊助：<a href="https://aihubmix.com/?aff=8Ds9" target="_blank"><img src="./static/image/logo_aihubmix.png" alt="666ghj%2FBettaFish | Trendshift" height="40"/></a>

所羅門博客LionCC.ai；編程拼車codecodex.ai；編程算力VibeCodingAPI.ai：</a><span style="margin-left: 10px"><a href="https://aihubmix.com/?aff=8Ds9" target="_blank"><img src="./static/image/logo_loincc.png" alt="666ghj%2FBettaFish | Trendshift" height="40"/></a>

## 🏗️ 系統架構

### 整體架構圖

**Insight Agent** 私有數據庫挖掘：私有輿情數據庫深度分析AI代理

**Media Agent** 多模態內容分析：具備強大多模態能力的AI代理

**Query Agent** 精準信息搜索：具備國內外網頁搜索能力的AI代理

**Report Agent** 智能報告生成：內置模板的多輪報告生成AI代理

<div align="center">
<img src="static/image/framework.png" alt="banner" width="800">
</div>

### 一次完整分析流程

| 步驟 | 階段名稱 | 主要操作 | 參與組件 | 循環特性 |
|------|----------|----------|----------|----------|
| 1 | 用戶提問 | Flask主應用接收查詢 | Flask主應用 | - |
| 2 | 並行啓動 | 三個Agent同時開始工作 | Query Agent、Media Agent、Insight Agent | - |
| 3 | 初步分析 | 各Agent使用專屬工具進行概覽搜索 | 各Agent + 專屬工具集 | - |
| 4 | 策略制定 | 基於初步結果制定分塊研究策略 | 各Agent內部決策模塊 | - |
| 5-N | **循環階段** | **論壇協作 + 深度研究** | **ForumEngine + 所有Agent** | **多輪循環** |
| 5.1 | 深度研究 | 各Agent基於論壇主持人引導進行專項搜索 | 各Agent + 反思機制 + 論壇引導 | 每輪循環 |
| 5.2 | 論壇協作 | ForumEngine監控Agent發言並生成主持人總結 | ForumEngine + LLM主持人 | 每輪循環 |
| 5.3 | 交流融合 | 各Agent根據討論調整研究方向 | 各Agent + forum_reader工具 | 每輪循環 |
| N+1 | 結果整合 | Report Agent收集所有分析結果和論壇內容 | Report Agent | - |
| N+2 | 報告生成 | 動態選擇模板和樣式，多輪生成最終報告 | Report Agent + 模板引擎 | - |

### 項目代碼結構樹

```
Weibo_PublicOpinion_AnalysisSystem/
├── QueryEngine/                   # 國內外新聞廣度搜索Agent
│   ├── agent.py                   # Agent主邏輯
│   ├── llms/                      # LLM接口封裝
│   ├── nodes/                     # 處理節點
│   ├── tools/                     # 搜索工具
│   ├── utils/                     # 工具函數
│   └── ...                        # 其他模塊
├── MediaEngine/                   # 強大的多模態理解Agent
│   ├── agent.py                   # Agent主邏輯
│   ├── nodes/                     # 處理節點
│   ├── llms/                      # LLM接口
│   ├── tools/                     # 搜索工具
│   ├── utils/                     # 工具函數
│   └── ...                        # 其他模塊
├── InsightEngine/                 # 私有數據庫挖掘Agent
│   ├── agent.py                   # Agent主邏輯
│   ├── llms/                      # LLM接口封裝
│   │   └── base.py                # 統一的 OpenAI 兼容客戶端
│   ├── nodes/                     # 處理節點
│   │   ├── base_node.py           # 基礎節點類
│   │   ├── formatting_node.py     # 格式化節點
│   │   ├── report_structure_node.py # 報告結構節點
│   │   ├── search_node.py         # 搜索節點
│   │   └── summary_node.py        # 總結節點
│   ├── tools/                     # 數據庫查詢和分析工具
│   │   ├── keyword_optimizer.py   # Qwen關鍵詞優化中間件
│   │   ├── search.py              # 數據庫操作工具集
│   │   └── sentiment_analyzer.py  # 情感分析集成工具
│   ├── state/                     # 狀態管理
│   │   ├── __init__.py
│   │   └── state.py               # Agent狀態定義
│   ├── prompts/                   # 提示詞模板
│   │   ├── __init__.py
│   │   └── prompts.py             # 各類提示詞
│   └── utils/                     # 工具函數
│       ├── __init__.py
│       ├── config.py              # 配置管理
│       └── text_processing.py     # 文本處理工具
├── ReportEngine/                  # 多輪報告生成Agent
│   ├── agent.py                   # Agent主邏輯
│   ├── llms/                      # LLM接口
│   ├── nodes/                     # 報告生成節點
│   │   ├── template_selection.py  # 模板選擇節點
│   │   └── html_generation.py     # HTML生成節點
│   ├── report_template/           # 報告模板庫
│   │   ├── 社會公共熱點事件分析.md
│   │   ├── 商業品牌輿情監測.md
│   │   └── ...                    # 更多模板
│   └── flask_interface.py         # Flask API接口
├── ForumEngine/                   # 論壇引擎簡易實現
│   ├── monitor.py                 # 日誌監控和論壇管理
│   └── llm_host.py                # 論壇主持人LLM模塊
├── MindSpider/                    # 微博爬蟲系統
│   ├── main.py                    # 爬蟲主程序
│   ├── config.py                  # 爬蟲配置文件
│   ├── BroadTopicExtraction/      # 話題提取模塊
│   │   ├── database_manager.py    # 數據庫管理器
│   │   ├── get_today_news.py      # 今日新聞獲取
│   │   ├── main.py                # 話題提取主程序
│   │   └── topic_extractor.py     # 話題提取器
│   ├── DeepSentimentCrawling/     # 深度輿情爬取
│   │   ├── keyword_manager.py     # 關鍵詞管理器
│   │   ├── main.py                # 深度爬取主程序
│   │   ├── MediaCrawler/          # 媒體爬蟲核心
│   │   └── platform_crawler.py    # 平臺爬蟲管理
│   └── schema/                    # 數據庫結構
│       ├── db_manager.py          # 數據庫管理器
│       ├── init_database.py       # 數據庫初始化
│       └── mindspider_tables.sql  # 數據庫表結構
├── SentimentAnalysisModel/        # 情感分析模型集合
│   ├── WeiboSentiment_Finetuned/  # 微調BERT/GPT-2模型
│   ├── WeiboMultilingualSentiment/# 多語言情感分析（推薦）
│   ├── WeiboSentiment_SmallQwen/  # 小參數Qwen3微調
│   └── WeiboSentiment_MachineLearning/ # 傳統機器學習方法
├── SingleEngineApp/               # 單獨Agent的Streamlit應用
│   ├── query_engine_streamlit_app.py
│   ├── media_engine_streamlit_app.py
│   └── insight_engine_streamlit_app.py
├── templates/                     # Flask模板
│   └── index.html                 # 主界面前端
├── static/                        # 靜態資源
├── logs/                          # 運行日誌目錄
├── final_reports/                 # 最終生成的HTML報告文件
├── utils/                         # 通用工具函數
│   ├── forum_reader.py            # Agent間論壇通信
│   └── retry_helper.py            # 網絡請求重試機制工具
├── app.py                         # Flask主應用入口
├── config.py                      # 全局配置文件
└── requirements.txt               # Python依賴包清單
```

## 🚀 快速開始

> 如果你是初次學習一個Agent系統的搭建，可以從一個非常簡單的demo開始：[Deep Search Agent Demo](https://github.com/666ghj/DeepSearchAgent-Demo)

### 環境要求

- **操作系統**: Windows、Linux、MacOS
- **Python版本**: 3.9+
- **Conda**: Anaconda或Miniconda
- **數據庫**: MySQL（可選擇我們的雲數據庫服務）
- **內存**: 建議2GB以上

### 1. 創建環境

#### 如果使用Conda

```bash
# 創建conda環境
conda create -n your_conda_name python=3.11
conda activate your_conda_name
```

#### 如果使用uv

```bash
# 創建uv環境
uv venv --python 3.11 # 創建3.11環境
```

### 2. 安裝依賴包

```bash
# 基礎依賴安裝
pip install -r requirements.txt

# uv版本命令（更快速安裝）
uv pip install -r requirements.txt
# 如果不想使用本地情感分析模型（算力需求很小，默認安裝cpu版本），可以將該文件中的“機器學習”部分註釋掉再執行指令
```

### 3. 安裝Playwright瀏覽器驅動

```bash
# 安裝瀏覽器驅動（用於爬蟲功能）
playwright install chromium
```

### 4. 配置系統

#### 4.1 配置API密鑰

複製一份 項目根目錄 `.env.example` 文件，命名爲 `.env`

編輯 `.env` 文件，填入您的API密鑰（您也可以選擇自己的模型、搜索代理，詳情見根目錄.env.example文件內或根目錄config.py中的說明）：

```python
# MySQL數據庫配置
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "your_username"
DB_PASSWORD = "your_password"
DB_NAME = "your_db_name"
DB_CHARSET = "utf8mb4"

# LLM配置
# 您可以更改每個部分LLM使用的API，只要兼容OpenAI請求格式都可以

# Insight Agent
INSIGHT_ENGINE_API_KEY = "your_api_key"
INSIGHT_ENGINE_BASE_URL = "https://api.moonshot.cn/v1"
INSIGHT_ENGINE_MODEL_NAME = "kimi-k2-0711-preview"
# Media Agent
...
```
推薦LLM API供應商：[推理時代](https://aihubmix.com/?aff=8Ds9)

#### 4.2 數據庫初始化

**選擇1：使用本地數據庫**

> ~~MindSpider爬蟲系統跟輿情繫統是各自獨立的，所以需要再去`MindSpider\config.py`配置一下，複製`MindSpider`文件夾下的 `config.py.example` 文件，命名爲 `config.py`~~  
現版本已更改爲基於環境變量配置，請複製項目根目錄.env.example文件爲.env文件，並在其中填寫各項配置
```bash
# 本地MySQL數據庫初始化
cd MindSpider
# 項目初始化
python main.py --setup

```

**選擇2：使用雲數據庫服務（推薦）**

我們提供便捷的雲數據庫服務，包含日均10萬+真實輿情數據，目前**免費申請**！

- 真實輿情數據，實時更新
- 多維度標籤分類
- 高可用雲端服務
- 專業技術支持

**聯繫我們申請免費雲數據庫訪問：📧 670939375@qq.com**

> 爲進行數據合規性審查與服務升級，雲數據庫自2025年10月1日起暫停接收新的使用申請

### 5. 啓動系統

#### 5.1 完整系統啓動（推薦）

```bash
# 在項目根目錄下，激活conda環境
conda activate your_conda_name

# 啓動主應用即可
python app.py
```

uv 版本啓動命令 
```bash
# 在項目根目錄下，激活uv環境
.venv\Scripts\activate

# 啓動主應用即可
python app.py
```

> 注1：一次運行終止後，streamlit app可能結束異常仍然佔用端口，此時搜索佔用端口的進程kill掉即可

> 注2：數據爬取需要單獨操作，見5.3指引

> 注3：如果服務器遠程部署出現頁面顯示問題，見[PR#45](https://github.com/666ghj/BettaFish/pull/45)

訪問 http://localhost:5000 即可使用完整系統

#### 5.2 單獨啓動某個Agent

```bash
# 啓動QueryEngine
streamlit run SingleEngineApp/query_engine_streamlit_app.py --server.port 8503

# 啓動MediaEngine  
streamlit run SingleEngineApp/media_engine_streamlit_app.py --server.port 8502

# 啓動InsightEngine
streamlit run SingleEngineApp/insight_engine_streamlit_app.py --server.port 8501
```

#### 5.3 爬蟲系統單獨使用

這部分有詳細的配置文檔：[MindSpider使用說明](./MindSpider/README.md)

<div align="center">
<img src="MindSpider\img\example.png" alt="banner" width="600">

MindSpider 運行示例
</div>

```bash
# 進入爬蟲目錄
cd MindSpider

# 項目初始化
python main.py --setup

# 運行話題提取（獲取熱點新聞和關鍵詞）
python main.py --broad-topic

# 運行完整爬蟲流程
python main.py --complete --date 2024-01-20

# 僅運行話題提取
python main.py --broad-topic --date 2024-01-20

# 僅運行深度爬取
python main.py --deep-sentiment --platforms xhs dy wb
```

## ⚙️ 高級配置（已過時，已經統一爲項目根目錄.env文件管理，其他子agent自動繼承根目錄配置）

### 修改關鍵參數

#### Agent配置參數

每個Agent都有專門的配置文件，可根據需求調整，下面是部分示例：

```python
# QueryEngine/utils/config.py
class Config:
    max_reflections = 2           # 反思輪次
    max_search_results = 15       # 最大搜索結果數
    max_content_length = 8000     # 最大內容長度
    
# MediaEngine/utils/config.py  
class Config:
    comprehensive_search_limit = 10  # 綜合搜索限制
    web_search_limit = 15           # 網頁搜索限制
    
# InsightEngine/utils/config.py
class Config:
    default_search_topic_globally_limit = 200    # 全局搜索限制
    default_get_comments_limit = 500             # 評論獲取限制
    max_search_results_for_llm = 50              # 傳給LLM的最大結果數
```

#### 情感分析模型配置

```python
# InsightEngine/tools/sentiment_analyzer.py
SENTIMENT_CONFIG = {
    'model_type': 'multilingual',     # 可選: 'bert', 'multilingual', 'qwen'等
    'confidence_threshold': 0.8,      # 置信度閾值
    'batch_size': 32,                 # 批處理大小
    'max_sequence_length': 512,       # 最大序列長度
}
```

### 接入不同的LLM模型

支持任意openAI調用格式的LLM提供商，只需要在/config.py中填寫對應的KEY、BASE_URL、MODEL_NAME即可。

> 什麼是openAI調用格式？下面提供一個簡單的例子：
>```python
>from openai import OpenAI
>
>client = OpenAI(api_key="your_api_key", 
>                base_url="https://api.siliconflow.cn/v1")
>
>response = client.chat.completions.create(
>    model="Qwen/Qwen2.5-72B-Instruct",
>    messages=[
>        {'role': 'user', 
>         'content': "推理模型會給市場帶來哪些新的機會"}
>    ],
>)
>
>complete_response = response.choices[0].message.content
>print(complete_response)
>```

### 更改情感分析模型

系統集成了多種情感分析方法，可根據需求選擇：

#### 1. 多語言情感分析

```bash
cd SentimentAnalysisModel/WeiboMultilingualSentiment
python predict.py --text "This product is amazing!" --lang "en"
```

#### 2. 小參數Qwen3微調

```bash
cd SentimentAnalysisModel/WeiboSentiment_SmallQwen
python predict_universal.py --text "這次活動辦得很成功"
```

#### 3. 基於BERT的微調模型

```bash
# 使用BERT中文模型
cd SentimentAnalysisModel/WeiboSentiment_Finetuned/BertChinese-Lora
python predict.py --text "這個產品真的很不錯"
```

#### 4. GPT-2 LoRA微調模型

```bash
cd SentimentAnalysisModel/WeiboSentiment_Finetuned/GPT2-Lora
python predict.py --text "今天心情不太好"
```

#### 5. 傳統機器學習方法

```bash
cd SentimentAnalysisModel/WeiboSentiment_MachineLearning
python predict.py --model_type "svm" --text "服務態度需要改進"
```

### 接入自定義業務數據庫

#### 1. 修改數據庫連接配置

```python
# config.py 中添加您的業務數據庫配置
BUSINESS_DB_HOST = "your_business_db_host"
BUSINESS_DB_PORT = 3306
BUSINESS_DB_USER = "your_business_user"
BUSINESS_DB_PASSWORD = "your_business_password"
BUSINESS_DB_NAME = "your_business_database"
```

#### 2. 創建自定義數據訪問工具

```python
# InsightEngine/tools/custom_db_tool.py
class CustomBusinessDBTool:
    """自定義業務數據庫查詢工具"""
    
    def __init__(self):
        self.connection_config = {
            'host': config.BUSINESS_DB_HOST,
            'port': config.BUSINESS_DB_PORT,
            'user': config.BUSINESS_DB_USER,
            'password': config.BUSINESS_DB_PASSWORD,
            'database': config.BUSINESS_DB_NAME,
        }
    
    def search_business_data(self, query: str, table: str):
        """查詢業務數據"""
        # 實現您的業務邏輯
        pass
    
    def get_customer_feedback(self, product_id: str):
        """獲取客戶反饋數據"""
        # 實現客戶反饋查詢邏輯
        pass
```

#### 3. 集成到InsightEngine

```python
# InsightEngine/agent.py 中集成自定義工具
from .tools.custom_db_tool import CustomBusinessDBTool

class DeepSearchAgent:
    def __init__(self, config=None):
        # ... 其他初始化代碼
        self.custom_db_tool = CustomBusinessDBTool()
    
    def execute_custom_search(self, query: str):
        """執行自定義業務數據搜索"""
        return self.custom_db_tool.search_business_data(query, "your_table")
```

### 自定義報告模板

#### 1. 在Web界面中上傳

系統支持上傳自定義模板文件（.md或.txt格式），可在生成報告時選擇使用。

#### 2. 創建模板文件

在 `ReportEngine/report_template/` 目錄下創建新的模板，我們的Agent會自行選用最合適的模板。

## 🤝 貢獻指南

我們歡迎所有形式的貢獻！

### 如何貢獻

1. **Fork項目**到您的GitHub賬號
2. **創建Feature分支**：`git checkout -b feature/AmazingFeature`
3. **提交更改**：`git commit -m 'Add some AmazingFeature'`
4. **推送到分支**：`git push origin feature/AmazingFeature`
5. **開啓Pull Request**

### 開發規範

- 代碼遵循PEP8規範
- 提交信息使用清晰的中英文描述
- 新功能需要包含相應的測試用例
- 更新相關文檔

## 🦖 下一步開發計劃

現在系統只完成了"三板斧"中的前兩步，即：輸入要求->詳細分析，還缺少一步預測，直接將他繼續交給LLM是不具有說服力的。

<div align="center">
<img src="static/image/banner_compressed.png" alt="banner" width="800">
</div>

目前我們經過很長一段時間的爬取收集，擁有了大量全網話題熱度隨時間、爆點等的變化趨勢熱度數據，已經具備了可以開發預測模型的條件。我們團隊將運用時序模型、圖神經網絡、多模態融合等預測模型技術儲備於此，實現真正基於數據驅動的輿情預測功能。

## ⚠️ 免責聲明

**重要提醒：本項目僅供學習、學術研究和教育目的使用**

1. **合規性聲明**：
   - 本項目中的所有代碼、工具和功能均僅供學習、學術研究和教育目的使用
   - 嚴禁將本項目用於任何商業用途或盈利性活動
   - 嚴禁將本項目用於任何違法、違規或侵犯他人權益的行爲

2. **爬蟲功能免責**：
   - 項目中的爬蟲功能僅用於技術學習和研究目的
   - 使用者必須遵守目標網站的robots.txt協議和使用條款
   - 使用者必須遵守相關法律法規，不得進行惡意爬取或數據濫用
   - 因使用爬蟲功能產生的任何法律後果由使用者自行承擔

3. **數據使用免責**：
   - 項目涉及的數據分析功能僅供學術研究使用
   - 嚴禁將分析結果用於商業決策或盈利目的
   - 使用者應確保所分析數據的合法性和合規性

4. **技術免責**：
   - 本項目按"現狀"提供，不提供任何明示或暗示的保證
   - 作者不對使用本項目造成的任何直接或間接損失承擔責任
   - 使用者應自行評估項目的適用性和風險

5. **責任限制**：
   - 使用者在使用本項目前應充分了解相關法律法規
   - 使用者應確保其使用行爲符合當地法律法規要求
   - 因違反法律法規使用本項目而產生的任何後果由使用者自行承擔

**請在使用本項目前仔細閱讀並理解上述免責聲明。使用本項目即表示您已同意並接受上述所有條款。**

## 📄 許可證

本項目採用 [GPL-2.0許可證](LICENSE)。詳細信息請參閱LICENSE文件。

## 🎉 支持與聯繫

### 獲取幫助

- **項目主頁**：[GitHub倉庫](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem)
- **問題反饋**：[Issues頁面](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/issues)
- **功能建議**：[Discussions頁面](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/discussions)

### 聯繫方式

- 📧 **郵箱**：670939375@qq.com

### 商務合作

- **企業定製開發**
- **大數據服務**
- **學術合作**
- **技術培訓**

## 👥 貢獻者

感謝以下優秀的貢獻者們：

[![Contributors](https://contrib.rocks/image?repo=666ghj/Weibo_PublicOpinion_AnalysisSystem)](https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/graphs/contributors)

## 📈 項目統計

<a href="https://www.star-history.com/#666ghj/BettaFish&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=666ghj/BettaFish&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=666ghj/BettaFish&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=666ghj/BettaFish&type=date&legend=top-left" />
 </picture>
</a>

![Alt](https://repobeats.axiom.co/api/embed/e04e3eea4674edc39c148a7845c8d09c1b7b1922.svg "Repobeats analytics image")
