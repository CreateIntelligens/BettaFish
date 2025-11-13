# 🔥 MediaCrawler - 自媒體平臺爬蟲 🕷️

<div align="center" markdown="1">
   <sup>Special thanks to:</sup>
   <br>
   <br>
   <a href="https://go.warp.dev/MediaCrawler">
      <img alt="Warp sponsorship" width="400" src="https://github.com/warpdotdev/brand-assets/blob/main/Github/Sponsor/Warp-Github-LG-02.png?raw=true">
   </a>

### [Warp is built for coding with multiple AI agents](https://go.warp.dev/MediaCrawler)


</div>
<hr>

<div align="center">

<a href="https://trendshift.io/repositories/8291" target="_blank">
  <img src="https://trendshift.io/api/badge/repositories/8291" alt="NanmiCoder%2FMediaCrawler | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>

[![GitHub Stars](https://img.shields.io/github/stars/NanmiCoder/MediaCrawler?style=social)](https://github.com/NanmiCoder/MediaCrawler/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/NanmiCoder/MediaCrawler?style=social)](https://github.com/NanmiCoder/MediaCrawler/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/pulls)
[![License](https://img.shields.io/github/license/NanmiCoder/MediaCrawler)](https://github.com/NanmiCoder/MediaCrawler/blob/main/LICENSE)
[![中文](https://img.shields.io/badge/🇨🇳_中文-當前-blue)](README.md)
[![English](https://img.shields.io/badge/🇺🇸_English-Available-green)](README_en.md)
[![Español](https://img.shields.io/badge/🇪🇸_Español-Available-green)](README_es.md)
</div>



> **免責聲明：**
> 
> 大家請以學習爲目的使用本倉庫⚠️⚠️⚠️⚠️，[爬蟲違法違規的案件](https://github.com/HiddenStrawberry/Crawler_Illegal_Cases_In_China)  <br>
>
>本倉庫的所有內容僅供學習和參考之用，禁止用於商業用途。任何人或組織不得將本倉庫的內容用於非法用途或侵犯他人合法權益。本倉庫所涉及的爬蟲技術僅用於學習和研究，不得用於對其他平臺進行大規模爬蟲或其他非法行爲。對於因使用本倉庫內容而引起的任何法律責任，本倉庫不承擔任何責任。使用本倉庫的內容即表示您同意本免責聲明的所有條款和條件。
>
> 點擊查看更爲詳細的免責聲明。[點擊跳轉](#disclaimer)




## 📖 項目簡介

一個功能強大的**多平臺自媒體數據採集工具**，支持小紅書、抖音、快手、B站、微博、貼吧、知乎等主流平臺的公開信息抓取。

### 🔧 技術原理

- **核心技術**：基於 [Playwright](https://playwright.dev/) 瀏覽器自動化框架登錄保存登錄態
- **無需JS逆向**：利用保留登錄態的瀏覽器上下文環境，通過 JS 表達式獲取簽名參數
- **優勢特點**：無需逆向複雜的加密算法，大幅降低技術門檻

## ✨ 功能特性
| 平臺   | 關鍵詞搜索 | 指定帖子ID爬取 | 二級評論 | 指定創作者主頁 | 登錄態緩存 | IP代理池 | 生成評論詞雲圖 |
| ------ | ---------- | -------------- | -------- | -------------- | ---------- | -------- | -------------- |
| 小紅書 | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 抖音   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 快手   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| B 站   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 微博   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 貼吧   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |
| 知乎   | ✅          | ✅              | ✅        | ✅              | ✅          | ✅        | ✅              |



### 🚀 MediaCrawlerPro 重磅發佈！

> 專注於學習成熟項目的架構設計，不僅僅是爬蟲技術，Pro 版本的代碼設計思路同樣值得深入學習！

[MediaCrawlerPro](https://github.com/MediaCrawlerPro) 相較於開源版本的核心優勢：

#### 🎯 核心功能升級
- ✅ **斷點續爬功能**（重點特性）
- ✅ **多賬號 + IP代理池支持**（重點特性）
- ✅ **去除 Playwright 依賴**，使用更簡單
- ✅ **完整 Linux 環境支持**

#### 🏗️ 架構設計優化
- ✅ **代碼重構優化**，更易讀易維護（解耦 JS 簽名邏輯）
- ✅ **企業級代碼質量**，適合構建大型爬蟲項目
- ✅ **完美架構設計**，高擴展性，源碼學習價值更大

#### 🎁 額外功能
- ✅ **自媒體視頻下載器桌面端**（適合學習全棧開發）
- ✅ **多平臺首頁信息流推薦**（HomeFeed）
- [ ] **基於自媒體平臺的AI Agent正在開發中 🚀🚀**

點擊查看：[MediaCrawlerPro 項目主頁](https://github.com/MediaCrawlerPro) 更多介紹


## 🚀 快速開始

> 💡 **開源不易，如果這個項目對您有幫助，請給個 ⭐ Star 支持一下！**

## 📋 前置依賴

### 🚀 uv 安裝（推薦）

在進行下一步操作之前，請確保電腦上已經安裝了 uv：

- **安裝地址**：[uv 官方安裝指南](https://docs.astral.sh/uv/getting-started/installation)
- **驗證安裝**：終端輸入命令 `uv --version`，如果正常顯示版本號，證明已經安裝成功
- **推薦理由**：uv 是目前最強的 Python 包管理工具，速度快、依賴解析準確

### 🟢 Node.js 安裝

項目依賴 Node.js，請前往官網下載安裝：

- **下載地址**：https://nodejs.org/en/download/
- **版本要求**：>= 16.0.0

### 📦 Python 包安裝

```shell
# 進入項目目錄
cd MediaCrawler

# 使用 uv sync 命令來保證 python 版本和相關依賴包的一致性
uv sync
```

### 🌐 瀏覽器驅動安裝

```shell
# 安裝瀏覽器驅動
uv run playwright install
```

> **💡 提示**：MediaCrawler 目前已經支持使用 playwright 連接你本地的 Chrome 瀏覽器了，一些因爲 Webdriver 導致的問題迎刃而解了。
>
> 目前開放了 `xhs` 和 `dy` 這兩個使用 CDP 的方式連接本地瀏覽器，如有需要，查看 `config/base_config.py` 中的配置項。

## 🚀 運行爬蟲程序

```shell
# 項目默認是沒有開啓評論爬取模式，如需評論請在 config/base_config.py 中的 ENABLE_GET_COMMENTS 變量修改
# 一些其他支持項，也可以在 config/base_config.py 查看功能，寫的有中文註釋

# 從配置文件中讀取關鍵詞搜索相關的帖子並爬取帖子信息與評論
uv run main.py --platform xhs --lt qrcode --type search

# 從配置文件中讀取指定的帖子ID列表獲取指定帖子的信息與評論信息
uv run main.py --platform xhs --lt qrcode --type detail

# 打開對應APP掃二維碼登錄

# 其他平臺爬蟲使用示例，執行下面的命令查看
uv run main.py --help
```

<details>
<summary>🔗 <strong>使用 Python 原生 venv 管理環境（不推薦）</strong></summary>

#### 創建並激活 Python 虛擬環境

> 如果是爬取抖音和知乎，需要提前安裝 nodejs 環境，版本大於等於：`16` 即可

```shell
# 進入項目根目錄
cd MediaCrawler

# 創建虛擬環境
# 我的 python 版本是：3.9.6，requirements.txt 中的庫是基於這個版本的
# 如果是其他 python 版本，可能 requirements.txt 中的庫不兼容，需自行解決
python -m venv venv

# macOS & Linux 激活虛擬環境
source venv/bin/activate

# Windows 激活虛擬環境
venv\Scripts\activate
```

#### 安裝依賴庫

```shell
pip install -r requirements.txt
```

#### 安裝 playwright 瀏覽器驅動

```shell
playwright install
```

#### 運行爬蟲程序（原生環境）

```shell
# 項目默認是沒有開啓評論爬取模式，如需評論請在 config/base_config.py 中的 ENABLE_GET_COMMENTS 變量修改
# 一些其他支持項，也可以在 config/base_config.py 查看功能，寫的有中文註釋

# 從配置文件中讀取關鍵詞搜索相關的帖子並爬取帖子信息與評論
python main.py --platform xhs --lt qrcode --type search

# 從配置文件中讀取指定的帖子ID列表獲取指定帖子的信息與評論信息
python main.py --platform xhs --lt qrcode --type detail

# 打開對應APP掃二維碼登錄

# 其他平臺爬蟲使用示例，執行下面的命令查看
python main.py --help
```

</details>


## 💾 數據保存

支持多種數據存儲方式：
- **CSV 文件**：支持保存到 CSV 中（`data/` 目錄下）
- **JSON 文件**：支持保存到 JSON 中（`data/` 目錄下）
- **數據庫存儲**
  - 使用參數 `--init_db` 進行數據庫初始化（使用`--init_db`時不需要攜帶其他optional）
  - **SQLite 數據庫**：輕量級數據庫，無需服務器，適合個人使用（推薦）
    1. 初始化：`--init_db sqlite`
    2. 數據存儲：`--save_data_option sqlite`
  - **MySQL 數據庫**：支持關係型數據庫 MySQL 中保存（需要提前創建數據庫）
    1. 初始化：`--init_db mysql`
    2. 數據存儲：`--save_data_option db`（db 參數爲兼容歷史更新保留）


### 使用示例：
```shell
# 初始化 SQLite 數據庫（使用'--init_db'時不需要攜帶其他optional）
uv run main.py --init_db sqlite
# 使用 SQLite 存儲數據（推薦個人用戶使用）
uv run main.py --platform xhs --lt qrcode --type search --save_data_option sqlite
```
```shell
# 初始化 MySQL 數據庫
uv run main.py --init_db mysql
# 使用 MySQL 存儲數據（爲適配歷史更新，db參數進行沿用）
uv run main.py --platform xhs --lt qrcode --type search --save_data_option db
```


[🚀 MediaCrawlerPro 重磅發佈 🚀！更多的功能，更好的架構設計！](https://github.com/MediaCrawlerPro)


### 💬 交流羣組
- **微信交流羣**：[點擊加入](https://nanmicoder.github.io/MediaCrawler/%E5%BE%AE%E4%BF%A1%E4%BA%A4%E6%B5%81%E7%BE%A4.html)

### 📚 其他
- **常見問題**：[MediaCrawler 完整文檔](https://nanmicoder.github.io/MediaCrawler/)
- **爬蟲入門教程**：[CrawlerTutorial 免費教程](https://github.com/NanmiCoder/CrawlerTutorial)
- **新聞爬蟲開源項目**：[NewsCrawlerCollection](https://github.com/NanmiCoder/NewsCrawlerCollection)
---

### 💰 贊助商展示

<a href="https://h.wandouip.com">
<img src="docs/static/images/img_8.jpg">
<br>
豌豆HTTP自營千萬級IP資源池，IP純淨度≥99.8%，每日保持IP高頻更新，快速響應，穩定連接,滿足多種業務場景，支持按需定製，註冊免費提取10000ip。
</a>

---

<p align="center">
  <a href="https://tikhub.io/?utm_source=github.com/NanmiCoder/MediaCrawler&utm_medium=marketing_social&utm_campaign=retargeting&utm_content=carousel_ad">
    <img style="border-radius:20px" width="500" alt="TikHub IO_Banner zh" src="docs/static/images/tikhub_banner_zh.png">
  </a>
</p>

[TikHub](https://tikhub.io/?utm_source=github.com/NanmiCoder/MediaCrawler&utm_medium=marketing_social&utm_campaign=retargeting&utm_content=carousel_ad) 提供超過 **700 個端點**，可用於從 **14+ 個社交媒體平臺** 獲取與分析數據 —— 包括視頻、用戶、評論、商店、商品與趨勢等，一站式完成所有數據訪問與分析。

通過每日簽到，可以獲取免費額度。可以使用我的註冊鏈接：[https://user.tikhub.io/users/signup?referral_code=cfzyejV9](https://user.tikhub.io/users/signup?referral_code=cfzyejV9&utm_source=github.com/NanmiCoder/MediaCrawler&utm_medium=marketing_social&utm_campaign=retargeting&utm_content=carousel_ad) 或使用邀請碼：`cfzyejV9`，註冊並充值即可獲得 **$2 免費額度**。

[TikHub](https://tikhub.io/?utm_source=github.com/NanmiCoder/MediaCrawler&utm_medium=marketing_social&utm_campaign=retargeting&utm_content=carousel_ad) 提供以下服務：

- 🚀 豐富的社交媒體數據接口（TikTok、Douyin、XHS、YouTube、Instagram等）
- 💎 每日簽到免費領取額度
- ⚡ 高成功率與高併發支持
- 🌐 官網：[https://tikhub.io/](https://tikhub.io/?utm_source=github.com/NanmiCoder/MediaCrawler&utm_medium=marketing_social&utm_campaign=retargeting&utm_content=carousel_ad)
- 💻 GitHub地址：[https://github.com/TikHubIO/](https://github.com/TikHubIO/)

---
<p align="center">
  <a href="https://app.nstbrowser.io/account/register?utm_source=official&utm_term=mediacrawler">
    <img style="border-radius:20px"  alt="NstBrowser Banner " src="docs/static/images/nstbrowser.jpg">
  </a>
</p>

Nstbrowser 指紋瀏覽器 — 多賬號運營&自動化管理的最佳解決方案
<br>
多賬號安全管理與會話隔離；指紋定製結合反檢測瀏覽器環境，兼顧真實度與穩定性；覆蓋店鋪管理、電商監控、社媒營銷、廣告驗證、Web3、投放監控與聯盟營銷等業務線；提供生產級併發與定製化企業服務；提供可一鍵部署的雲端瀏覽器方案，配套全球高質量 IP 池，爲您構建長期行業競爭力
<br>
[點擊此處即刻開始免費使用](https://app.nstbrowser.io/account/register?utm_source=official&utm_term=mediacrawler)
<br>
使用 NSTBROWSER 可獲得 10% 充值贈禮



### 🤝 成爲贊助者

成爲贊助者，可以將您的產品展示在這裏，每天獲得大量曝光！

**聯繫方式**：
- 微信：`relakkes`
- 郵箱：`relakkes@gmail.com`

---

## ⭐ Star 趨勢圖

如果這個項目對您有幫助，請給個 ⭐ Star 支持一下，讓更多的人看到 MediaCrawler！

[![Star History Chart](https://api.star-history.com/svg?repos=NanmiCoder/MediaCrawler&type=Date)](https://star-history.com/#NanmiCoder/MediaCrawler&Date)



## 📚 參考

- **小紅書客戶端**：[ReaJason 的 xhs 倉庫](https://github.com/ReaJason/xhs)
- **短信轉發**：[SmsForwarder 參考倉庫](https://github.com/pppscn/SmsForwarder)
- **內網穿透工具**：[ngrok 官方文檔](https://ngrok.com/docs/)


# 免責聲明
<div id="disclaimer"> 

## 1. 項目目的與性質
本項目（以下簡稱“本項目”）是作爲一個技術研究與學習工具而創建的，旨在探索和學習網絡數據採集技術。本項目專注於自媒體平臺的數據爬取技術研究，旨在提供給學習者和研究者作爲技術交流之用。

## 2. 法律合規性聲明
本項目開發者（以下簡稱“開發者”）鄭重提醒用戶在下載、安裝和使用本項目時，嚴格遵守中華人民共和國相關法律法規，包括但不限於《中華人民共和國網絡安全法》、《中華人民共和國反間諜法》等所有適用的國家法律和政策。用戶應自行承擔一切因使用本項目而可能引起的法律責任。

## 3. 使用目的限制
本項目嚴禁用於任何非法目的或非學習、非研究的商業行爲。本項目不得用於任何形式的非法侵入他人計算機系統，不得用於任何侵犯他人知識產權或其他合法權益的行爲。用戶應保證其使用本項目的目的純屬個人學習和技術研究，不得用於任何形式的非法活動。

## 4. 免責聲明
開發者已盡最大努力確保本項目的正當性及安全性，但不對用戶使用本項目可能引起的任何形式的直接或間接損失承擔責任。包括但不限於由於使用本項目而導致的任何數據丟失、設備損壞、法律訴訟等。

## 5. 知識產權聲明
本項目的知識產權歸開發者所有。本項目受到著作權法和國際著作權條約以及其他知識產權法律和條約的保護。用戶在遵守本聲明及相關法律法規的前提下，可以下載和使用本項目。

## 6. 最終解釋權
關於本項目的最終解釋權歸開發者所有。開發者保留隨時更改或更新本免責聲明的權利，恕不另行通知。
</div>
