# BettaFish - 繁體中文優化分支

## 📢 致謝

本專案為 [666ghj/BettaFish](https://github.com/666ghj/BettaFish) 的個人優化分支。

**特別感謝原作者 [@666ghj](https://github.com/666ghj) 創建並維護這個優秀的多智能體輿情分析系統！**

**完整的專案說明、架構介紹與使用指南，請參閱原專案的 [官方 README](https://github.com/666ghj/BettaFish/blob/main/README.md)**

---

## 🎯 本分支的主要改進

本 Fork 版本基於原專案進行了以下優化：

### 語言優化
- ✅ **全面繁體中文化**：將所有代碼註釋、日誌輸出、UI 介面轉換為繁體中文
- ✅ **保持代碼邏輯一致**：僅改變語言，不改變核心功能

### 系統穩定性改進
- ✅ **修復 logger 相關的 AttributeError**：解決 `ReportEngine/agent.py` 中的 logger 引用問題
- ✅ **整合 upstream 的最新改進**：
  - 流式 LLM 調用（`stream_invoke_to_string`）以改善 UTF-8 處理
  - ForumEngine 的 ERROR 塊過濾機制
  - 增強的節點識別模式（支持多種格式）
  - 改進的配置管理系統（`reload_settings()`）

### 配置優化
- ✅ **Docker Compose 改進**：保留 GPU 配置同時支持環境變量設定
- ✅ **健康檢查優化**：統一的健康檢查 URL 構建與代理設定
- ✅ **更長的超時設置**：調整 ForumEngine 的無活動超時時間

---

## 🚀 快速開始

### 環境需求
- Python 3.10+
- PostgreSQL 15+
- Docker & Docker Compose（可選）

### 安裝步驟

1. **克隆本分支**
```bash
git clone https://github.com/YOUR_USERNAME/BettaFish.git
cd BettaFish
```

2. **配置環境變量**
```bash
cp .env.example .env
# 編輯 .env 填入您的 API 金鑰與配置
```

3. **使用 Docker Compose 啟動（推薦）**
```bash
docker-compose up -d
```

4. **或手動安裝**
```bash
pip install -r requirements.txt
python app.py
```

5. **訪問系統**
打開瀏覽器訪問 `http://localhost:5000`

---

## 📋 主要差異對照

| 項目 | 原專案 | 本分支 |
|------|--------|--------|
| 語言 | 簡體中文 | 繁體中文 |
| Logger 修復 | - | ✅ 已修復 |
| 流式 LLM | 部分支持 | ✅ 全面整合 |
| ForumEngine ERROR 過濾 | - | ✅ 已整合 |
| 配置管理 | 基礎版本 | ✅ reload_settings() |

---

## 🔄 與原專案同步

本分支會定期與上游專案同步最新功能：

```bash
# 添加原專案為 upstream
git remote add upstream https://github.com/666ghj/BettaFish.git

# 拉取最新變更
git fetch upstream
git merge upstream/main
```

---

## 📚 詳細文檔

- **完整功能說明**：請參閱 [原專案 README](https://github.com/666ghj/BettaFish/blob/main/README.md)
- **系統架構**：請參閱 [原專案架構圖](https://github.com/666ghj/BettaFish#%EF%B8%8F-系統架構)
- **安裝指南**：請參閱 [原專案安裝文檔](https://github.com/666ghj/BettaFish#-快速開始)

---

## 📝 授權條款

本專案遵循與原專案相同的授權條款。詳見 [LICENSE](./LICENSE)

---

## 🙏 再次感謝

再次感謝 [@666ghj](https://github.com/666ghj) 與所有為 BettaFish 專案做出貢獻的開發者！

如果您覺得這個專案有幫助，請：
- ⭐ 給原專案 [666ghj/BettaFish](https://github.com/666ghj/BettaFish) 一個 Star
- 💬 加入原專案的技術交流群
- 🤝 參與原專案的開發與討論

---

<div align="center">

**原專案連結**：[https://github.com/666ghj/BettaFish](https://github.com/666ghj/BettaFish)

</div>
