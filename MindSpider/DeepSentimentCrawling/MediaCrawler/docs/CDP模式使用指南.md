# CDP模式使用指南

## 概述

CDP（Chrome DevTools Protocol）模式是一種高級的反檢測爬蟲技術，通過控制用戶現有的Chrome/Edge瀏覽器來進行網頁爬取。與傳統的Playwright自動化相比，CDP模式具有以下優勢：

### 🎯 主要優勢

1. **真實瀏覽器環境**: 使用用戶實際安裝的瀏覽器，包含所有擴展、插件和個人設置
2. **更好的反檢測能力**: 瀏覽器指紋更加真實，難以被網站檢測爲自動化工具
3. **保留用戶狀態**: 自動繼承用戶的登錄狀態、Cookie和瀏覽歷史
4. **擴展支持**: 可以利用用戶安裝的廣告攔截器、代理擴展等工具
5. **更自然的行爲**: 瀏覽器行爲模式更接近真實用戶

## 快速開始

### 1. 啓用CDP模式

在 `config/base_config.py` 中設置：

```python
# 啓用CDP模式
ENABLE_CDP_MODE = True

# CDP調試端口（可選，默認9222）
CDP_DEBUG_PORT = 9222

# 是否在無頭模式下運行（建議設爲False以獲得最佳反檢測效果）
CDP_HEADLESS = False

# 程序結束時是否自動關閉瀏覽器
AUTO_CLOSE_BROWSER = True
```

### 2. 運行測試

```bash
# 運行CDP功能測試
python examples/cdp_example.py

# 運行小紅書爬蟲（CDP模式）
python main.py
```

## 配置選項詳解

### 基礎配置

| 配置項 | 類型 | 默認值 | 說明 |
|--------|------|--------|------|
| `ENABLE_CDP_MODE` | bool | False | 是否啓用CDP模式 |
| `CDP_DEBUG_PORT` | int | 9222 | CDP調試端口 |
| `CDP_HEADLESS` | bool | False | CDP模式下的無頭模式 |
| `AUTO_CLOSE_BROWSER` | bool | True | 程序結束時是否關閉瀏覽器 |

### 高級配置

| 配置項 | 類型 | 默認值 | 說明 |
|--------|------|--------|------|
| `CUSTOM_BROWSER_PATH` | str | "" | 自定義瀏覽器路徑 |
| `BROWSER_LAUNCH_TIMEOUT` | int | 30 | 瀏覽器啓動超時時間（秒） |

### 自定義瀏覽器路徑

如果系統自動檢測失敗，可以手動指定瀏覽器路徑：

```python
# Windows示例
CUSTOM_BROWSER_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# macOS示例  
CUSTOM_BROWSER_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Linux示例
CUSTOM_BROWSER_PATH = "/usr/bin/google-chrome"
```

## 支持的瀏覽器

### Windows
- Google Chrome (穩定版、Beta、Dev、Canary)
- Microsoft Edge (穩定版、Beta、Dev、Canary)

### macOS
- Google Chrome (穩定版、Beta、Dev、Canary)
- Microsoft Edge (穩定版、Beta、Dev、Canary)

### Linux
- Google Chrome / Chromium
- Microsoft Edge

## 使用示例

### 基本使用

```python
import asyncio
from playwright.async_api import async_playwright
from tools.cdp_browser import CDPBrowserManager

async def main():
    cdp_manager = CDPBrowserManager()
    
    async with async_playwright() as playwright:
        # 啓動CDP瀏覽器
        browser_context = await cdp_manager.launch_and_connect(
            playwright=playwright,
            user_agent="自定義User-Agent",
            headless=False
        )
        
        # 創建頁面並訪問網站
        page = await browser_context.new_page()
        await page.goto("https://example.com")
        
        # 執行爬取操作...
        
        # 清理資源
        await cdp_manager.cleanup()

asyncio.run(main())
```

### 在爬蟲中使用

CDP模式已集成到所有平臺爬蟲中，只需啓用配置即可：

```python
# 在config/base_config.py中
ENABLE_CDP_MODE = True

# 然後正常運行爬蟲
python main.py
```

## 故障排除

### 常見問題

#### 1. 瀏覽器檢測失敗
**錯誤**: `未找到可用的瀏覽器`

**解決方案**:
- 確保已安裝Chrome或Edge瀏覽器
- 檢查瀏覽器是否在標準路徑下
- 使用`CUSTOM_BROWSER_PATH`指定瀏覽器路徑

#### 2. 端口被佔用
**錯誤**: `無法找到可用的端口`

**解決方案**:
- 關閉其他使用調試端口的程序
- 修改`CDP_DEBUG_PORT`爲其他端口
- 系統會自動嘗試下一個可用端口

#### 3. 瀏覽器啓動超時
**錯誤**: `瀏覽器在30秒內未能啓動`

**解決方案**:
- 增加`BROWSER_LAUNCH_TIMEOUT`值
- 檢查系統資源是否充足
- 嘗試關閉其他佔用資源的程序

#### 4. CDP連接失敗
**錯誤**: `CDP連接失敗`

**解決方案**:
- 檢查防火牆設置
- 確保localhost訪問正常
- 嘗試重啓瀏覽器

### 調試技巧

#### 1. 啓用詳細日誌
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 手動測試CDP連接
```bash
# 手動啓動Chrome
chrome --remote-debugging-port=9222

# 訪問調試頁面
curl http://localhost:9222/json
```

#### 3. 檢查瀏覽器進程
```bash
# Windows
tasklist | findstr chrome

# macOS/Linux  
ps aux | grep chrome
```

## 最佳實踐

### 1. 反檢測優化
- 保持`CDP_HEADLESS = False`以獲得最佳反檢測效果
- 使用真實的User-Agent字符串
- 避免過於頻繁的請求

### 2. 性能優化
- 合理設置`AUTO_CLOSE_BROWSER`
- 複用瀏覽器實例而不是頻繁重啓
- 監控內存使用情況

### 3. 安全考慮
- 不要在生產環境中保存敏感Cookie
- 定期清理瀏覽器數據
- 注意用戶隱私保護

### 4. 兼容性
- 測試不同瀏覽器版本的兼容性
- 準備回退方案（標準Playwright模式）
- 監控目標網站的反爬策略變化

## 技術原理

CDP模式的工作原理：

1. **瀏覽器檢測**: 自動掃描系統中的Chrome/Edge安裝路徑
2. **進程啓動**: 使用`--remote-debugging-port`參數啓動瀏覽器
3. **CDP連接**: 通過WebSocket連接到瀏覽器的調試接口
4. **Playwright集成**: 使用`connectOverCDP`方法接管瀏覽器控制
5. **上下文管理**: 創建或複用瀏覽器上下文進行操作

這種方式繞過了傳統WebDriver的檢測機制，提供了更加隱蔽的自動化能力。

## 更新日誌

### v1.0.0
- 初始版本發佈
- 支持Windows和macOS的Chrome/Edge檢測
- 集成到所有平臺爬蟲
- 提供完整的配置選項和錯誤處理

## 貢獻

歡迎提交Issue和Pull Request來改進CDP模式功能。

## 許可證

本功能遵循項目的整體許可證條款，僅供學習和研究使用。
