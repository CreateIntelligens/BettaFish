# MediaCrawler使用方法

## 創建並激活 python 虛擬環境
> 如果是爬取抖音和知乎，需要提前安裝nodejs環境，版本大於等於：`16`即可 <br>
   ```shell   
   # 進入項目根目錄
   cd MediaCrawler
   
   # 創建虛擬環境
   # 我的python版本是：3.9.6，requirements.txt中的庫是基於這個版本的，如果是其他python版本，可能requirements.txt中的庫不兼容，自行解決一下。
   python -m venv venv
   
   # macos & linux 激活虛擬環境
   source venv/bin/activate

   # windows 激活虛擬環境
   venv\Scripts\activate

   ```

## 安裝依賴庫

   ```shell
   pip install -r requirements.txt
   ```

## 安裝 playwright瀏覽器驅動

   ```shell
   playwright install
   ```

## 運行爬蟲程序

   ```shell
   ### 項目默認是沒有開啓評論爬取模式，如需評論請在config/base_config.py中的 ENABLE_GET_COMMENTS 變量修改
   ### 一些其他支持項，也可以在config/base_config.py查看功能，寫的有中文註釋
   
   # 從配置文件中讀取關鍵詞搜索相關的帖子並爬取帖子信息與評論
   python main.py --platform xhs --lt qrcode --type search
   
   # 從配置文件中讀取指定的帖子ID列表獲取指定帖子的信息與評論信息
   python main.py --platform xhs --lt qrcode --type detail
   
   # 使用SQLite數據庫存儲數據（推薦個人用戶使用）
   python main.py --platform xhs --lt qrcode --type search --save_data_option sqlite
   
   # 使用MySQL數據庫存儲數據
   python main.py --platform xhs --lt qrcode --type search --save_data_option db
  
   # 打開對應APP掃二維碼登錄
     
   # 其他平臺爬蟲使用示例，執行下面的命令查看
   python main.py --help    
   ```

## 💾 數據存儲

支持多種數據存儲方式：
- **CSV 文件**: 支持保存至 CSV (位於 `data/` 目錄下)
- **JSON 文件**: 支持保存至 JSON (位於 `data/` 目錄下)
- **數據庫存儲**
  - 使用 `--init_db` 參數進行數據庫初始化 (使用 `--init_db` 時，無需其他可選參數)
  - **SQLite 數據庫**: 輕量級數據庫，無需服務器，適合個人使用 (推薦)
    1. 初始化: `--init_db sqlite`
    2. 數據存儲: `--save_data_option sqlite`
  - **MySQL 數據庫**: 支持保存至關係型數據庫 MySQL (需提前創建數據庫)
    1. 初始化: `--init_db mysql`
    2. 數據存儲: `--save_data_option db` (db 參數爲兼容歷史更新保留)

## 免責聲明
> **免責聲明：**
> 
> 大家請以學習爲目的使用本倉庫，爬蟲違法違規的案件：https://github.com/HiddenStrawberry/Crawler_Illegal_Cases_In_China  <br>
>
>本項目的所有內容僅供學習和參考之用，禁止用於商業用途。任何人或組織不得將本倉庫的內容用於非法用途或侵犯他人合法權益。本倉庫所涉及的爬蟲技術僅用於學習和研究，不得用於對其他平臺進行大規模爬蟲或其他非法行爲。對於因使用本倉庫內容而引起的任何法律責任，本倉庫不承擔任何責任。使用本倉庫的內容即表示您同意本免責聲明的所有條款和條件。

