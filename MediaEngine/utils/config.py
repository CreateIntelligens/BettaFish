"""
Configuration management module for the Media Engine (pydantic_settings style).
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


# 計算 .env 優先級：優先當前工作目錄，其次項目根目錄
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))

class Settings(BaseSettings):
    """
    全局配置；支持 .env 和環境變量自動加載。
    變量名與原 config.py 大寫一致，便於平滑過渡。
    """
    # ====================== 數據庫配置 ======================
    DB_HOST: str = Field("your_db_host", description="數據庫主機，例如localhost 或 127.0.0.1。我們也提供雲數據庫資源便捷配置，日均10w+數據，可免費申請，聯繫我們：670939375@qq.com NOTE：爲進行數據合規性審查與服務升級，雲數據庫自2025年10月1日起暫停接收新的使用申請")
    DB_PORT: int = Field(3306, description="數據庫端口號，默認爲3306")
    DB_USER: str = Field("your_db_user", description="數據庫用戶名")
    DB_PASSWORD: str = Field("your_db_password", description="數據庫密碼")
    DB_NAME: str = Field("your_db_name", description="數據庫名稱")
    DB_CHARSET: str = Field("utf8mb4", description="數據庫字符集，推薦utf8mb4，兼容emoji")
    DB_DIALECT: str = Field("mysql", description="數據庫類型，例如 'mysql' 或 'postgresql'。用於支持多種數據庫後端（如 SQLAlchemy，請與連接信息共同配置）")

    # ======================= LLM 相關 =======================
    INSIGHT_ENGINE_API_KEY: str = Field(None, description="Insight Agent（推薦Kimi，https://platform.moonshot.cn/）API密鑰，用於主LLM。您可以更改每個部分LLM使用的API，🚩只要兼容OpenAI請求格式都可以，定義好KEY、BASE_URL與MODEL_NAME即可正常使用。重要提醒：我們強烈推薦您先使用推薦的配置申請API，先跑通再進行您的更改！")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field("https://api.moonshot.cn/v1", description="Insight Agent LLM接口BaseUrl，可自定義廠商API")
    INSIGHT_ENGINE_MODEL_NAME: str = Field("kimi-k2-0711-preview", description="Insight Agent LLM模型名稱，如kimi-k2-0711-preview")
    
    MEDIA_ENGINE_API_KEY: str = Field(None, description="Media Agent（推薦Gemini，這裏我用了一箇中轉廠商，你也可以換成你自己的，申請地址：https://www.chataiapi.com/）API密鑰")
    MEDIA_ENGINE_BASE_URL: Optional[str] = Field("https://www.chataiapi.com/v1", description="Media Agent LLM接口BaseUrl")
    MEDIA_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Media Agent LLM模型名稱，如gemini-2.5-pro")
    
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha Web Search API Key")
    BOCHA_API_KEY: Optional[str] = Field(None, description="Bocha 兼容鍵（別名）")
    
    SEARCH_TIMEOUT: int = Field(240, description="搜索超時（秒）")
    SEARCH_CONTENT_MAX_LENGTH: int = Field(20000, description="用於提示的最長內容長度")
    MAX_REFLECTIONS: int = Field(2, description="最大反思輪數")
    MAX_PARAGRAPHS: int = Field(5, description="最大段落數")
    
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MindSpider API密鑰")
    MINDSPIDER_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="MindSpider LLM接口BaseUrl")
    MINDSPIDER_MODEL_NAME: str = Field("deepseek-reasoner", description="MindSpider LLM模型名稱，如deepseek-reasoner")
    
    OUTPUT_DIR: str = Field("reports", description="輸出目錄")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="是否保存中間狀態")

    
    QUERY_ENGINE_API_KEY: str = Field(None, description="Query Agent（推薦DeepSeek，https://www.deepseek.com/）API密鑰")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="Query Agent LLM接口BaseUrl")
    QUERY_ENGINE_MODEL_NAME: str = Field("deepseek-reasoner", description="Query Agent LLM模型，如deepseek-reasoner")
    
    REPORT_ENGINE_API_KEY: str = Field(None, description="Report Agent（推薦Gemini，這裏我用了一箇中轉廠商，你也可以換成你自己的，申請地址：https://www.chataiapi.com/）API密鑰")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field("https://www.chataiapi.com/v1", description="Report Agent LLM接口BaseUrl")
    REPORT_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Report Agent LLM模型，如gemini-2.5-pro")
    
    FORUM_HOST_API_KEY: str = Field(None, description="Forum Host（Qwen3最新模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）API密鑰")
    FORUM_HOST_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Forum Host LLM BaseUrl")
    FORUM_HOST_MODEL_NAME: str = Field("Qwen/Qwen3-235B-A22B-Instruct-2507", description="Forum Host LLM模型名，如Qwen/Qwen3-235B-A22B-Instruct-2507")
    
    KEYWORD_OPTIMIZER_API_KEY: str = Field(None, description="SQL keyword Optimizer（小參數Qwen3模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）API密鑰")
    KEYWORD_OPTIMIZER_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Keyword Optimizer BaseUrl")
    KEYWORD_OPTIMIZER_MODEL_NAME: str = Field("Qwen/Qwen3-30B-A3B-Instruct-2507", description="Keyword Optimizer LLM模型名稱，如Qwen/Qwen3-30B-A3B-Instruct-2507")

    # ================== 網絡工具配置 ====================
    TAVILY_API_KEY: str = Field(None, description="Tavily API（申請地址：https://www.tavily.com/）API密鑰，用於Tavily網絡搜索")
    BOCHA_BASE_URL: Optional[str] = Field("https://api.bochaai.com/v1/ai-search", description="Bocha AI 搜索BaseUrl或博查網頁搜索BaseUrl")
    BOCHA_WEB_SEARCH_API_KEY: str = Field(None, description="Bocha API（申請地址：https://open.bochaai.com/）API密鑰，用於Bocha搜索")

    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"


settings = Settings()
