# -*- coding: utf-8 -*-
"""
微輿配置文件

此模塊使用 pydantic-settings 管理全局配置，支持從環境變量和 .env 文件自動加載。
數據模型定義位置：
- 本文件 - 配置模型定義
"""

import os
import time
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional
from loguru import logger


# 計算 .env 優先級：優先當前工作目錄，其次項目根目錄
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    """
    全局配置；支持 .env 和環境變量自動加載。
    變量名與原 config.py 大寫一致，便於平滑過渡。
    """
    # ================== Flask 服務器配置 ====================
    HOST: str = Field("0.0.0.0", description="Flask服務器主機地址，默認0.0.0.0（允許外部訪問）")
    PORT: int = Field(5000, description="Flask服務器端口號，默認5000")

    # ====================== 數據庫配置 ======================
    DB_DIALECT: str = Field("mysql", description="數據庫類型，例如 'mysql' 或 'postgresql'。用於支持多種數據庫後端（如 SQLAlchemy，請與連接信息共同配置）")
    DB_HOST: str = Field("your_db_host", description="數據庫主機，例如localhost 或 127.0.0.1。我們也提供雲數據庫資源便捷配置，日均10w+數據，可免費申請，聯繫我們：670939375@qq.com NOTE：為進行數據合規性審查與服務升級，雲數據庫自2025年10月1日起暫停接收新的使用申請")
    DB_PORT: int = Field(3306, description="數據庫端口號，默認為3306")
    DB_USER: str = Field("your_db_user", description="數據庫用戶名")
    DB_PASSWORD: str = Field("your_db_password", description="數據庫密碼")
    DB_NAME: str = Field("your_db_name", description="數據庫名稱")
    DB_CHARSET: str = Field("utf8mb4", description="數據庫字符集，推薦utf8mb4，兼容emoji")

    # ======================= LLM 相關 =======================
    # 我們的LLM模型API贊助商有：https://share.302.ai/P66Qe3、https://aihubmix.com/?aff=8Ds9，提供了非常全面的模型api

    # Insight Agent（推薦Kimi，申請地址：https://platform.moonshot.cn/）
    INSIGHT_ENGINE_API_KEY: Optional[str] = Field(None, description="Insight Agent（推薦Kimi，https://platform.moonshot.cn/）API密鑰，用於主LLM。您可以更改每個部分LLM使用的API，🚩只要兼容OpenAI請求格式都可以，定義好KEY、BASE_URL與MODEL_NAME即可正常使用。重要提醒：我們強烈推薦您先使用推薦的配置申請API，先跑通再進行您的更改！")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field("https://api.moonshot.cn/v1", description="Insight Agent LLM接口BaseUrl，可自定義廠商API")
    INSIGHT_ENGINE_MODEL_NAME: str = Field("kimi-k2-0711-preview", description="Insight Agent LLM模型名稱，如kimi-k2-0711-preview")
    
    # Media Agent（推薦Gemini，推薦中轉廠商：https://aihubmix.com/?aff=8Ds9）
    MEDIA_ENGINE_API_KEY: Optional[str] = Field(None, description="Media Agent（推薦Gemini，推薦中轉api廠商：https://aihubmix.com/?aff=8Ds9")
    MEDIA_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Media Agent LLM接口BaseUrl")
    MEDIA_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Media Agent LLM模型名稱，如gemini-2.5-pro")
    
    # Query Agent（推薦DeepSeek，申請地址：https://www.deepseek.com/）
    QUERY_ENGINE_API_KEY: Optional[str] = Field(None, description="Query Agent（推薦DeepSeek，https://www.deepseek.com/）API密鑰")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="Query Agent LLM接口BaseUrl")
    QUERY_ENGINE_MODEL_NAME: str = Field("deepseek-reasoner", description="Query Agent LLM模型，如deepseek-reasoner")
    
    # Report Agent（推薦Gemini，推薦中轉廠商：https://aihubmix.com/?aff=8Ds9）
    REPORT_ENGINE_API_KEY: Optional[str] = Field(None, description="Report Agent（推薦Gemini，推薦中轉api廠商：https://aihubmix.com/?aff=8Ds9")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Report Agent LLM接口BaseUrl")
    REPORT_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Report Agent LLM模型，如gemini-2.5-pro")
    
    # Forum Host（Qwen3最新模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）
    FORUM_HOST_API_KEY: Optional[str] = Field(None, description="Forum Host（Qwen3最新模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）API密鑰")
    FORUM_HOST_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Forum Host LLM BaseUrl")
    FORUM_HOST_MODEL_NAME: str = Field("Qwen/Qwen3-235B-A22B-Instruct-2507", description="Forum Host LLM模型名，如Qwen/Qwen3-235B-A22B-Instruct-2507")
    
    # SQL keyword Optimizer（小參數Qwen3模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）
    KEYWORD_OPTIMIZER_API_KEY: Optional[str] = Field(None, description="SQL keyword Optimizer（小參數Qwen3模型，這裏我使用了硅基流動這個平臺，申請地址：https://cloud.siliconflow.cn/）API密鑰")
    KEYWORD_OPTIMIZER_BASE_URL: Optional[str] = Field("https://api.siliconflow.cn/v1", description="Keyword Optimizer BaseUrl")
    KEYWORD_OPTIMIZER_MODEL_NAME: str = Field("Qwen/Qwen3-30B-A3B-Instruct-2507", description="Keyword Optimizer LLM模型名稱，如Qwen/Qwen3-30B-A3B-Instruct-2507")
    
    # ================== 網絡工具配置 ====================
    # Tavily API（申請地址：https://www.tavily.com/）
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily API（申請地址：https://www.tavily.com/）API密鑰，用於Tavily網絡搜索")
    
    BOCHA_BASE_URL: Optional[str] = Field("https://api.bochaai.com/v1/ai-search", description="Bocha AI 搜索BaseUrl或博查網頁搜索BaseUrl")
    # Bocha API（申請地址：https://open.bochaai.com/）
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha API（申請地址：https://open.bochaai.com/）API密鑰，用於Bocha搜索")
    
    # ================== Insight Engine 搜索配置 ====================
    DEFAULT_SEARCH_HOT_CONTENT_LIMIT: int = Field(100, description="熱榜內容默認最大數")
    DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE: int = Field(50, description="按表全局話題最大數")
    DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE: int = Field(100, description="按日期話題最大數")
    DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT: int = Field(500, description="單話題評論最大數")
    DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT: int = Field(200, description="平臺搜索話題最大數")
    MAX_SEARCH_RESULTS_FOR_LLM: int = Field(0, description="供LLM用搜索結果最大數")
    MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS: int = Field(0, description="高置信度情感分析最大數")
    MAX_REFLECTIONS: int = Field(3, description="最大反思次數")
    MAX_PARAGRAPHS: int = Field(6, description="最大段落數")
    SEARCH_TIMEOUT: int = Field(240, description="單次搜索請求超時")
    MAX_CONTENT_LENGTH: int = Field(500000, description="搜索最大內容長度")
    
    # ================== 系統環境 ====================
    TIMEZONE: str = Field("Etc/GMT-8", description="系統時區，默認為UTC+8（Etc/GMT-8）")
    
    model_config = ConfigDict(
        env_file=ENV_FILE,
        env_prefix="",
        case_sensitive=False,
        extra="allow"
    )


# 創建全局配置實例
settings = Settings()


def _normalize_timezone(value: str) -> str:
    """解析常見時區別名如 UTC+8 到 tz 資料庫名稱。"""
    if not value:
        return ""
    cleaned = value.strip()
    alias = {
        "UTC+8": "Etc/GMT-8",
        "UTC-8": "Etc/GMT+8",
        "UTC+0": "Etc/UTC",
        "UTC": "Etc/UTC",
    }.get(cleaned.upper())
    return alias or cleaned


def _apply_timezone():
    tz_value = _normalize_timezone(settings.TIMEZONE)
    if not tz_value:
        return
    os.environ["TZ"] = tz_value
    if hasattr(time, "tzset"):
        time.tzset()


def reload_settings() -> Settings:
    """
    重新加載配置

    從 .env 文件和環境變量重新加載配置，更新全局 settings 實例。
    用於在運行時動態更新配置。

    Returns:
        Settings: 新創建的配置實例
    """

    global settings
    settings = Settings()
    _apply_timezone()  # 重新應用時區
    return settings


_apply_timezone()
