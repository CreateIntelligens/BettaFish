"""
Configuration management module for the Insight Engine.
Handles environment variables and config file parameters.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from loguru import logger

class Settings(BaseSettings):
    INSIGHT_ENGINE_API_KEY: Optional[str] = Field(None, description="Insight Engine LLM API密鑰")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field(None, description="Insight Engine LLM base url，可選")
    INSIGHT_ENGINE_MODEL_NAME: Optional[str] = Field(None, description="Insight Engine LLM模型名稱")
    INSIGHT_ENGINE_PROVIDER: Optional[str] = Field(None, description="Insight Engine模型提供者，不再建議使用")
    DB_HOST: Optional[str] = Field(None, description="數據庫主機")
    DB_USER: Optional[str] = Field(None, description="數據庫用戶名")
    DB_PASSWORD: Optional[str] = Field(None, description="數據庫密碼")
    DB_NAME: Optional[str] = Field(None, description="數據庫名稱")
    DB_PORT: int = Field(3306, description="數據庫端口")
    DB_CHARSET: str = Field("utf8mb4", description="數據庫字符集")
    DB_DIALECT: Optional[str] = Field("mysql", description="數據庫方言，如mysql、postgresql等，SQLAlchemy後端選擇")
    MAX_REFLECTIONS: int = Field(3, description="最大反思次數")
    MAX_PARAGRAPHS: int = Field(6, description="最大段落數")
    SEARCH_TIMEOUT: int = Field(240, description="單次搜索請求超時")
    MAX_CONTENT_LENGTH: int = Field(500000, description="搜索最大內容長度")
    DEFAULT_SEARCH_HOT_CONTENT_LIMIT: int = Field(100, description="熱榜內容默認最大數")
    DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE: int = Field(50, description="按表全局話題最大數")
    DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE: int = Field(100, description="按日期話題最大數")
    DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT: int = Field(500, description="單話題評論最大數")
    DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT: int = Field(200, description="平臺搜索話題最大數")
    MAX_SEARCH_RESULTS_FOR_LLM: int = Field(0, description="供LLM用搜索結果最大數")
    MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS: int = Field(0, description="高置信度情感分析最大數")
    OUTPUT_DIR: str = Field("reports", description="輸出路徑")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="是否保存中間狀態")

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False
        extra = "allow"

settings = Settings()