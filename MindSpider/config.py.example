# -*- coding: utf-8 -*-
"""
存儲數據庫連接信息和API密鑰
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field
from pathlib import Path

# 計算 .env 優先級：優先當前工作目錄，其次項目根目錄（MindSpider 的上級目錄）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))

class Settings(BaseSettings):
    """全局配置管理，優先從環境變量和.env加載。支持MySQL/PostgreSQL統一數據庫參數命名。"""
    DB_DIALECT: str = Field("mysql", description="數據庫類型，支持'mysql'或'postgresql'")
    DB_HOST: str = Field("your_host", description="數據庫主機名或IP地址")
    DB_PORT: int = Field(3306, description="數據庫端口號")
    DB_USER: str = Field("your_username", description="數據庫用戶名")
    DB_PASSWORD: str = Field("your_password", description="數據庫密碼")
    DB_NAME: str = Field("mindspider", description="數據庫名稱")
    DB_CHARSET: str = Field("utf8mb4", description="數據庫字符集")
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MINDSPIDER API密鑰")
    MINDSPIDER_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="MINDSPIDER API基礎URL，推薦deepseek-chat模型使用https://api.deepseek.com")
    MINDSPIDER_MODEL_NAME: Optional[str] = Field("deepseek-chat", description="MINDSPIDER API模型名稱, 推薦deepseek-chat")

    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"

settings = Settings()
