"""
Configuration management module for the Report Engine.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

from loguru import logger

class Settings(BaseSettings):
    """Report Engine 配置，環境變量與字段均爲REPORT_ENGINE_前綴一致大寫。"""
    REPORT_ENGINE_API_KEY: Optional[str] = Field(None, description="Report Engine LLM API密鑰")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field(None, description="Report Engine LLM基礎URL")
    REPORT_ENGINE_MODEL_NAME: Optional[str] = Field(None, description="Report Engine LLM模型名稱")
    REPORT_ENGINE_PROVIDER: Optional[str] = Field(None, description="模型服務商，僅兼容保留")
    MAX_CONTENT_LENGTH: int = Field(200000, description="最大內容長度")
    OUTPUT_DIR: str = Field("final_reports", description="主輸出目錄")
    TEMPLATE_DIR: str = Field("ReportEngine/report_template", description="多模板目錄")
    API_TIMEOUT: float = Field(900.0, description="單API超時時間（秒）")
    MAX_RETRY_DELAY: float = Field(180.0, description="最大重試間隔（秒）")
    MAX_RETRIES: int = Field(8, description="最大重試次數")
    LOG_FILE: str = Field("logs/report.log", description="日誌輸出文件")
    ENABLE_PDF_EXPORT: bool = Field(True, description="是否允許導出PDF")
    CHART_STYLE: str = Field("modern", description="圖表樣式：modern/classic/")

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False
        extra = "allow"

settings = Settings()


def print_config(config: Settings):
    message = ""
    message += "\n=== Report Engine 配置 ===\n"
    message += f"LLM 模型: {config.REPORT_ENGINE_MODEL_NAME}\n"
    message += f"LLM Base URL: {config.REPORT_ENGINE_BASE_URL or '(默認)'}\n"
    message += f"最大內容長度: {config.MAX_CONTENT_LENGTH}\n"
    message += f"輸出目錄: {config.OUTPUT_DIR}\n"
    message += f"模板目錄: {config.TEMPLATE_DIR}\n"
    message += f"API 超時時間: {config.API_TIMEOUT} 秒\n"
    message += f"最大重試間隔: {config.MAX_RETRY_DELAY} 秒\n"
    message += f"最大重試次數: {config.MAX_RETRIES}\n"
    message += f"日誌文件: {config.LOG_FILE}\n"
    message += f"PDF 導出: {config.ENABLE_PDF_EXPORT}\n"
    message += f"圖表樣式: {config.CHART_STYLE}\n"
    message += f"LLM API Key: {'已配置' if config.REPORT_ENGINE_API_KEY else '未配置'}\n"
    message += "=========================\n"
    logger.info(message)
