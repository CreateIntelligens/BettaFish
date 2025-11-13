"""
Query Engine 配置管理模塊

此模塊使用 pydantic-settings 管理 Query Engine 的配置，支持從環境變量和 .env 文件自動加載。
數據模型定義位置：
- 本文件 - 配置模型定義
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from loguru import logger


# 計算 .env 優先級：優先當前工作目錄，其次項目根目錄
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    """
    Query Engine 全局配置；支持 .env 和環境變量自動加載。
    變量名與原 config.py 大寫一致，便於平滑過渡。
    """
    
    # ======================= LLM 相關 =======================
    QUERY_ENGINE_API_KEY: str = Field(..., description="Query Engine LLM API密鑰，用於主LLM。您可以更改每個部分LLM使用的API，🚩只要兼容OpenAI請求格式都可以，定義好KEY、BASE_URL與MODEL_NAME即可正常使用。")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field(None, description="Query Engine LLM接口BaseUrl，可自定義廠商API")
    QUERY_ENGINE_MODEL_NAME: str = Field(..., description="Query Engine LLM模型名稱")
    QUERY_ENGINE_PROVIDER: Optional[str] = Field(None, description="Query Engine LLM提供商（兼容字段）")
    
    # ================== 網絡工具配置 ====================
    TAVILY_API_KEY: str = Field(..., description="Tavily API（申請地址：https://www.tavily.com/）API密鑰，用於Tavily網絡搜索")
    
    # ================== 搜索參數配置 ====================
    SEARCH_TIMEOUT: int = Field(240, description="搜索超時（秒）")
    SEARCH_CONTENT_MAX_LENGTH: int = Field(20000, description="用於提示的最長內容長度")
    MAX_REFLECTIONS: int = Field(2, description="最大反思輪數")
    MAX_PARAGRAPHS: int = Field(5, description="最大段落數")
    MAX_SEARCH_RESULTS: int = Field(20, description="最大搜索結果數")
    
    # ================== 輸出配置 ====================
    OUTPUT_DIR: str = Field("reports", description="輸出目錄")
    SAVE_INTERMEDIATE_STATES: bool = Field(True, description="是否保存中間狀態")
    
    class Config:
        env_file = ENV_FILE
        env_prefix = ""
        case_sensitive = False
        extra = "allow"


# 創建全局配置實例
settings = Settings()

def print_config(config: Settings):
    """
    打印配置信息
    
    Args:
        config: Settings配置對象
    """
    message = ""
    message += "=== Query Engine 配置 ===\n"
    message += f"LLM 模型: {config.QUERY_ENGINE_MODEL_NAME}\n"
    message += f"LLM Base URL: {config.QUERY_ENGINE_BASE_URL or '(默認)'}\n"
    message += f"Tavily API Key: {'已配置' if config.TAVILY_API_KEY else '未配置'}\n"
    message += f"搜索超時: {config.SEARCH_TIMEOUT} 秒\n"
    message += f"最長內容長度: {config.SEARCH_CONTENT_MAX_LENGTH}\n"
    message += f"最大反思次數: {config.MAX_REFLECTIONS}\n"
    message += f"最大段落數: {config.MAX_PARAGRAPHS}\n"
    message += f"最大搜索結果數: {config.MAX_SEARCH_RESULTS}\n"
    message += f"輸出目錄: {config.OUTPUT_DIR}\n"
    message += f"保存中間狀態: {config.SAVE_INTERMEDIATE_STATES}\n"
    message += f"LLM API Key: {'已配置' if config.QUERY_ENGINE_API_KEY else '未配置'}\n"
    message += "========================\n"
    logger.info(message)
