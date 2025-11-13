"""
節點基類
定義所有處理節點的基礎接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger
from ..llms.base import LLMClient
from ..state.state import State


class BaseNode(ABC):
    """節點基類"""

    def __init__(self, llm_client: LLMClient, node_name: str = ""):
        """
        初始化節點

        Args:
            llm_client: LLM客戶端
            node_name: 節點名稱
        """
        self.llm_client = llm_client
        self.node_name = node_name or self.__class__.__name__

    @abstractmethod
    def run(self, input_data: Any, **kwargs) -> Any:
        """
        執行節點處理邏輯

        Args:
            input_data: 輸入數據
            **kwargs: 額外參數

        Returns:
            處理結果
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        驗證輸入數據

        Args:
            input_data: 輸入數據

        Returns:
            驗證是否通過
        """
        return True

    def process_output(self, output: Any) -> Any:
        """
        處理輸出數據

        Args:
            output: 原始輸出

        Returns:
            處理後的輸出
        """
        return output

    def log_info(self, message: str):
        """記錄信息日誌"""
        logger.info(f"[{self.node_name}] {message}")
    
    def log_warning(self, message: str):
        """記錄警告日誌"""
        logger.warning(f"[{self.node_name}] 警告: {message}")

    def log_error(self, message: str):
        """記錄錯誤日誌"""
        logger.error(f"[{self.node_name}] 錯誤: {message}")


class StateMutationNode(BaseNode):
    """帶狀態修改功能的節點基類"""
    
    @abstractmethod
    def mutate_state(self, input_data: Any, state: State, **kwargs) -> State:
        """
        修改狀態
        
        Args:
            input_data: 輸入數據
            state: 當前狀態
            **kwargs: 額外參數
            
        Returns:
            修改後的狀態
        """
        pass
