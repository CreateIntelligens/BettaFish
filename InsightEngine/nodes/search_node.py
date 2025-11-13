"""
搜索節點實現
負責生成搜索查詢和反思查詢
"""

import json
from typing import Dict, Any
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_FIRST_SEARCH, SYSTEM_PROMPT_REFLECTION
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json
)


class FirstSearchNode(BaseNode):
    """爲段落生成首次搜索查詢的節點"""
    
    def __init__(self, llm_client):
        """
        初始化首次搜索節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "FirstSearchNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                return "title" in data and "content" in data
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            return "title" in input_data and "content" in input_data
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        調用LLM生成搜索查詢和理由
        
        Args:
            input_data: 包含title和content的字符串或字典
            **kwargs: 額外參數
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("輸入數據格式錯誤，需要包含title和content字段")
            
            # 準備輸入數據
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("正在生成首次搜索查詢")
            
            # 調用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_FIRST_SEARCH, message)
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info(f"生成搜索查詢: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"生成首次搜索查詢失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        處理LLM輸出，提取搜索查詢和推理
        
        Args:
            output: LLM原始輸出
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            # 清理響應文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 記錄清理後的輸出用於調試
            logger.info(f"清理後的輸出: {cleaned_output}")
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON解析成功")
            except JSONDecodeError as e:
                logger.exception(f"JSON解析失敗: {str(e)}")
                # 使用更強大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON解析失敗，嘗試修復...")
                    # 嘗試修復JSON
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON修復成功")
                        except JSONDecodeError:
                            logger.error("JSON修復失敗")
                            # 返回默認查詢
                            return self._get_default_search_query()
                    else:
                        logger.error("無法修復JSON，使用默認查詢")
                        return self._get_default_search_query()
            
            # 驗證和清理結果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("未找到搜索查詢，使用默認查詢")
                return self._get_default_search_query()
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.log_error(f"處理輸出失敗: {str(e)}")
            # 返回默認查詢
            return self._get_default_search_query()
    
    def _get_default_search_query(self) -> Dict[str, str]:
        """
        獲取默認搜索查詢
        
        Returns:
            默認的搜索查詢字典
        """
        return {
            "search_query": "相關主題研究",
            "reasoning": "由於解析失敗，使用默認搜索查詢"
        }


class ReflectionNode(BaseNode):
    """反思段落並生成新搜索查詢的節點"""
    
    def __init__(self, llm_client):
        """
        初始化反思節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "ReflectionNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, str]:
        """
        調用LLM反思並生成搜索查詢
        
        Args:
            input_data: 包含title、content和paragraph_latest_state的字符串或字典
            **kwargs: 額外參數
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("輸入數據格式錯誤，需要包含title、content和paragraph_latest_state字段")
            
            # 準備輸入數據
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("正在進行反思並生成新搜索查詢")
            
            # 調用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_REFLECTION, message)
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info(f"反思生成搜索查詢: {processed_response.get('search_query', 'N/A')}")
            return processed_response
            
        except Exception as e:
            logger.exception(f"反思生成搜索查詢失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> Dict[str, str]:
        """
        處理LLM輸出，提取搜索查詢和推理
        
        Args:
            output: LLM原始輸出
            
        Returns:
            包含search_query和reasoning的字典
        """
        try:
            # 清理響應文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 記錄清理後的輸出用於調試
            logger.info(f"清理後的輸出: {cleaned_output}")
            
            # 解析JSON
            try:
                result = json.loads(cleaned_output)
                logger.info("JSON解析成功")
            except JSONDecodeError as e:
                logger.exception(f"JSON解析失敗: {str(e)}")
                # 使用更強大的提取方法
                result = extract_clean_response(cleaned_output)
                if "error" in result:
                    logger.error("JSON解析失敗，嘗試修復...")
                    # 嘗試修復JSON
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            result = json.loads(fixed_json)
                            logger.info("JSON修復成功")
                        except JSONDecodeError:
                            logger.error("JSON修復失敗")
                            # 返回默認查詢
                            return self._get_default_reflection_query()
                    else:
                        logger.error("無法修復JSON，使用默認查詢")
                        return self._get_default_reflection_query()
            
            # 驗證和清理結果
            search_query = result.get("search_query", "")
            reasoning = result.get("reasoning", "")
            
            if not search_query:
                logger.warning("未找到搜索查詢，使用默認查詢")
                return self._get_default_reflection_query()
            
            return {
                "search_query": search_query,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.exception(f"處理輸出失敗: {str(e)}")
            # 返回默認查詢
            return self._get_default_reflection_query()
    
    def _get_default_reflection_query(self) -> Dict[str, str]:
        """
        獲取默認反思搜索查詢
        
        Returns:
            默認的反思搜索查詢字典
        """
        return {
            "search_query": "深度研究補充信息",
            "reasoning": "由於解析失敗，使用默認反思搜索查詢"
        }
