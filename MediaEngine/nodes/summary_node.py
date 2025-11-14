"""
總結節點實現
負責根據搜索結果生成和更新段落內容
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_FIRST_SUMMARY, SYSTEM_PROMPT_REFLECTION_SUMMARY
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json,
    format_search_results_for_prompt
)

# 導入論壇讀取工具
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.forum_reader import get_latest_host_speech, format_host_speech_for_prompt
    FORUM_READER_AVAILABLE = True
except ImportError:
    FORUM_READER_AVAILABLE = False
    logger.warning("無法導入forum_reader模塊，將跳過HOST發言讀取功能")


class FirstSummaryNode(StateMutationNode):
    """根據搜索結果生成段落首次總結的節點"""
    
    def __init__(self, llm_client):
        """
        初始化首次總結節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "FirstSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        調用LLM生成段落總結
        
        Args:
            input_data: 包含title、content、search_query和search_results的數據
            **kwargs: 額外參數
            
        Returns:
            段落總結內容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("輸入數據格式錯誤")
            
            # 準備輸入數據
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data
            
            # 讀取最新的HOST發言（如果可用）
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # 將HOST發言添加到輸入數據中
                        data['host_speech'] = host_speech
                        logger.info(f"已讀取HOST發言，長度: {len(host_speech)}字符")
                except Exception as e:
                    logger.exception(f"讀取HOST發言失敗: {str(e)}")
            
            # 轉換爲JSON字符串
            message = json.dumps(data, ensure_ascii=False)
            
            # 如果有HOST發言，添加到消息前面作爲參考
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message
            
            logger.info("正在生成首次段落總結")

            # 調用LLM生成總結（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(
                SYSTEM_PROMPT_FIRST_SUMMARY,
                message,
            )
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info("成功生成首次段落總結")
            return processed_response
            
        except Exception as e:
            logger.exception(f"生成首次總結失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        處理LLM輸出，提取段落內容
        
        Args:
            output: LLM原始輸出
            
        Returns:
            段落內容
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
                logger.error(f"JSON解析失敗: {str(e)}")
                # 嘗試修復JSON
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON修復成功")
                    except JSONDecodeError:
                        logger.exception("JSON修復失敗，直接使用清理後的文本")
                        # 如果不是JSON格式，直接返回清理後的文本
                        return cleaned_output
                else:
                    logger.exception("無法修復JSON，直接使用清理後的文本")
                    # 如果不是JSON格式，直接返回清理後的文本
                    return cleaned_output
            
            # 提取段落內容
            if isinstance(result, dict):
                paragraph_content = result.get("paragraph_latest_state", "")
                if paragraph_content:
                    return paragraph_content
            
            # 如果提取失敗，返回原始清理後的文本
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"處理輸出失敗: {str(e)}")
            return "段落總結生成失敗"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        更新段落的最新總結到狀態
        
        Args:
            input_data: 輸入數據
            state: 當前狀態
            paragraph_index: 段落索引
            **kwargs: 額外參數
            
        Returns:
            更新後的狀態
        """
        try:
            # 生成總結
            summary = self.run(input_data, **kwargs)
            
            # 更新狀態
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = summary
                logger.info(f"已更新段落 {paragraph_index} 的首次總結")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出範圍")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"狀態更新失敗: {str(e)}")
            raise e


class ReflectionSummaryNode(StateMutationNode):
    """根據反思搜索結果更新段落總結的節點"""
    
    def __init__(self, llm_client):
        """
        初始化反思總結節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "ReflectionSummaryNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
                return all(field in data for field in required_fields)
            except JSONDecodeError:
                return False
        elif isinstance(input_data, dict):
            required_fields = ["title", "content", "search_query", "search_results", "paragraph_latest_state"]
            return all(field in input_data for field in required_fields)
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        調用LLM更新段落內容
        
        Args:
            input_data: 包含完整反思信息的數據
            **kwargs: 額外參數
            
        Returns:
            更新後的段落內容
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("輸入數據格式錯誤")
            
            # 準備輸入數據
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data.copy() if isinstance(input_data, dict) else input_data
            
            # 讀取最新的HOST發言（如果可用）
            if FORUM_READER_AVAILABLE:
                try:
                    host_speech = get_latest_host_speech()
                    if host_speech:
                        # 將HOST發言添加到輸入數據中
                        data['host_speech'] = host_speech
                        logger.info(f"已讀取HOST發言，長度: {len(host_speech)}字符")
                except Exception as e:
                    logger.exception(f"讀取HOST發言失敗: {str(e)}")
            
            # 轉換爲JSON字符串
            message = json.dumps(data, ensure_ascii=False)
            
            # 如果有HOST發言，添加到消息前面作爲參考
            if FORUM_READER_AVAILABLE and 'host_speech' in data and data['host_speech']:
                formatted_host = format_host_speech_for_prompt(data['host_speech'])
                message = formatted_host + "\n" + message
            
            logger.info("正在生成反思總結")

            # 調用LLM生成總結（流式，安全拼接UTF-8）
            response = self.llm_client.stream_invoke_to_string(
                SYSTEM_PROMPT_REFLECTION_SUMMARY,
                message,
            )
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info("成功生成反思總結")
            return processed_response
            
        except Exception as e:
            logger.exception(f"生成反思總結失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        處理LLM輸出，提取更新後的段落內容
        
        Args:
            output: LLM原始輸出
            
        Returns:
            更新後的段落內容
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
                logger.error(f"JSON解析失敗: {str(e)}")
                # 嘗試修復JSON
                fixed_json = fix_incomplete_json(cleaned_output)
                if fixed_json:
                    try:
                        result = json.loads(fixed_json)
                        logger.info("JSON修復成功")
                    except JSONDecodeError:
                        logger.error("JSON修復失敗，直接使用清理後的文本")
                        # 如果不是JSON格式，直接返回清理後的文本
                        return cleaned_output
                else:
                    logger.error("無法修復JSON，直接使用清理後的文本")
                    # 如果不是JSON格式，直接返回清理後的文本
                    return cleaned_output
            
            # 提取更新後的段落內容
            if isinstance(result, dict):
                updated_content = result.get("updated_paragraph_latest_state", "")
                if updated_content:
                    return updated_content
            
            # 如果提取失敗，返回原始清理後的文本
            return cleaned_output
            
        except Exception as e:
            logger.exception(f"處理輸出失敗: {str(e)}")
            return "反思總結生成失敗"
    
    def mutate_state(self, input_data: Any, state: State, paragraph_index: int, **kwargs) -> State:
        """
        將更新後的總結寫入狀態
        
        Args:
            input_data: 輸入數據
            state: 當前狀態
            paragraph_index: 段落索引
            **kwargs: 額外參數
            
        Returns:
            更新後的狀態
        """
        try:
            # 生成更新後的總結
            updated_summary = self.run(input_data, **kwargs)
            
            # 更新狀態
            if 0 <= paragraph_index < len(state.paragraphs):
                state.paragraphs[paragraph_index].research.latest_summary = updated_summary
                state.paragraphs[paragraph_index].research.increment_reflection()
                logger.info(f"已更新段落 {paragraph_index} 的反思總結")
            else:
                raise ValueError(f"段落索引 {paragraph_index} 超出範圍")
            
            state.update_timestamp()
            return state
            
        except Exception as e:
            logger.exception(f"狀態更新失敗: {str(e)}")
            raise e
