"""
報告結構生成節點
負責根據查詢生成報告的整體結構
"""

import json
from typing import Dict, Any, List
from json.decoder import JSONDecodeError
from loguru import logger

from .base_node import StateMutationNode
from ..state.state import State
from ..prompts import SYSTEM_PROMPT_REPORT_STRUCTURE
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_json_tags,
    extract_clean_response,
    fix_incomplete_json
)


class ReportStructureNode(StateMutationNode):
    """生成報告結構的節點"""
    
    def __init__(self, llm_client, query: str):
        """
        初始化報告結構節點
        
        Args:
            llm_client: LLM客戶端
            query: 用戶查詢
        """
        super().__init__(llm_client, "ReportStructureNode")
        self.query = query
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        return isinstance(self.query, str) and len(self.query.strip()) > 0
    
    def run(self, input_data: Any = None, **kwargs) -> List[Dict[str, str]]:
        """
        調用LLM生成報告結構
        
        Args:
            input_data: 輸入數據（這裏不使用，使用初始化時的query）
            **kwargs: 額外參數
            
        Returns:
            報告結構列表
        """
        try:
            logger.info(f"正在爲查詢生成報告結構: {self.query}")
            
            # 調用LLM
            response = self.llm_client.invoke(SYSTEM_PROMPT_REPORT_STRUCTURE, self.query)
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info(f"成功生成 {len(processed_response)} 個段落結構")
            return processed_response
            
        except Exception as e:
            logger.exception(f"生成報告結構失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> List[Dict[str, str]]:
        """
        處理LLM輸出，提取報告結構
        
        Args:
            output: LLM原始輸出
            
        Returns:
            處理後的報告結構列表
        """
        try:
            # 清理響應文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_json_tags(cleaned_output)
            
            # 記錄清理後的輸出用於調試
            logger.info(f"清理後的輸出: {cleaned_output}")
            
            # 解析JSON
            try:
                report_structure = json.loads(cleaned_output)
                logger.info("JSON解析成功")
            except JSONDecodeError as e:
                logger.exception(f"JSON解析失敗: {str(e)}")
                # 使用更強大的提取方法
                report_structure = extract_clean_response(cleaned_output)
                if "error" in report_structure:
                    logger.exception("JSON解析失敗，嘗試修復...")
                    # 嘗試修復JSON
                    fixed_json = fix_incomplete_json(cleaned_output)
                    if fixed_json:
                        try:
                            report_structure = json.loads(fixed_json)
                            logger.info("JSON修復成功")
                        except JSONDecodeError:
                            logger.exception("JSON修復失敗")
                            # 返回默認結構
                            return self._generate_default_structure()
                    else:
                        logger.exception("無法修復JSON，使用默認結構")
                        return self._generate_default_structure()
            
            # 驗證結構
            if not isinstance(report_structure, list):
                logger.info("報告結構不是列表，嘗試轉換...")
                if isinstance(report_structure, dict):
                    # 如果是單個對象，包裝成列表
                    report_structure = [report_structure]
                else:
                    logger.exception("報告結構格式無效，使用默認結構")
                    return self._generate_default_structure()
            
            # 驗證每個段落
            validated_structure = []
            for i, paragraph in enumerate(report_structure):
                if not isinstance(paragraph, dict):
                    logger.warning(f"段落 {i+1} 不是字典格式，跳過")
                    continue
                
                title = paragraph.get("title", f"段落 {i+1}")
                content = paragraph.get("content", "")
                
                if not title or not content:
                    logger.warning(f"段落 {i+1} 缺少標題或內容，跳過")
                    continue
                
                validated_structure.append({
                    "title": title,
                    "content": content
                })
            
            if not validated_structure:
                logger.warning("沒有有效的段落結構，使用默認結構")
                return self._generate_default_structure()
            
            logger.info(f"成功驗證 {len(validated_structure)} 個段落結構")
            return validated_structure
            
        except Exception as e:
            logger.exception(f"處理輸出失敗: {str(e)}")
            return self._generate_default_structure()
    
    def _generate_default_structure(self) -> List[Dict[str, str]]:
        """
        生成默認的報告結構
        
        Returns:
            默認的報告結構列表
        """
        logger.info("生成默認報告結構")
        return [
            {
                "title": "研究概述",
                "content": "對查詢主題進行總體概述和分析"
            },
            {
                "title": "深度分析",
                "content": "深入分析查詢主題的各個方面"
            }
        ]
    
    def mutate_state(self, input_data: Any = None, state: State = None, **kwargs) -> State:
        """
        將報告結構寫入狀態
        
        Args:
            input_data: 輸入數據
            state: 當前狀態，如果爲None則創建新狀態
            **kwargs: 額外參數
            
        Returns:
            更新後的狀態
        """
        if state is None:
            state = State()
        
        try:
            # 生成報告結構
            report_structure = self.run(input_data, **kwargs)
            
            # 設置查詢和報告標題
            state.query = self.query
            if not state.report_title:
                state.report_title = f"關於'{self.query}'的深度研究報告"
            
            # 添加段落到狀態
            for paragraph_data in report_structure:
                state.add_paragraph(
                    title=paragraph_data["title"],
                    content=paragraph_data["content"]
                )
            
            logger.info(f"已將 {len(report_structure)} 個段落添加到狀態中")
            return state
            
        except Exception as e:
            logger.exception(f"狀態更新失敗: {str(e)}")
            raise e
