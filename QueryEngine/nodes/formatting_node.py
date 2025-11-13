"""
報告格式化節點
負責將最終研究結果格式化爲美觀的Markdown報告
"""

import json
from typing import List, Dict, Any

from .base_node import BaseNode
from loguru import logger
from ..prompts import SYSTEM_PROMPT_REPORT_FORMATTING
from ..utils.text_processing import (
    remove_reasoning_from_output,
    clean_markdown_tags
)


class ReportFormattingNode(BaseNode):
    """格式化最終報告的節點"""
    
    def __init__(self, llm_client):
        """
        初始化報告格式化節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "ReportFormattingNode")
    
    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入數據"""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
                return isinstance(data, list) and all(
                    isinstance(item, dict) and "title" in item and "paragraph_latest_state" in item
                    for item in data
                )
            except:
                return False
        elif isinstance(input_data, list):
            return all(
                isinstance(item, dict) and "title" in item and "paragraph_latest_state" in item
                for item in input_data
            )
        return False
    
    def run(self, input_data: Any, **kwargs) -> str:
        """
        調用LLM生成Markdown格式報告
        
        Args:
            input_data: 包含所有段落信息的列表
            **kwargs: 額外參數
            
        Returns:
            格式化的Markdown報告
        """
        try:
            if not self.validate_input(input_data):
                raise ValueError("輸入數據格式錯誤，需要包含title和paragraph_latest_state的列表")
            
            # 準備輸入數據
            if isinstance(input_data, str):
                message = input_data
            else:
                message = json.dumps(input_data, ensure_ascii=False)
            
            logger.info("正在格式化最終報告")
            
            # 調用LLM生成Markdown格式
            response = self.llm_client.invoke(
                SYSTEM_PROMPT_REPORT_FORMATTING,
                message,
            )
            
            # 處理響應
            processed_response = self.process_output(response)
            
            logger.info("成功生成格式化報告")
            return processed_response
            
        except Exception as e:
            logger.exception(f"報告格式化失敗: {str(e)}")
            raise e
    
    def process_output(self, output: str) -> str:
        """
        處理LLM輸出，清理Markdown格式
        
        Args:
            output: LLM原始輸出
            
        Returns:
            清理後的Markdown報告
        """
        try:
            # 清理響應文本
            cleaned_output = remove_reasoning_from_output(output)
            cleaned_output = clean_markdown_tags(cleaned_output)
            
            # 確保報告有基本結構
            if not cleaned_output.strip():
                return "# 報告生成失敗\n\n無法生成有效的報告內容。"
            
            # 如果沒有標題，添加一個默認標題
            if not cleaned_output.strip().startswith('#'):
                cleaned_output = "# 深度研究報告\n\n" + cleaned_output
            
            return cleaned_output.strip()
            
        except Exception as e:
            logger.exception(f"處理輸出失敗: {str(e)}")
            return "# 報告處理失敗\n\n報告格式化過程中發生錯誤。"
    
    def format_report_manually(self, paragraphs_data: List[Dict[str, str]], 
                             report_title: str = "深度研究報告") -> str:
        """
        手動格式化報告（備用方法）
        
        Args:
            paragraphs_data: 段落數據列表
            report_title: 報告標題
            
        Returns:
            格式化的Markdown報告
        """
        try:
            logger.info("使用手動格式化方法")
            
            # 構建報告
            report_lines = [
                f"# {report_title}",
                "",
                "---",
                ""
            ]
            
            # 添加各個段落
            for i, paragraph in enumerate(paragraphs_data, 1):
                title = paragraph.get("title", f"段落 {i}")
                content = paragraph.get("paragraph_latest_state", "")
                
                if content:
                    report_lines.extend([
                        f"## {title}",
                        "",
                        content,
                        "",
                        "---",
                        ""
                    ])
            
            # 添加結論
            if len(paragraphs_data) > 1:
                report_lines.extend([
                    "## 結論",
                    "",
                    "本報告通過深度搜索和研究，對相關主題進行了全面分析。"
                    "以上各個方面的內容爲理解該主題提供了重要參考。",
                    ""
                ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.exception(f"手動格式化失敗: {str(e)}")
            return "# 報告生成失敗\n\n無法完成報告格式化。"
