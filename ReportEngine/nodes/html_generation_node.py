"""
HTML生成節點
將整合後的內容轉換爲美觀的HTML報告
"""

import json
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from .base_node import StateMutationNode
from ..llms.base import LLMClient
from ..state.state import ReportState
from ..prompts import SYSTEM_PROMPT_HTML_GENERATION
# 不再需要text_processing依賴


class HTMLGenerationNode(StateMutationNode):
    """HTML生成處理節點"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化HTML生成節點
        
        Args:
            llm_client: LLM客戶端
        """
        super().__init__(llm_client, "HTMLGenerationNode")
    
    def run(self, input_data: Dict[str, Any], **kwargs) -> str:
        """
        執行HTML生成
        
        Args:
            input_data: 包含報告數據的字典
                - query: 原始查詢
                - query_engine_report: QueryEngine報告內容
                - media_engine_report: MediaEngine報告內容  
                - insight_engine_report: InsightEngine報告內容
                - forum_logs: 論壇日誌內容
                - selected_template: 選擇的模板內容
                
        Returns:
            生成的HTML內容
        """
        logger.info("開始生成HTML報告...")
        
        try:
            # 準備LLM輸入數據
            llm_input = {
                "query": input_data.get('query', ''),
                "query_engine_report": input_data.get('query_engine_report', ''),
                "media_engine_report": input_data.get('media_engine_report', ''),
                "insight_engine_report": input_data.get('insight_engine_report', ''),
                "forum_logs": input_data.get('forum_logs', ''),
                "selected_template": input_data.get('selected_template', '')
            }
            
            # 轉換爲JSON格式傳遞給LLM
            message = json.dumps(llm_input, ensure_ascii=False, indent=2)
            
            # 調用LLM生成HTML
            response = self.llm_client.invoke(SYSTEM_PROMPT_HTML_GENERATION, message)
            
            # 處理響應（簡化版）
            processed_response = self.process_output(response)
            
            logger.info("HTML報告生成完成")
            return processed_response
            
        except Exception as e:
            logger.exception(f"HTML生成失敗: {str(e)}")
            # 返回備用HTML
            return self._generate_fallback_html(input_data)
    
    def mutate_state(self, input_data: Dict[str, Any], state: ReportState, **kwargs) -> ReportState:
        """
        修改報告狀態，添加生成的HTML內容
        
        Args:
            input_data: 輸入數據
            state: 當前報告狀態
            **kwargs: 額外參數
            
        Returns:
            更新後的報告狀態
        """
        # 生成HTML
        html_content = self.run(input_data, **kwargs)
        
        # 更新狀態
        state.html_content = html_content
        state.mark_completed()
        
        return state
    
    def process_output(self, output: str) -> str:
        """
        處理LLM輸出，提取HTML內容
        
        Args:
            output: LLM原始輸出
            
        Returns:
            HTML內容
        """
        try:
            logger.info(f"處理LLM原始輸出，長度: {len(output)} 字符")
            
            html_content = output.strip()
            
            # 清理markdown代碼塊標記（如果存在）
            if html_content.startswith('```html'):
                html_content = html_content[7:]  # 移除 '```html'
                if html_content.endswith('```'):
                    html_content = html_content[:-3]  # 移除結尾的 '```'
            elif html_content.startswith('```') and html_content.endswith('```'):
                html_content = html_content[3:-3]  # 移除前後的 '```'
            
            html_content = html_content.strip()
            
            # 如果內容爲空，返回原始輸出
            if not html_content:
                logger.info("處理後內容爲空，返回原始輸出")
                html_content = output
            
            logger.info(f"HTML處理完成，最終長度: {len(html_content)} 字符")
            return html_content
            
        except Exception as e:
            logger.exception(f"處理HTML輸出失敗: {str(e)}，返回原始輸出")
            return output
    
    def _generate_fallback_html(self, input_data: Dict[str, Any]) -> str:
        """
        生成備用HTML報告（當LLM失敗時使用）
        
        Args:
            input_data: 輸入數據
            
        Returns:
            備用HTML內容
        """
        logger.info("使用備用HTML生成方法")
        
        query = input_data.get('query', '智能輿情分析報告')
        query_report = input_data.get('query_engine_report', '')
        media_report = input_data.get('media_engine_report', '')
        insight_report = input_data.get('insight_engine_report', '')
        forum_logs = input_data.get('forum_logs', '')
        
        generation_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{query} - 智能輿情分析報告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid #3498db;
            background: #f8f9fa;
        }}
        .meta {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #666;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{query}</h1>
        
        <div class="meta">
            <strong>報告生成時間:</strong> {generation_time}<br>
            <strong>數據來源:</strong> QueryEngine、MediaEngine、InsightEngine、ForumEngine<br>
            <strong>報告類型:</strong> 綜合輿情分析報告
        </div>
        
        <h2>執行摘要</h2>
        <div class="section">
            本報告整合了多個分析引擎的研究結果，爲您提供全面的輿情分析洞察。
            通過對查詢主題"{query}"的深度分析，我們從多個維度展現了當前的輿情態勢。
        </div>
        
        {f'<h2>QueryEngine分析結果</h2><div class="section"><pre>{query_report}</pre></div>' if query_report else ''}
        
        {f'<h2>MediaEngine分析結果</h2><div class="section"><pre>{media_report}</pre></div>' if media_report else ''}
        
        {f'<h2>InsightEngine分析結果</h2><div class="section"><pre>{insight_report}</pre></div>' if insight_report else ''}
        
        {f'<h2>論壇監控數據</h2><div class="section"><pre>{forum_logs}</pre></div>' if forum_logs else ''}
        
        <h2>綜合結論</h2>
        <div class="section">
            基於多個分析引擎的綜合研究，我們對"{query}"主題進行了全面分析。
            各引擎從不同角度提供了深入洞察，爲決策提供了重要參考。
        </div>
        
        <div class="footer">
            <p>本報告由智能輿情分析平臺自動生成</p>
            <p>ReportEngine v1.0 | 生成時間: {generation_time}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_content
    

