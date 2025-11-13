"""
模板選擇節點
根據查詢內容和可用模板選擇最合適的報告模板
"""

import os
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_TEMPLATE_SELECTION


class TemplateSelectionNode(BaseNode):
    """模板選擇處理節點"""
    
    def __init__(self, llm_client, template_dir: str = "ReportEngine/report_template"):
        """
        初始化模板選擇節點
        
        Args:
            llm_client: LLM客戶端
            template_dir: 模板目錄路徑
        """
        super().__init__(llm_client, "TemplateSelectionNode")
        self.template_dir = template_dir
        
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        執行模板選擇
        
        Args:
            input_data: 包含查詢和報告內容的字典
                - query: 原始查詢
                - reports: 三個子agent的報告列表
                - forum_logs: 論壇日誌內容
                
        Returns:
            選擇的模板信息
        """
        logger.info("開始模板選擇...")
        
        query = input_data.get('query', '')
        reports = input_data.get('reports', [])
        forum_logs = input_data.get('forum_logs', '')
        
        # 獲取可用模板
        available_templates = self._get_available_templates()
        
        if not available_templates:
            logger.info("未找到預設模板，使用內置默認模板")
            return self._get_fallback_template()
        
        # 使用LLM進行模板選擇
        try:
            llm_result = self._llm_template_selection(query, reports, forum_logs, available_templates)
            if llm_result:
                return llm_result
        except Exception as e:
            logger.exception(f"LLM模板選擇失敗: {str(e)}")
        
        # 如果LLM選擇失敗，使用備選方案
        return self._get_fallback_template()
    

    
    def _llm_template_selection(self, query: str, reports: List[Any], forum_logs: str, 
                              available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """使用LLM進行模板選擇"""
        logger.info("嘗試使用LLM進行模板選擇...")
        
        # 構建模板列表
        template_list = "\n".join([f"- {t['name']}: {t['description']}" for t in available_templates])
        
        # 構建報告內容摘要
        reports_summary = ""
        if reports:
            reports_summary = "\n\n=== 分析引擎報告內容 ===\n"
            for i, report in enumerate(reports, 1):
                # 獲取報告內容，支持不同的數據格式
                if isinstance(report, dict):
                    content = report.get('content', str(report))
                elif hasattr(report, 'content'):
                    content = report.content
                else:
                    content = str(report)
                
                # 截斷過長的內容，保留前1000個字符
                if len(content) > 1000:
                    content = content[:1000] + "...(內容已截斷)"
                
                reports_summary += f"\n報告{i}內容:\n{content}\n"
        
        # 構建論壇日誌摘要
        forum_summary = ""
        if forum_logs and forum_logs.strip():
            forum_summary = "\n\n=== 三個引擎的討論內容 ===\n"
            # 截斷過長的日誌內容，保留前800個字符
            if len(forum_logs) > 800:
                forum_content = forum_logs[:800] + "...(討論內容已截斷)"
            else:
                forum_content = forum_logs
            forum_summary += forum_content
        
        user_message = f"""查詢內容: {query}

報告數量: {len(reports)} 個分析引擎報告
論壇日誌: {'有' if forum_logs else '無'}
{reports_summary}{forum_summary}

可用模板:
{template_list}

請根據查詢內容、報告內容和論壇日誌的具體情況，選擇最合適的模板。"""
        
        # 調用LLM
        response = self.llm_client.invoke(SYSTEM_PROMPT_TEMPLATE_SELECTION, user_message)
        
        # 檢查響應是否爲空
        if not response or not response.strip():
            logger.error("LLM返回空響應")
            return None
        
        logger.info(f"LLM原始響應: {response}")
        
        # 嘗試解析JSON響應
        try:
            # 清理響應文本
            cleaned_response = self._clean_llm_response(response)
            result = json.loads(cleaned_response)
            
            # 驗證選擇的模板是否存在
            selected_template_name = result.get('template_name', '')
            for template in available_templates:
                if template['name'] == selected_template_name or selected_template_name in template['name']:
                    logger.info(f"LLM選擇模板: {selected_template_name}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': result.get('selection_reason', 'LLM智能選擇')
                    }
            
            logger.error(f"LLM選擇的模板不存在: {selected_template_name}")
            return None
            
        except json.JSONDecodeError as e:
            logger.exception(f"JSON解析失敗: {str(e)}")
            # 嘗試從文本響應中提取模板信息
            return self._extract_template_from_text(response, available_templates)
    
    def _clean_llm_response(self, response: str) -> str:
        """清理LLM響應"""
        # 移除可能的markdown代碼塊標記
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        # 移除前後空白
        response = response.strip()
        
        return response
    
    def _extract_template_from_text(self, response: str, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """從文本響應中提取模板信息"""
        logger.info("嘗試從文本響應中提取模板信息")
        
        # 查找響應中是否包含模板名稱
        for template in available_templates:
            template_name_variants = [
                template['name'],
                template['name'].replace('.md', ''),
                template['name'].replace('模板', ''),
            ]
            
            for variant in template_name_variants:
                if variant in response:
                    logger.info(f"在響應中找到模板: {template['name']}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': '從文本響應中提取'
                    }
        
        return None
    
    def _get_available_templates(self) -> List[Dict[str, Any]]:
        """獲取可用的模板列表"""
        templates = []
        
        if not os.path.exists(self.template_dir):
            logger.error(f"模板目錄不存在: {self.template_dir}")
            return templates
        
        # 查找所有markdown模板文件
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.md'):
                template_path = os.path.join(self.template_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    template_name = filename.replace('.md', '')
                    description = self._extract_template_description(template_name)
                    
                    templates.append({
                        'name': template_name,
                        'path': template_path,
                        'content': content,
                        'description': description
                    })
                except Exception as e:
                    logger.exception(f"讀取模板文件失敗 {filename}: {str(e)}")
        
        return templates
    
    def _extract_template_description(self, template_name: str) -> str:
        """根據模板名稱生成描述"""
        if '企業品牌' in template_name:
            return "適用於企業品牌聲譽和形象分析"
        elif '市場競爭' in template_name:
            return "適用於市場競爭格局和對手分析"
        elif '日常' in template_name or '定期' in template_name:
            return "適用於日常監測和定期彙報"
        elif '政策' in template_name or '行業' in template_name:
            return "適用於政策影響和行業動態分析"
        elif '熱點' in template_name or '社會' in template_name:
            return "適用於社會熱點和公共事件分析"
        elif '突發' in template_name or '危機' in template_name:
            return "適用於突發事件和危機公關"
        
        return "通用報告模板"
    

    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """獲取備用默認模板（空模板，讓LLM自行發揮）"""
        logger.info("未找到合適模板，使用空模板讓LLM自行發揮")
        
        return {
            'template_name': '自由發揮模板',
            'template_content': '',
            'selection_reason': '未找到合適的預設模板，讓LLM根據內容自行設計報告結構'
        }
