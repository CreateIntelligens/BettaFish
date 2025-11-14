"""
Report Agent主類
整合所有模塊，實現完整的報告生成流程
"""

import json
import os
from loguru import logger
from datetime import datetime
from typing import Optional, Dict, Any, List

from .llms import LLMClient
from .nodes import (
    TemplateSelectionNode,
    HTMLGenerationNode
)
from .state import ReportState
from .utils.config import settings, Settings


class FileCountBaseline:
    """文件數量基準管理器"""
    
    def __init__(self):
        self.baseline_file = 'logs/report_baseline.json'
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, int]:
        """加載基準數據"""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.exception(f"加載基準數據失敗: {e}")
        return {}
    
    def _save_baseline(self):
        """保存基準數據"""
        try:
            os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(self.baseline_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"保存基準數據失敗: {e}")
    
    def initialize_baseline(self, directories: Dict[str, str]) -> Dict[str, int]:
        """初始化文件數量基準"""
        current_counts = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
            else:
                current_counts[engine] = 0
        
        # 保存基準數據
        self.baseline_data = current_counts.copy()
        self._save_baseline()
        
        logger.info(f"文件數量基準已初始化: {current_counts}")
        return current_counts
    
    def check_new_files(self, directories: Dict[str, str]) -> Dict[str, Any]:
        """檢查是否有新文件"""
        current_counts = {}
        new_files_found = {}
        all_have_new = True
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                current_counts[engine] = len(md_files)
                baseline_count = self.baseline_data.get(engine, 0)
                
                if current_counts[engine] > baseline_count:
                    new_files_found[engine] = current_counts[engine] - baseline_count
                else:
                    new_files_found[engine] = 0
                    all_have_new = False
            else:
                current_counts[engine] = 0
                new_files_found[engine] = 0
                all_have_new = False
        
        return {
            'ready': all_have_new,
            'baseline_counts': self.baseline_data,
            'current_counts': current_counts,
            'new_files_found': new_files_found,
            'missing_engines': [engine for engine, count in new_files_found.items() if count == 0]
        }
    
    def get_latest_files(self, directories: Dict[str, str]) -> Dict[str, str]:
        """獲取每個目錄的最新文件"""
        latest_files = {}
        
        for engine, directory in directories.items():
            if os.path.exists(directory):
                md_files = [f for f in os.listdir(directory) if f.endswith('.md')]
                if md_files:
                    latest_file = max(md_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
                    latest_files[engine] = os.path.join(directory, latest_file)
        
        return latest_files


class ReportAgent:
    """Report Agent主類"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        初始化Report Agent
        
        Args:
            config: 配置對象，如果不提供則自動加載
        """
        # 加載配置
        self.config = config or settings
        
        # 初始化文件基準管理器
        self.file_baseline = FileCountBaseline()
        
        # 初始化日誌
        self._setup_logging()
        
        # 初始化LLM客戶端
        self.llm_client = self._initialize_llm()
        
        # 初始化節點
        self._initialize_nodes()
        
        # 初始化文件數量基準
        self._initialize_file_baseline()
        
        # 狀態
        self.state = ReportState()

        # 確保輸出目錄存在
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        logger.info("Report Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        
    def _setup_logging(self):
        """設置日誌"""
        # 確保日誌目錄存在
        log_dir = os.path.dirname(self.config.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

        # 創建專用的logger，避免與其他模塊衝突
        logger.add(self.config.LOG_FILE, level="INFO")
        
    def _initialize_file_baseline(self):
        """初始化文件數量基準"""
        directories = {
            'insight': 'insight_engine_streamlit_reports',
            'media': 'media_engine_streamlit_reports',
            'query': 'query_engine_streamlit_reports'
        }
        self.file_baseline.initialize_baseline(directories)
    
    def _initialize_llm(self) -> LLMClient:
        """初始化LLM客戶端"""
        return LLMClient(
            api_key=self.config.REPORT_ENGINE_API_KEY,
            model_name=self.config.REPORT_ENGINE_MODEL_NAME,
            base_url=self.config.REPORT_ENGINE_BASE_URL,
        )
    
    def _initialize_nodes(self):
        """初始化處理節點"""
        self.template_selection_node = TemplateSelectionNode(
            self.llm_client,
            self.config.TEMPLATE_DIR
        )
        self.html_generation_node = HTMLGenerationNode(self.llm_client)
    
    def generate_report(self, query: str, reports: List[Any], forum_logs: str = "", 
                       custom_template: str = "", save_report: bool = True) -> str:
        """
        生成綜合報告
        
        Args:
            query: 原始查詢
            reports: 三個子agent的報告列表（按順序：QueryEngine, MediaEngine, InsightEngine）
            forum_logs: 論壇日誌內容
            custom_template: 用戶自定義模板（可選）
            save_report: 是否保存報告到文件

        Returns:
            dict: 包含HTML內容與保存文件信息
        """
        start_time = datetime.now()

        # 為新的查詢重置狀態，確保文件命名信息完整
        self.state = ReportState(query=query)
        self.state.metadata.query = query
        self.state.query = query
        self.state.mark_processing()

        logger.info(f"開始生成報告: {query}")
        logger.info(f"輸入數據 - 報告數量: {len(reports)}, 論壇日誌長度: {len(forum_logs)}")
        
        try:
            # Step 1: 模板選擇
            template_result = self._select_template(query, reports, forum_logs, custom_template)
            
            # Step 2: 直接生成HTML報告
            html_report = self._generate_html_report(query, reports, forum_logs, template_result)

            # Step 3: 保存報告
            saved_files = {}
            if save_report:
                saved_files = self._save_report(html_report)
            
            # 更新生成時間
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            self.state.metadata.generation_time = generation_time
            
            logger.info(f"報告生成完成，耗時: {generation_time:.2f} 秒")
            
            return {
                'html_content': html_report,
                **saved_files
            }
            
        except Exception as e:
            logger.exception(f"報告生成過程中發生錯誤: {str(e)}")
            raise e
    
    def _select_template(self, query: str, reports: List[Any], forum_logs: str, custom_template: str):
        """選擇報告模板"""
        logger.info("選擇報告模板...")
        
        # 如果用戶提供了自定義模板，直接使用
        if custom_template:
            logger.info("使用用戶自定義模板")
            return {
                'template_name': 'custom',
                'template_content': custom_template,
                'selection_reason': '用戶指定的自定義模板'
            }
        
        template_input = {
            'query': query,
            'reports': reports,
            'forum_logs': forum_logs
        }
        
        try:
            template_result = self.template_selection_node.run(template_input)
            
            # 更新狀態
            self.state.metadata.template_used = template_result['template_name']
            
            logger.info(f"選擇模板: {template_result['template_name']}")
            logger.info(f"選擇理由: {template_result['selection_reason']}")
            
            return template_result
        except Exception as e:
            logger.error(f"模板選擇失敗，使用默認模板: {str(e)}")
            # 直接使用備用模板
            fallback_template = {
                'template_name': '社會公共熱點事件分析報告模板',
                'template_content': self._get_fallback_template_content(),
                'selection_reason': '模板選擇失敗，使用默認社會熱點事件分析模板'
            }
            self.state.metadata.template_used = fallback_template['template_name']
            return fallback_template
    
    def _generate_html_report(self, query: str, reports: List[Any], forum_logs: str, template_result: Dict[str, Any]) -> str:
        """生成HTML報告"""
        logger.info("多輪生成HTML報告...")
        
        # 準備報告內容，確保有3個報告
        query_report = reports[0] if len(reports) > 0 else ""
        media_report = reports[1] if len(reports) > 1 else ""
        insight_report = reports[2] if len(reports) > 2 else ""
        
        # 轉換爲字符串格式
        query_report = str(query_report) if query_report else ""
        media_report = str(media_report) if media_report else ""
        insight_report = str(insight_report) if insight_report else ""
        
        html_input = {
            'query': query,
            'query_engine_report': query_report,
            'media_engine_report': media_report,
            'insight_engine_report': insight_report,
            'forum_logs': forum_logs,
            'selected_template': template_result.get('template_content', '')
        }
        
        # 使用HTML生成節點生成報告
        html_content = self.html_generation_node.run(html_input)
        
        # 更新狀態
        self.state.html_content = html_content
        self.state.mark_completed()
        
        logger.info("HTML報告生成完成")
        return html_content
    
    def _get_fallback_template_content(self) -> str:
        """獲取備用模板內容"""
        return """# 社會公共熱點事件分析報告

## 執行摘要
本報告針對當前社會熱點事件進行綜合分析，整合了多方信息源的觀點和數據。

## 事件概況
### 基本信息
- 事件性質：{event_nature}
- 發生時間：{event_time}
- 涉及範圍：{event_scope}

## 輿情態勢分析
### 整體趨勢
{sentiment_analysis}

### 主要觀點分佈
{opinion_distribution}

## 媒體報道分析
### 主流媒體態度
{media_analysis}

### 報道重點
{report_focus}

## 社會影響評估
### 直接影響
{direct_impact}

### 潛在影響
{potential_impact}

## 應對建議
### 即時措施
{immediate_actions}

### 長期策略
{long_term_strategy}

## 結論與展望
{conclusion}

---
*報告類型：社會公共熱點事件分析*
*生成時間：{generation_time}*
"""
    
    def _save_report(self, html_content: str):
        """保存報告到文件"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.metadata.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"final_report_{query_safe}_{timestamp}.html"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # 保存HTML報告
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        abs_report_path = os.path.abspath(filepath)
        rel_report_path = os.path.relpath(abs_report_path, os.getcwd())
        logger.info(f"報告已保存到: {abs_report_path}")

        # 保存狀態
        state_filename = f"report_state_{query_safe}_{timestamp}.json"
        state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
        self.state.save_to_file(state_filepath)
        abs_state_path = os.path.abspath(state_filepath)
        rel_state_path = os.path.relpath(abs_state_path, os.getcwd())
        logger.info(f"狀態已保存到: {abs_state_path}")

        return {
            'report_filename': filename,
            'report_filepath': abs_report_path,
            'report_relative_path': rel_report_path,
            'state_filename': state_filename,
            'state_filepath': abs_state_path,
            'state_relative_path': rel_state_path
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """獲取進度摘要"""
        return self.state.to_dict()
    
    def load_state(self, filepath: str):
        """從文件加載狀態"""
        self.state = ReportState.load_from_file(filepath)
        logger.info(f"狀態已從 {filepath} 加載")
    
    def save_state(self, filepath: str):
        """保存狀態到文件"""
        self.state.save_to_file(filepath)
        logger.info(f"狀態已保存到 {filepath}")
    
    def check_input_files(self, insight_dir: str, media_dir: str, query_dir: str, forum_log_path: str) -> Dict[str, Any]:
        """
        檢查輸入文件是否準備就緒（基於文件數量增加）
        
        Args:
            insight_dir: InsightEngine報告目錄
            media_dir: MediaEngine報告目錄
            query_dir: QueryEngine報告目錄
            forum_log_path: 論壇日誌文件路徑
            
        Returns:
            檢查結果字典
        """
        # 檢查各個報告目錄的文件數量變化
        directories = {
            'insight': insight_dir,
            'media': media_dir,
            'query': query_dir
        }
        
        # 使用文件基準管理器檢查新文件
        check_result = self.file_baseline.check_new_files(directories)
        
        # 檢查論壇日誌
        forum_ready = os.path.exists(forum_log_path)
        
        # 構建返回結果
        result = {
            'ready': check_result['ready'] and forum_ready,
            'baseline_counts': check_result['baseline_counts'],
            'current_counts': check_result['current_counts'],
            'new_files_found': check_result['new_files_found'],
            'missing_files': [],
            'files_found': [],
            'latest_files': {}
        }
        
        # 構建詳細信息
        for engine, new_count in check_result['new_files_found'].items():
            current_count = check_result['current_counts'][engine]
            baseline_count = check_result['baseline_counts'].get(engine, 0)
            
            if new_count > 0:
                result['files_found'].append(f"{engine}: {current_count}個文件 (新增{new_count}個)")
            else:
                result['missing_files'].append(f"{engine}: {current_count}個文件 (基準{baseline_count}個，無新增)")
        
        # 檢查論壇日誌
        if forum_ready:
            result['files_found'].append(f"forum: {os.path.basename(forum_log_path)}")
        else:
            result['missing_files'].append("forum: 日誌文件不存在")
        
        # 獲取最新文件路徑（用於實際報告生成）
        if result['ready']:
            result['latest_files'] = self.file_baseline.get_latest_files(directories)
            if forum_ready:
                result['latest_files']['forum'] = forum_log_path
        
        return result
    
    def load_input_files(self, file_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        加載輸入文件內容
        
        Args:
            file_paths: 文件路徑字典
            
        Returns:
            加載的內容字典
        """
        content = {
            'reports': [],
            'forum_logs': ''
        }
        
        # 加載報告文件
        engines = ['query', 'media', 'insight']
        for engine in engines:
            if engine in file_paths:
                try:
                    with open(file_paths[engine], 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    content['reports'].append(report_content)
                    logger.info(f"已加載 {engine} 報告: {len(report_content)} 字符")
                except Exception as e:
                    logger.exception(f"加載 {engine} 報告失敗: {str(e)}")
                    content['reports'].append("")
        
        # 加載論壇日誌
        if 'forum' in file_paths:
            try:
                with open(file_paths['forum'], 'r', encoding='utf-8') as f:
                    content['forum_logs'] = f.read()
                logger.info(f"已加載論壇日誌: {len(content['forum_logs'])} 字符")
            except Exception as e:
                logger.exception(f"加載論壇日誌失敗: {str(e)}")
        
        return content


def create_agent(config_file: Optional[str] = None) -> ReportAgent:
    """
    創建Report Agent實例的便捷函數
    
    Args:
        config_file: 配置文件路徑
        
    Returns:
        ReportAgent實例
    """

    config = Settings() # 以空配置初始化，而從環境變量初始化
    return ReportAgent(config)
