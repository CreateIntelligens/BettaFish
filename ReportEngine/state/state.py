"""
Report Engine狀態管理
定義報告生成過程中的簡化狀態數據結構
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class ReportMetadata:
    """簡化的報告元數據"""
    query: str = ""                      # 原始查詢
    template_used: str = ""              # 使用的模板名稱
    generation_time: float = 0.0         # 生成耗時（秒）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "query": self.query,
            "template_used": self.template_used,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp
        }


@dataclass 
class ReportState:
    """簡化的報告狀態管理"""
    # 基本信息
    task_id: str = ""                    # 任務ID
    query: str = ""                      # 原始查詢
    status: str = "pending"              # 狀態: pending, processing, completed, failed
    
    # 輸入數據
    query_engine_report: str = ""        # QueryEngine報告
    media_engine_report: str = ""        # MediaEngine報告  
    insight_engine_report: str = ""      # InsightEngine報告
    forum_logs: str = ""                 # 論壇日誌
    
    # 處理結果
    selected_template: str = ""          # 選擇的模板
    html_content: str = ""               # 最終HTML內容
    
    # 元數據
    metadata: ReportMetadata = field(default_factory=ReportMetadata)
    
    def __post_init__(self):
        """初始化後處理"""
        if not self.task_id:
            self.task_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metadata.query = self.query
    
    def mark_processing(self):
        """標記爲處理中"""
        self.status = "processing"
    
    def mark_completed(self):
        """標記爲完成"""
        self.status = "completed"
    
    def mark_failed(self, error_message: str = ""):
        """標記爲失敗"""
        self.status = "failed"
        self.error_message = error_message
    
    def is_completed(self) -> bool:
        """檢查是否完成"""
        return self.status == "completed" and bool(self.html_content)
    
    def get_progress(self) -> float:
        """獲取進度百分比"""
        if self.status == "completed":
            return 100.0
        elif self.status == "processing":
            # 簡單的進度計算
            progress = 0.0
            if self.selected_template:
                progress += 30.0
            if self.html_content:
                progress += 70.0
            return progress
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "status": self.status,
            "progress": self.get_progress(),
            "selected_template": self.selected_template,
            "has_html_content": bool(self.html_content),
            "html_content_length": len(self.html_content) if self.html_content else 0,
            "metadata": self.metadata.to_dict()
        }
    
    def save_to_file(self, file_path: str):
        """保存狀態到文件"""
        try:
            state_data = self.to_dict()
            # 不保存完整的HTML內容到狀態文件（太大）
            state_data.pop("html_content", None)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存狀態文件失敗: {str(e)}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Optional["ReportState"]:
        """從文件加載狀態"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 創建ReportState對象
            state = cls(
                task_id=data.get("task_id", ""),
                query=data.get("query", ""),
                status=data.get("status", "pending"),
                selected_template=data.get("selected_template", "")
            )
            
            # 設置元數據
            metadata_data = data.get("metadata", {})
            state.metadata.template_used = metadata_data.get("template_used", "")
            state.metadata.generation_time = metadata_data.get("generation_time", 0.0)
            
            return state
            
        except Exception as e:
            print(f"加載狀態文件失敗: {str(e)}")
            return None