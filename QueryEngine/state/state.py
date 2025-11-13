"""
Deep Search Agent狀態管理
定義所有狀態數據結構和操作方法
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


@dataclass
class Search:
    """單個搜索結果的狀態"""
    query: str = ""                    # 搜索查詢
    url: str = ""                      # 搜索結果的鏈接
    title: str = ""                    # 搜索結果標題
    content: str = ""                  # 搜索返回的內容
    score: Optional[float] = None      # 相關度評分
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "query": self.query,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Search":
        """從字典創建Search對象"""
        return cls(
            query=data.get("query", ""),
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            score=data.get("score"),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class Research:
    """段落研究過程的狀態"""
    search_history: List[Search] = field(default_factory=list)     # 搜索記錄列表
    latest_summary: str = ""                                       # 當前段落的最新總結
    reflection_iteration: int = 0                                  # 反思迭代次數
    is_completed: bool = False                                     # 是否完成研究
    
    def add_search(self, search: Search):
        """添加搜索記錄"""
        self.search_history.append(search)
    
    def add_search_results(self, query: str, results: List[Dict[str, Any]]):
        """批量添加搜索結果"""
        for result in results:
            search = Search(
                query=query,
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=result.get("score")
            )
            self.add_search(search)
    
    def get_search_count(self) -> int:
        """獲取搜索次數"""
        return len(self.search_history)
    
    def increment_reflection(self):
        """增加反思次數"""
        self.reflection_iteration += 1
    
    def mark_completed(self):
        """標記爲完成"""
        self.is_completed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "search_history": [search.to_dict() for search in self.search_history],
            "latest_summary": self.latest_summary,
            "reflection_iteration": self.reflection_iteration,
            "is_completed": self.is_completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Research":
        """從字典創建Research對象"""
        search_history = [Search.from_dict(search_data) for search_data in data.get("search_history", [])]
        return cls(
            search_history=search_history,
            latest_summary=data.get("latest_summary", ""),
            reflection_iteration=data.get("reflection_iteration", 0),
            is_completed=data.get("is_completed", False)
        )


@dataclass
class Paragraph:
    """報告中單個段落的狀態"""
    title: str = ""                                                # 段落標題
    content: str = ""                                              # 段落的預期內容（初始規劃）
    research: Research = field(default_factory=Research)          # 研究進度
    order: int = 0                                                 # 段落順序
    
    def is_completed(self) -> bool:
        """檢查段落是否完成"""
        return self.research.is_completed and bool(self.research.latest_summary)
    
    def get_final_content(self) -> str:
        """獲取最終內容"""
        return self.research.latest_summary or self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "title": self.title,
            "content": self.content,
            "research": self.research.to_dict(),
            "order": self.order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paragraph":
        """從字典創建Paragraph對象"""
        research_data = data.get("research", {})
        research = Research.from_dict(research_data) if research_data else Research()
        
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            research=research,
            order=data.get("order", 0)
        )


@dataclass
class State:
    """整個報告的狀態"""
    query: str = ""                                                # 原始查詢
    report_title: str = ""                                         # 報告標題
    paragraphs: List[Paragraph] = field(default_factory=list)     # 段落列表
    final_report: str = ""                                         # 最終報告內容
    is_completed: bool = False                                     # 是否完成
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_paragraph(self, title: str, content: str) -> int:
        """
        添加段落
        
        Args:
            title: 段落標題
            content: 段落內容
            
        Returns:
            段落索引
        """
        order = len(self.paragraphs)
        paragraph = Paragraph(title=title, content=content, order=order)
        self.paragraphs.append(paragraph)
        self.update_timestamp()
        return order
    
    def get_paragraph(self, index: int) -> Optional[Paragraph]:
        """獲取指定索引的段落"""
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None
    
    def get_completed_paragraphs_count(self) -> int:
        """獲取已完成段落數量"""
        return sum(1 for p in self.paragraphs if p.is_completed())
    
    def get_total_paragraphs_count(self) -> int:
        """獲取總段落數量"""
        return len(self.paragraphs)
    
    def is_all_paragraphs_completed(self) -> bool:
        """檢查是否所有段落都完成"""
        return all(p.is_completed() for p in self.paragraphs) if self.paragraphs else False
    
    def mark_completed(self):
        """標記整個報告爲完成"""
        self.is_completed = True
        self.update_timestamp()
    
    def update_timestamp(self):
        """更新時間戳"""
        self.updated_at = datetime.now().isoformat()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """獲取進度摘要"""
        completed = self.get_completed_paragraphs_count()
        total = self.get_total_paragraphs_count()
        
        return {
            "total_paragraphs": total,
            "completed_paragraphs": completed,
            "progress_percentage": (completed / total * 100) if total > 0 else 0,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換爲字典格式"""
        return {
            "query": self.query,
            "report_title": self.report_title,
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "final_report": self.final_report,
            "is_completed": self.is_completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_json(self, indent: int = 2) -> str:
        """轉換爲JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """從字典創建State對象"""
        paragraphs = [Paragraph.from_dict(p_data) for p_data in data.get("paragraphs", [])]
        
        return cls(
            query=data.get("query", ""),
            report_title=data.get("report_title", ""),
            paragraphs=paragraphs,
            final_report=data.get("final_report", ""),
            is_completed=data.get("is_completed", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "State":
        """從JSON字符串創建State對象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """保存狀態到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "State":
        """從文件加載狀態"""
        with open(filepath, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return cls.from_json(json_str)
