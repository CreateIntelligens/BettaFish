"""
專爲 AI Agent 設計的輿情搜索工具集 (Tavily)

版本: 1.5
最後更新: 2025-08-22

此腳本將複雜的Tavily搜索功能分解爲一系列目標明確、參數極少的獨立工具，
專爲AI Agent調用而設計。Agent只需根據任務意圖選擇合適的工具，
無需理解複雜的參數組合。所有工具默認搜索“新聞”(topic='news')。

新特性:
- 新增 `basic_search_news` 工具，用於執行標準、通用的新聞搜索。
- 每個搜索結果現在都包含 `published_date` (新聞發佈日期)。

主要工具:
- basic_search_news: (新增) 執行標準、快速的通用新聞搜索。
- deep_search_news: 對主題進行最全面的深度分析。
- search_news_last_24_hours: 獲取24小時內的最新動態。
- search_news_last_week: 獲取過去一週的主要報道。
- search_images_for_news: 查找與新聞主題相關的圖片。
- search_news_by_date: 在指定的歷史日期範圍內搜索。
"""

import os
import sys
from typing import List, Dict, Any, Optional

# 添加utils目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG
from dataclasses import dataclass, field

# 運行前請確保已安裝Tavily庫: pip install tavily-python
try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("Tavily庫未安裝，請運行 `pip install tavily-python` 進行安裝。")

# --- 1. 數據結構定義 ---

@dataclass
class SearchResult:
    """
    網頁搜索結果數據類
    包含 published_date 屬性來存儲新聞發佈日期
    """
    title: str
    url: str
    content: str
    score: Optional[float] = None
    raw_content: Optional[str] = None
    published_date: Optional[str] = None

@dataclass
class ImageResult:
    """圖片搜索結果數據類"""
    url: str
    description: Optional[str] = None

@dataclass
class TavilyResponse:
    """封裝Tavily API的完整返回結果，以便在工具間傳遞"""
    query: str
    answer: Optional[str] = None
    results: List[SearchResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    response_time: Optional[float] = None


# --- 2. 核心客戶端與專用工具集 ---

class TavilyNewsAgency:
    """
    一個包含多種專用新聞輿情搜索工具的客戶端。
    每個公共方法都設計爲供 AI Agent 獨立調用的工具。
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客戶端。
        Args:
            api_key: Tavily API密鑰，若不提供則從環境變量 TAVILY_API_KEY 讀取。
        """
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("Tavily API Key未找到！請設置TAVILY_API_KEY環境變量或在初始化時提供")
        self._client = TavilyClient(api_key=api_key)

    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=TavilyResponse(query="搜索失敗"))
    def _search_internal(self, **kwargs) -> TavilyResponse:
        """內部通用的搜索執行器，所有工具最終都調用此方法"""
        try:
            kwargs['topic'] = 'general'
            api_params = {k: v for k, v in kwargs.items() if v is not None}
            response_dict = self._client.search(**api_params)
            
            search_results = [
                SearchResult(
                    title=item.get('title'),
                    url=item.get('url'),
                    content=item.get('content'),
                    score=item.get('score'),
                    raw_content=item.get('raw_content'),
                    published_date=item.get('published_date')
                ) for item in response_dict.get('results', [])
            ]
            
            image_results = [ImageResult(url=item.get('url'), description=item.get('description')) for item in response_dict.get('images', [])]

            return TavilyResponse(
                query=response_dict.get('query'), answer=response_dict.get('answer'),
                results=search_results, images=image_results,
                response_time=response_dict.get('response_time')
            )
        except Exception as e:
            print(f"搜索時發生錯誤: {str(e)}")
            raise e  # 讓重試機制捕獲並處理

    # --- Agent 可用的工具方法 ---

    def basic_search_news(self, query: str, max_results: int = 7) -> TavilyResponse:
        """
        【工具】基礎新聞搜索: 執行一次標準、快速的新聞搜索。
        這是最常用的通用搜索工具，適用於不確定需要何種特定搜索時。
        Agent可提供搜索查詢(query)和可選的最大結果數(max_results)。
        """
        print(f"--- TOOL: 基礎新聞搜索 (query: {query}) ---")
        return self._search_internal(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False
        )

    def deep_search_news(self, query: str) -> TavilyResponse:
        """
        【工具】深度新聞分析: 對一個主題進行最全面、最深入的搜索。
        返回AI生成的“高級”詳細摘要答案和最多20條最相關的新聞結果。適用於需要全面瞭解某個事件背景的場景。
        Agent只需提供搜索查詢(query)。
        """
        print(f"--- TOOL: 深度新聞分析 (query: {query}) ---")
        return self._search_internal(
            query=query, search_depth="advanced", max_results=20, include_answer="advanced"
        )

    def search_news_last_24_hours(self, query: str) -> TavilyResponse:
        """
        【工具】搜索24小時內新聞: 獲取關於某個主題的最新動態。
        此工具專門查找過去24小時內發佈的新聞。適用於追蹤突發事件或最新進展。
        Agent只需提供搜索查詢(query)。
        """
        print(f"--- TOOL: 搜索24小時內新聞 (query: {query}) ---")
        return self._search_internal(query=query, time_range='d', max_results=10)

    def search_news_last_week(self, query: str) -> TavilyResponse:
        """
        【工具】搜索本週新聞: 獲取關於某個主題過去一週內的主要新聞報道。
        適用於進行周度輿情總結或回顧。
        Agent只需提供搜索查詢(query)。
        """
        print(f"--- TOOL: 搜索本週新聞 (query: {query}) ---")
        return self._search_internal(query=query, time_range='w', max_results=10)

    def search_images_for_news(self, query: str) -> TavilyResponse:
        """
        【工具】查找新聞圖片: 搜索與某個新聞主題相關的圖片。
        此工具會返回圖片鏈接及描述，適用於需要爲報告或文章配圖的場景。
        Agent只需提供搜索查詢(query)。
        """
        print(f"--- TOOL: 查找新聞圖片 (query: {query}) ---")
        return self._search_internal(
            query=query, include_images=True, include_image_descriptions=True, max_results=5
        )

    def search_news_by_date(self, query: str, start_date: str, end_date: str) -> TavilyResponse:
        """
        【工具】按指定日期範圍搜索新聞: 在一個明確的歷史時間段內搜索新聞。
        這是唯一需要Agent提供詳細時間參數的工具。適用於需要對特定歷史事件進行分析的場景。
        Agent需要提供查詢(query)、開始日期(start_date)和結束日期(end_date)，格式均爲 'YYYY-MM-DD'。
        """
        print(f"--- TOOL: 按指定日期範圍搜索新聞 (query: {query}, from: {start_date}, to: {end_date}) ---")
        return self._search_internal(
            query=query, start_date=start_date, end_date=end_date, max_results=15
        )


# --- 3. 測試與使用示例 ---

def print_response_summary(response: TavilyResponse):
    """簡化的打印函數，用於展示測試結果，現在會顯示發佈日期"""
    if not response or not response.query:
        print("未能獲取有效響應。")
        return
        
    print(f"\n查詢: '{response.query}' | 耗時: {response.response_time}s")
    if response.answer:
        print(f"AI摘要: {response.answer[:120]}...")
    print(f"找到 {len(response.results)} 條網頁, {len(response.images)} 張圖片。")
    if response.results:
        first_result = response.results[0]
        date_info = f"(發佈於: {first_result.published_date})" if first_result.published_date else ""
        print(f"第一條結果: {first_result.title} {date_info}")
    print("-" * 60)


if __name__ == "__main__":
    # 在運行前，請確保您已設置 TAVILY_API_KEY 環境變量
    
    try:
        # 初始化“新聞社”客戶端，它內部包含了所有工具
        agency = TavilyNewsAgency()

        # 場景1: Agent 進行一次常規、快速的搜索
        response1 = agency.basic_search_news(query="奧運會最新賽況", max_results=5)
        print_response_summary(response1)

        # 場景2: Agent 需要全面瞭解“全球芯片技術競爭”的背景
        response2 = agency.deep_search_news(query="全球芯片技術競爭")
        print_response_summary(response2)

        # 場景3: Agent 需要追蹤“GTC大會”的最新消息
        response3 = agency.search_news_last_24_hours(query="Nvidia GTC大會 最新發布")
        print_response_summary(response3)
        
        # 場景4: Agent 需要爲一篇關於“自動駕駛”的週報查找素材
        response4 = agency.search_news_last_week(query="自動駕駛商業化落地")
        print_response_summary(response4)
        
        # 場景5: Agent 需要查找“韋伯太空望遠鏡”的新聞圖片
        response5 = agency.search_images_for_news(query="韋伯太空望遠鏡最新發現")
        print_response_summary(response5)

        # 場景6: Agent 需要研究2025年第一季度關於“人工智能法規”的新聞
        response6 = agency.search_news_by_date(
            query="人工智能法規",
            start_date="2025-01-01",
            end_date="2025-03-31"
        )
        print_response_summary(response6)

    except ValueError as e:
        print(f"初始化失敗: {e}")
        print("請確保 TAVILY_API_KEY 環境變量已正確設置。")
    except Exception as e:
        print(f"測試過程中發生未知錯誤: {e}")