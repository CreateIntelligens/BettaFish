"""
專爲 AI Agent 設計的多模態搜索工具集 (Bocha)

版本: 1.1
最後更新: 2025-08-22

此腳本將複雜的 Bocha AI Search 功能分解爲一系列目標明確、參數極少的獨立工具，
專爲 AI Agent 調用而設計。Agent 只需根據任務意圖（如常規搜索、查找結構化數據或時效性新聞）
選擇合適的工具，無需理解複雜的參數組合。

核心特性:
- 強大多模態能力: 能同時返回網頁、圖片、AI總結、追問建議，以及豐富的“模態卡”結構化數據。
- 模態卡支持: 針對天氣、股票、匯率、百科、醫療等特定查詢，可直接返回結構化數據卡片，便於Agent直接解析和使用。

主要工具:
- comprehensive_search: 執行全面搜索，返回網頁、圖片、AI總結及可能的模態卡。
- search_for_structured_data: 專門用於查詢天氣、股票、匯率等可觸發“模態卡”的結構化信息。
- web_search_only: 執行純網頁搜索，不請求AI總結，速度更快。
- search_last_24_hours: 獲取過去24小時內的最新信息。
- search_last_week: 獲取過去一週內的主要報道。
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional, Literal

from loguru import logger
from config import settings

# 運行前請確保已安裝 requests 庫: pip install requests
try:
    import requests
except ImportError:
    raise ImportError("requests 庫未安裝，請運行 `pip install requests` 進行安裝。")

# 添加utils目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG

# --- 1. 數據結構定義 ---
from dataclasses import dataclass, field

@dataclass
class WebpageResult:
    """網頁搜索結果"""
    name: str
    url: str
    snippet: str
    display_url: Optional[str] = None
    date_last_crawled: Optional[str] = None

@dataclass
class ImageResult:
    """圖片搜索結果"""
    name: str
    content_url: str
    host_page_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class ModalCardResult:
    """
    模態卡結構化數據結果
    這是 Bocha 搜索的核心特色，用於返回特定類型的結構化信息。
    """
    card_type: str  # 例如: weather_china, stock, baike_pro, medical_common
    content: Dict[str, Any]  # 解析後的JSON內容

@dataclass
class BochaResponse:
    """封裝 Bocha API 的完整返回結果，以便在工具間傳遞"""
    query: str
    conversation_id: Optional[str] = None
    answer: Optional[str] = None  # AI生成的總結答案
    follow_ups: List[str] = field(default_factory=list) # AI生成的追問
    webpages: List[WebpageResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    modal_cards: List[ModalCardResult] = field(default_factory=list)


# --- 2. 核心客戶端與專用工具集 ---

class BochaMultimodalSearch:
    """
    一個包含多種專用多模態搜索工具的客戶端。
    每個公共方法都設計爲供 AI Agent 獨立調用的工具。
    """

    BOCHA_BASE_URL = settings.BOCHA_BASE_URL or "https://api.bochaai.com/v1/ai-search"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客戶端。
        Args:
            api_key: Bocha API密鑰，若不提供則從環境變量 BOCHA_API_KEY 讀取。
        """
        if api_key is None:
            api_key = settings.BOCHA_WEB_SEARCH_API_KEY
            if not api_key:
                raise ValueError("Bocha API Key未找到！請設置 BOCHA_API_KEY 環境變量或在初始化時提供")

        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }

    def _parse_search_response(self, response_dict: Dict[str, Any], query: str) -> BochaResponse:
        """從API的原始字典響應中解析出結構化的BochaResponse對象"""

        final_response = BochaResponse(query=query)
        final_response.conversation_id = response_dict.get('conversation_id')

        messages = response_dict.get('messages', [])
        for msg in messages:
            role = msg.get('role')
            if role != 'assistant':
                continue

            msg_type = msg.get('type')
            content_type = msg.get('content_type')
            content_str = msg.get('content', '{}')

            try:
                content_data = json.loads(content_str)
            except json.JSONDecodeError:
                # 如果內容不是合法的JSON字符串（例如純文本的answer），則直接使用
                content_data = content_str

            if msg_type == 'answer' and content_type == 'text':
                final_response.answer = content_data

            elif msg_type == 'follow_up' and content_type == 'text':
                final_response.follow_ups.append(content_data)

            elif msg_type == 'source':
                if content_type == 'webpage':
                    web_results = content_data.get('value', [])
                    for item in web_results:
                        final_response.webpages.append(WebpageResult(
                            name=item.get('name'),
                            url=item.get('url'),
                            snippet=item.get('snippet'),
                            display_url=item.get('displayUrl'),
                            date_last_crawled=item.get('dateLastCrawled')
                        ))
                elif content_type == 'image':
                    final_response.images.append(ImageResult(
                        name=content_data.get('name'),
                        content_url=content_data.get('contentUrl'),
                        host_page_url=content_data.get('hostPageUrl'),
                        thumbnail_url=content_data.get('thumbnailUrl'),
                        width=content_data.get('width'),
                        height=content_data.get('height')
                    ))
                # 所有其他 content_type 都視爲模態卡
                else:
                    final_response.modal_cards.append(ModalCardResult(
                        card_type=content_type,
                        content=content_data
                    ))

        return final_response


    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=BochaResponse(query="搜索失敗"))
    def _search_internal(self, **kwargs) -> BochaResponse:
        """內部通用的搜索執行器，所有工具最終都調用此方法"""
        query = kwargs.get("query", "Unknown Query")
        payload = {
            "stream": False,  # Agent工具通常使用非流式以獲取完整結果
        }
        payload.update(kwargs)

        try:
            response = requests.post(self.BOCHA_BASE_URL, headers=self._headers, json=payload, timeout=30)
            response.raise_for_status()  # 如果HTTP狀態碼是4xx或5xx，則拋出異常

            response_dict = response.json()
            if response_dict.get("code") != 200:
                logger.error(f"API返回錯誤: {response_dict.get('msg', '未知錯誤')}")
                return BochaResponse(query=query)

            return self._parse_search_response(response_dict, query)

        except requests.exceptions.RequestException as e:
            logger.exception(f"搜索時發生網絡錯誤: {str(e)}")
            raise e  # 讓重試機制捕獲並處理
        except Exception as e:
            logger.exception(f"處理響應時發生未知錯誤: {str(e)}")
            raise e  # 讓重試機制捕獲並處理

    # --- Agent 可用的工具方法 ---

    def comprehensive_search(self, query: str, max_results: int = 10) -> BochaResponse:
        """
        【工具】全面綜合搜索: 執行一次標準的、包含所有信息類型的綜合搜索。
        返回網頁、圖片、AI總結、追問建議和可能的模態卡。這是最常用的通用搜索工具。
        Agent可提供搜索查詢(query)和可選的最大結果數(max_results)。
        """
        logger.info(f"--- TOOL: 全面綜合搜索 (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=True  # 開啓AI總結
        )

    def web_search_only(self, query: str, max_results: int = 15) -> BochaResponse:
        """
        【工具】純網頁搜索: 只獲取網頁鏈接和摘要，不請求AI生成答案。
        適用於需要快速獲取原始網頁信息，而不需要AI額外分析的場景。速度更快，成本更低。
        """
        logger.info(f"--- TOOL: 純網頁搜索 (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=False # 關閉AI總結
        )

    def search_for_structured_data(self, query: str) -> BochaResponse:
        """
        【工具】結構化數據查詢: 專門用於可能觸發“模態卡”的查詢。
        當Agent意圖是查詢天氣、股票、匯率、百科定義、火車票、汽車參數等結構化信息時，應優先使用此工具。
        它會返回所有信息，但Agent應重點關注結果中的 `modal_cards` 部分。
        """
        logger.info(f"--- TOOL: 結構化數據查詢 (query: {query}) ---")
        # 實現上與 comprehensive_search 相同，但通過命名和文檔引導Agent的意圖
        return self._search_internal(
            query=query,
            count=5, # 結構化查詢通常不需要太多網頁結果
            answer=True
        )

    def search_last_24_hours(self, query: str) -> BochaResponse:
        """
        【工具】搜索24小時內信息: 獲取關於某個主題的最新動態。
        此工具專門查找過去24小時內發佈的內容。適用於追蹤突發事件或最新進展。
        """
        logger.info(f"--- TOOL: 搜索24小時內信息 (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneDay', answer=True)

    def search_last_week(self, query: str) -> BochaResponse:
        """
        【工具】搜索本週信息: 獲取關於某個主題過去一週內的主要報道。
        適用於進行周度輿情總結或回顧。
        """
        logger.info(f"--- TOOL: 搜索本週信息 (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneWeek', answer=True)


# --- 3. 測試與使用示例 ---

def print_response_summary(response: BochaResponse):
    """簡化的打印函數，用於展示測試結果"""
    if not response or not response.query:
        logger.error("未能獲取有效響應。")
        return

    logger.info(f"\n查詢: '{response.query}' | 會話ID: {response.conversation_id}")
    if response.answer:
        logger.info(f"AI摘要: {response.answer[:150]}...")

    logger.info(f"找到 {len(response.webpages)} 個網頁, {len(response.images)} 張圖片, {len(response.modal_cards)} 個模態卡。")

    if response.modal_cards:
        first_card = response.modal_cards[0]
        logger.info(f"第一個模態卡類型: {first_card.card_type}")

    if response.webpages:
        first_result = response.webpages[0]
        logger.info(f"第一條網頁結果: {first_result.name}")

    if response.follow_ups:
        logger.info(f"建議追問: {response.follow_ups}")

    logger.info("-" * 60)


if __name__ == "__main__":
    # 在運行前，請確保您已設置 BOCHA_API_KEY 環境變量

    try:
        # 初始化多模態搜索客戶端，它內部包含了所有工具
        search_client = BochaMultimodalSearch()

        # 場景1: Agent進行一次常規的、需要AI總結的綜合搜索
        response1 = search_client.comprehensive_search(query="人工智能對未來教育的影響")
        print_response_summary(response1)

        # 場景2: Agent需要查詢特定結構化信息 - 天氣
        response2 = search_client.search_for_structured_data(query="上海明天天氣怎麼樣")
        print_response_summary(response2)
        # 深度解析第一個模態卡
        if response2.modal_cards and response2.modal_cards[0].card_type == 'weather_china':
             logger.info("天氣模態卡詳情:", json.dumps(response2.modal_cards[0].content, indent=2, ensure_ascii=False))


        # 場景3: Agent需要查詢特定結構化信息 - 股票
        response3 = search_client.search_for_structured_data(query="東方財富股票")
        print_response_summary(response3)

        # 場景4: Agent需要追蹤某個事件的最新進展
        response4 = search_client.search_last_24_hours(query="C929大飛機最新消息")
        print_response_summary(response4)

        # 場景5: Agent只需要快速獲取網頁信息，不需要AI總結
        response5 = search_client.web_search_only(query="Python dataclasses用法")
        print_response_summary(response5)

        # 場景6: Agent需要回顧一週內關於某項技術的新聞
        response6 = search_client.search_last_week(query="量子計算商業化")
        print_response_summary(response6)

        '''下面是測試程序的輸出：
        --- TOOL: 全面綜合搜索 (query: 人工智能對未來教育的影響) ---

查詢: '人工智能對未來教育的影響' | 會話ID: bf43bfe4c7bb4f7b8a3945515d8ab69e
AI摘要: 人工智能對未來教育有着多方面的影響。

從積極影響來看：
- 在教學資源方面，人工智能有助於教育資源的均衡分配[引用:4]。例如通過人工智能雲平臺，可以實現優質資源的共享，這對於偏遠地區來說意義重大，能讓那裏的學生也接觸到優質的教育內 容，一定程度上緩解師資短缺的問題，因爲AI驅動的智能教學助手或虛擬...
找到 10 個網頁, 1 張圖片, 1 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 人工智能如何影響教育變革
建議追問: [['人工智能將如何改變未來的教育模式？', '在未來教育中，人工智能會給教師帶來哪些挑戰？', '未來教育中，學生如何利用人工智能提升學習效果？']]
------------------------------------------------------------
--- TOOL: 結構化數據查詢 (query: 上海明天天氣怎麼樣) ---

查詢: '上海明天天氣怎麼樣' | 會話ID: e412aa1548cd43a295430e47a62adda2
AI摘要: 根據所給信息，無法確定上海明天的天氣情況。

首先，所提供的信息都是關於2025年8月22日的天氣狀況，包括當天的氣溫、降水、風力、溼度以及高溫預警等信息[引用:1][引用:2][引用:3][引用:5]。然而，這些信息沒有涉及到明天（8月23 日）天氣的預測內容。雖然提到了副熱帶高壓一直到8月底高溫都...
找到 5 個網頁, 1 張圖片, 2 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 今日衝擊38!上海八月高溫天數和夏季持續高溫天數有望雙雙破紀錄_天氣_低壓_氣象站
建議追問: [['能告訴我上海明天的氣溫範圍嗎？', '上海明天會有降雨嗎？', '上海明天的天氣是晴天還是陰天呢？']]
------------------------------------------------------------
--- TOOL: 結構化數據查詢 (query: 東方財富股票) ---

查詢: '東方財富股票' | 會話ID: 584d62ed97834473b967127852e1eaa0
AI摘要: 僅根據提供的上下文，無法確切獲取東方財富股票的相關信息。

從給出的這些數據來看，並沒有直接表明與東方財富股票相關的特定數據。例如，沒有東方財富股票的漲跌幅情況、成交量、市值等具體數據[引用:1][引用:3]。也沒有涉及東方財富股票在研報 、評級方面的信息[引用:2]。同時，上下文裏關於股票價格、成交...
找到 5 個網頁, 1 張圖片, 2 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 股票價格_分時成交_行情_走勢圖—東方財富網
建議追問: [['東方財富股票近期的走勢如何？', '東方財富股票有哪些主要的投資亮點？', '東方財富股票的歷史最高和最低股價是多少？']]
------------------------------------------------------------
--- TOOL: 搜索24小時內信息 (query: C929大飛機最新消息) ---

查詢: 'C929大飛機最新消息' | 會話ID: 5904021dc29d497e938e04db18d7f2e2
AI摘要: 根據提供的上下文，沒有關於C929大飛機的直接消息，無法確切給出C929大飛機的最新消息。

目前提供的上下文涵蓋了衆多航空領域相關事件，但多是圍繞波音787、空客A380相關專家的人事變動、國產飛機“C909雲端之旅”、科德數控的營收情況、俄製航空發動機供應相關以及其他非C929大飛機相關的內容。...
找到 10 個網頁, 1 張圖片, 1 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 放棄美國千萬年薪,波音787頂尖專家回國,或可協助破解C929
建議追問: [['C929大飛機目前的研發進度如何？', '有沒有關於C929大飛機預計首飛時間的消息？', 'C929大飛機在技術創新方面有哪些新進展？']]
------------------------------------------------------------
--- TOOL: 純網頁搜索 (query: Python dataclasses用法) ---

查詢: 'Python dataclasses用法' | 會話ID: 74c742759d2e4b17b52d8b735ce24537
找到 15 個網頁, 1 張圖片, 1 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 不可不知的dataclasses  python小知識_python dataclasses-CSDN博客
------------------------------------------------------------
--- TOOL: 搜索本週信息 (query: 量子計算商業化) ---

AI摘要: 量子計算商業化正在逐步推進。

量子計算商業化有着多方面的體現和推動因素。從國際上看，美國能源部橡樹嶺國家實驗室選擇IQM Radiance作爲其首臺本地部署的量子計算機，計劃於2025年第三季度交付並集成至高性能計算系統中[引用:4]；英國量子計算公司Oxford Ionics的全棧離子阱量子計算...
找到 10 個網頁, 1 張圖片, 1 個模態卡。
第一個模態卡類型: video
第一條網頁結果: 量子計算商業潛力釋放正酣,微美全息(WIMI.US)創新科技卡位“生態高地”
建議追問: [['量子計算商業化目前有哪些成功的案例？', '哪些公司在推動量子計算商業化進程？', '量子計算商業化面臨的主要挑戰是什麼？']]
------------------------------------------------------------'''

    except ValueError as e:
        logger.exception(f"初始化失敗: {e}")
        logger.error("請確保 BOCHA_API_KEY 環境變量已正確設置。")
    except Exception as e:
        logger.exception(f"測試過程中發生未知錯誤: {e}")