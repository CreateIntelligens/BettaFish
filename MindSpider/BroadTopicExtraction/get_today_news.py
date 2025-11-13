#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction模塊 - 新聞獲取和收集
整合新聞API調用和數據庫存儲功能
"""

import sys
import asyncio
import httpx
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from BroadTopicExtraction.database_manager import DatabaseManager
except ImportError as e:
    raise ImportError(f"導入模塊失敗: {e}")

# 新聞API基礎URL
BASE_URL = "https://newsnow.busiyi.world"

# 新聞源中文名稱映射
SOURCE_NAMES = {
    "weibo": "微博熱搜",
    "zhihu": "知乎熱榜",
    "bilibili-hot-search": "B站熱搜",
    "toutiao": "今日頭條",
    "douyin": "抖音熱榜",
    "github-trending-today": "GitHub趨勢",
    "coolapk": "酷安熱榜",
    "tieba": "百度貼吧",
    "wallstreetcn": "華爾街見聞",
    "thepaper": "澎湃新聞",
    "cls-hot": "財聯社",
    "xueqiu": "雪球熱榜"
}

class NewsCollector:
    """新聞收集器 - 整合API調用和數據庫存儲"""
    
    def __init__(self):
        """初始化新聞收集器"""
        self.db_manager = DatabaseManager()
        self.supported_sources = list(SOURCE_NAMES.keys())
    
    def close(self):
        """關閉資源"""
        if self.db_manager:
            self.db_manager.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== 新聞API調用 ====================
    
    async def fetch_news(self, source: str) -> dict:
        """從指定源獲取最新新聞"""
        url = f"{BASE_URL}/api/s?id={source}&latest"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": BASE_URL,
            "Connection": "keep-alive",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # 解析JSON響應
                data = response.json()
                return {
                    "source": source,
                    "status": "success",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
        except httpx.TimeoutException:
            return {
                "source": source,
                "status": "timeout",
                "error": f"請求超時: {source}({url})",
                "timestamp": datetime.now().isoformat()
            }
        except httpx.HTTPStatusError as e:
            return {
                "source": source,
                "status": "http_error",
                "error": f"HTTP錯誤: {source}({url}) - {e.response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "source": source,
                "status": "error",
                "error": f"未知錯誤: {source}({url}) - {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_popular_news(self, sources: List[str] = None) -> List[dict]:
        """獲取熱門新聞"""
        if sources is None:
            sources = list(SOURCE_NAMES.keys())
        
        logger.info(f"正在獲取 {len(sources)} 個新聞源的最新內容...")
        logger.info("=" * 80)
        
        results = []
        for source in sources:
            source_name = SOURCE_NAMES.get(source, source)
            logger.info(f"正在獲取 {source_name} 的新聞...")
            result = await self.fetch_news(source)
            results.append(result)
            
            if result["status"] == "success":
                data = result["data"]
                if 'items' in data and isinstance(data['items'], list):
                    count = len(data['items'])
                    logger.info(f"✓ {source_name}: 獲取成功，共 {count} 條新聞")
                else:
                    logger.info(f"✓ {source_name}: 獲取成功")
            else:
                logger.error(f"✗ {source_name}: {result.get('error', '獲取失敗')}")
            
            # 避免請求過快
            await asyncio.sleep(0.5)
        
        return results
    
    # ==================== 數據處理和存儲 ====================
    
    async def collect_and_save_news(self, sources: Optional[List[str]] = None) -> Dict:
        """
        收集並保存每日熱點新聞
        
        Args:
            sources: 指定的新聞源列表，None表示使用所有支持的源
            
        Returns:
            包含收集結果的字典
        """
        collection_summary_message = ""
        collection_summary_message += "\n開始收集每日熱點新聞...\n"
        collection_summary_message += f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # 選擇新聞源
        if sources is None:
            # 使用所有支持的新聞源
            sources = list(SOURCE_NAMES.keys())
        
        collection_summary_message += f"將從 {len(sources)} 個新聞源收集數據:\n"
        for source in sources:
            source_name = SOURCE_NAMES.get(source, source)
            collection_summary_message += f"  - {source_name}\n"
        
        logger.info(collection_summary_message)
        
        try:
            # 獲取新聞數據
            results = await self.get_popular_news(sources)
            
            # 處理結果
            processed_data = self._process_news_results(results)
            
            # 保存到數據庫（覆蓋模式）
            if processed_data['news_list']:
                saved_count = self.db_manager.save_daily_news(
                    processed_data['news_list'], 
                    date.today()
                )
                processed_data['saved_count'] = saved_count
            
            # 打印統計信息
            self._print_collection_summary(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.exception(f"收集新聞失敗: {e}")
            return {
                'success': False,
                'error': str(e),
                'news_list': [],
                'total_news': 0
            }
    
    def _process_news_results(self, results: List[Dict]) -> Dict:
        """處理新聞獲取結果"""
        news_list = []
        successful_sources = 0
        total_news = 0
        
        for result in results:
            source = result['source']
            status = result['status']
            
            if status == 'success':
                successful_sources += 1
                data = result['data']
                
                if 'items' in data and isinstance(data['items'], list):
                    source_news_count = len(data['items'])
                    total_news += source_news_count
                    
                    # 處理該源的新聞
                    for i, item in enumerate(data['items'], 1):
                        processed_news = self._process_news_item(item, source, i)
                        if processed_news:
                            news_list.append(processed_news)
        
        return {
            'success': True,
            'news_list': news_list,
            'successful_sources': successful_sources,
            'total_sources': len(results),
            'total_news': total_news,
            'collection_time': datetime.now().isoformat()
        }
    
    def _process_news_item(self, item: Dict, source: str, rank: int) -> Optional[Dict]:
        """處理單條新聞"""
        try:
            if isinstance(item, dict):
                title = item.get('title', '無標題').strip()
                url = item.get('url', '')
                
                # 生成新聞ID
                news_id = f"{source}_{item.get('id', f'rank_{rank}')}"
                
                return {
                    'id': news_id,
                    'title': title,
                    'url': url,
                    'source': source,
                    'rank': rank
                }
            else:
                # 處理字符串類型的新聞
                title = str(item)[:100] if len(str(item)) > 100 else str(item)
                return {
                    'id': f"{source}_rank_{rank}",
                    'title': title,
                    'url': '',
                    'source': source,
                    'rank': rank
                }
                
        except Exception as e:
            logger.exception(f"處理新聞項失敗: {e}")
            return None
    
    def _print_collection_summary(self, data: Dict):
        """打印收集摘要"""
        collection_summary_message = ""
        collection_summary_message += f"\n總新聞源: {data['total_sources']}\n"
        collection_summary_message += f"成功源數: {data['successful_sources']}\n"
        collection_summary_message += f"總新聞數: {data['total_news']}\n"
        if 'saved_count' in data:
            collection_summary_message += f"已保存數: {data['saved_count']}\n"
        logger.info(collection_summary_message)
    
    def get_today_news(self) -> List[Dict]:
        """獲取今天的新聞"""
        try:
            return self.db_manager.get_daily_news(date.today())
        except Exception as e:
            logger.exception(f"獲取今日新聞失敗: {e}")
            return []

async def main():
    """測試新聞收集器"""
    logger.info("測試新聞收集器...")
    
    async with NewsCollector() as collector:
        # 收集新聞
        result = await collector.collect_and_save_news(
            sources=["weibo", "zhihu"]  # 測試用，只使用兩個源
        )
        
        if result['success']:
            logger.info(f"收集成功！共獲取 {result['total_news']} 條新聞")
        else:
            logger.error(f"收集失敗: {result.get('error', '未知錯誤')}")

if __name__ == "__main__":
    asyncio.run(main())
