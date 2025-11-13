#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction模塊 - 主程序
整合話題提取的完整工作流程和命令行工具
"""

import sys
import asyncio
import argparse
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from BroadTopicExtraction.get_today_news import NewsCollector, SOURCE_NAMES
    from BroadTopicExtraction.topic_extractor import TopicExtractor
    from BroadTopicExtraction.database_manager import DatabaseManager
except ImportError as e:
    logger.exception(f"導入模塊失敗: {e}")
    logger.error("請確保在項目根目錄運行，並且已安裝所有依賴")
    sys.exit(1)

class BroadTopicExtraction:
    """BroadTopicExtraction主要工作流程"""
    
    def __init__(self):
        """初始化"""
        self.news_collector = NewsCollector()
        self.topic_extractor = TopicExtractor()
        self.db_manager = DatabaseManager()
        
        logger.info("BroadTopicExtraction 初始化完成")
    
    def close(self):
        """關閉資源"""
        if self.news_collector:
            self.news_collector.close()
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
    
    async def run_daily_extraction(self, 
                                  news_sources: Optional[List[str]] = None,
                                  max_keywords: int = 100) -> Dict:
        """
        運行每日話題提取流程
        
        Args:
            news_sources: 新聞源列表，None表示使用所有支持的源
            max_keywords: 最大關鍵詞數量
            
        Returns:
            包含完整提取結果的字典
        """
        extraction_result_message = ""
        extraction_result_message += "\nMindSpider AI爬蟲 - 每日話題提取\n"
        extraction_result_message += f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        extraction_result_message += f"目標日期: {date.today()}\n"
        
        if news_sources:
            extraction_result_message += f"指定平臺: {len(news_sources)} 個\n"
            for source in news_sources:
                source_name = SOURCE_NAMES.get(source, source)
                extraction_result_message += f"  - {source_name}\n"
        else:
            extraction_result_message += f"爬取平臺: 全部 {len(SOURCE_NAMES)} 個平臺\n"
        
        extraction_result_message += f"關鍵詞數: 最多 {max_keywords} 個\n"
        
        logger.info(extraction_result_message)
        
        extraction_result = {
            'success': False,
            'extraction_date': date.today().isoformat(),
            'start_time': datetime.now().isoformat(),
            'news_collection': {},
            'topic_extraction': {},
            'database_save': {},
            'error': None
        }
        
        try:
            # 步驟1: 收集新聞
            logger.info("【步驟1】收集熱點新聞...")
            news_result = await self.news_collector.collect_and_save_news(
                sources=news_sources
            )
            
            extraction_result['news_collection'] = {
                'success': news_result['success'],
                'total_news': news_result.get('total_news', 0),
                'successful_sources': news_result.get('successful_sources', 0),
                'total_sources': news_result.get('total_sources', 0)
            }
            
            if not news_result['success'] or not news_result['news_list']:
                raise Exception("新聞收集失敗或沒有獲取到新聞")
            
            # 步驟2: 提取關鍵詞和生成總結
            logger.info("【步驟2】提取關鍵詞和生成總結...")
            keywords, summary = self.topic_extractor.extract_keywords_and_summary(
                news_result['news_list'], 
                max_keywords=max_keywords
            )
            
            extraction_result['topic_extraction'] = {
                'success': len(keywords) > 0,
                'keywords_count': len(keywords),
                'keywords': keywords,
                'summary': summary
            }
            
            if not keywords:
                logger.warning("警告: 沒有提取到有效關鍵詞")
            
            # 步驟3: 保存到數據庫
            logger.info("【步驟3】保存分析結果到數據庫...")
            save_success = self.db_manager.save_daily_topics(
                keywords, summary, date.today()
            )
            
            extraction_result['database_save'] = {
                'success': save_success
            }
            
            extraction_result['success'] = True
            extraction_result['end_time'] = datetime.now().isoformat()
            
            logger.info("每日話題提取流程完成!")
            
            return extraction_result
            
        except Exception as e:
            logger.exception(f"話題提取流程失敗: {e}")
            extraction_result['error'] = str(e)
            extraction_result['end_time'] = datetime.now().isoformat()
            return extraction_result
    
    def print_extraction_results(self, extraction_result: Dict):
        """打印提取結果"""
        extraction_result_message = ""
        
        # 新聞收集結果
        news_data = extraction_result.get('news_collection', {})
        extraction_result_message += f"\n📰 新聞收集: {news_data.get('total_news', 0)} 條新聞\n"
        extraction_result_message += f"   成功源數: {news_data.get('successful_sources', 0)}/{news_data.get('total_sources', 0)}\n"
        
        # 話題提取結果
        topic_data = extraction_result.get('topic_extraction', {})
        keywords = topic_data.get('keywords', [])
        summary = topic_data.get('summary', '')
        
        extraction_result_message += f"\n🔑 提取關鍵詞: {len(keywords)} 個\n"
        if keywords:
            # 每行顯示5個關鍵詞
            for i in range(0, len(keywords), 5):
                keyword_group = keywords[i:i+5]
                extraction_result_message += f"   {', '.join(keyword_group)}\n"
        
        extraction_result_message += f"\n📝 新聞總結:\n   {summary}\n"
        
        # 數據庫保存結果
        db_data = extraction_result.get('database_save', {})
        if db_data.get('success'):
            extraction_result_message += f"\n💾 數據庫保存: 成功\n"
        else:
            extraction_result_message += f"\n💾 數據庫保存: 失敗\n"
        
        logger.info(extraction_result_message)
    
    def get_keywords_for_crawling(self, extract_date: date = None) -> List[str]:
        """
        獲取用於爬取的關鍵詞列表
        
        Args:
            extract_date: 提取日期，默認爲今天
            
        Returns:
            關鍵詞列表
        """
        try:
            # 從數據庫獲取話題分析
            topics_data = self.db_manager.get_daily_topics(extract_date)
            
            if not topics_data:
                logger.info(f"沒有找到 {extract_date or date.today()} 的話題數據")
                return []
            
            keywords = topics_data['keywords']
            
            # 生成搜索關鍵詞
            search_keywords = self.topic_extractor.get_search_keywords(keywords)
            
            logger.info(f"準備了 {len(search_keywords)} 個關鍵詞用於爬取")
            return search_keywords
            
        except Exception as e:
            logger.error(f"獲取爬取關鍵詞失敗: {e}")
            return []
    
    def get_daily_analysis(self, target_date: date = None) -> Optional[Dict]:
        """獲取指定日期的分析結果"""
        try:
            return self.db_manager.get_daily_topics(target_date)
        except Exception as e:
            logger.error(f"獲取每日分析失敗: {e}")
            return None
    
    def get_recent_analysis(self, days: int = 7) -> List[Dict]:
        """獲取最近幾天的分析結果"""
        try:
            return self.db_manager.get_recent_topics(days)
        except Exception as e:
            logger.error(f"獲取最近分析失敗: {e}")
            return []

# ==================== 命令行工具 ====================

async def run_extraction_command(sources=None, keywords_count=100, show_details=True):
    """運行話題提取命令"""
    
    try:
        async with BroadTopicExtraction() as extractor:
            # 運行話題提取
            result = await extractor.run_daily_extraction(
                news_sources=sources,
                max_keywords=keywords_count
            )
            
            if result['success']:
                if show_details:
                    # 顯示詳細結果
                    extractor.print_extraction_results(result)
                else:
                    # 只顯示簡要結果
                    news_data = result.get('news_collection', {})
                    topic_data = result.get('topic_extraction', {})
                    
                    logger.info(f"✅ 話題提取成功完成!")
                    logger.info(f"   收集新聞: {news_data.get('total_news', 0)} 條")
                    logger.info(f"   提取關鍵詞: {len(topic_data.get('keywords', []))} 個")
                    logger.info(f"   生成總結: {len(topic_data.get('summary', ''))} 字符")
                
                # 獲取爬取關鍵詞
                crawling_keywords = extractor.get_keywords_for_crawling()
                
                if crawling_keywords:
                    logger.info(f"\n🔑 爲DeepSentimentCrawling準備的搜索關鍵詞:")
                    logger.info(f"   {', '.join(crawling_keywords)}")
                    
                    # 保存關鍵詞到文件
                    keywords_file = project_root / "data" / "daily_keywords.txt"
                    keywords_file.parent.mkdir(exist_ok=True)
                    
                    with open(keywords_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(crawling_keywords))
                    
                    logger.info(f"   關鍵詞已保存到: {keywords_file}")
                
                return True
                
            else:
                logger.error(f"❌ 話題提取失敗: {result.get('error', '未知錯誤')}")
                return False
                
    except Exception as e:
        logger.error(f"❌ 執行過程中發生錯誤: {e}")
        return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="MindSpider每日話題提取工具")
    parser.add_argument("--sources", nargs="+", help="指定新聞源平臺", 
                       choices=list(SOURCE_NAMES.keys()))
    parser.add_argument("--keywords", type=int, default=100, help="最大關鍵詞數量 (默認100)")
    parser.add_argument("--quiet", action="store_true", help="簡化輸出模式")
    parser.add_argument("--list-sources", action="store_true", help="顯示支持的新聞源")
    
    args = parser.parse_args()
    
    # 顯示支持的新聞源
    if args.list_sources:
        logger.info("支持的新聞源平臺:")
        for source, name in SOURCE_NAMES.items():
            logger.info(f"  {source:<25} {name}")
        return
    
    # 驗證參數
    if args.keywords < 1 or args.keywords > 200:
        logger.error("關鍵詞數量應在1-200之間")
        sys.exit(1)
    
    # 運行提取
    try:
        success = asyncio.run(run_extraction_command(
            sources=args.sources,
            keywords_count=args.keywords,
            show_details=not args.quiet
        ))
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("用戶中斷操作")
        sys.exit(1)

if __name__ == "__main__":
    main()
