#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling模塊 - 主工作流程
基於BroadTopicExtraction提取的話題進行全平臺關鍵詞爬取
"""

import sys
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from keyword_manager import KeywordManager
from platform_crawler import PlatformCrawler

class DeepSentimentCrawling:
    """深度情感爬取主工作流程"""
    
    def __init__(self):
        """初始化深度情感爬取"""
        self.keyword_manager = KeywordManager()
        self.platform_crawler = PlatformCrawler()
        self.supported_platforms = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']
    
    def run_daily_crawling(self, target_date: date = None, platforms: List[str] = None, 
                          max_keywords_per_platform: int = 50, 
                          max_notes_per_platform: int = 50,
                          login_type: str = "qrcode") -> Dict:
        """
        執行每日爬取任務
        
        Args:
            target_date: 目標日期，默認爲今天
            platforms: 要爬取的平臺列表，默認爲所有支持的平臺
            max_keywords_per_platform: 每個平臺最大關鍵詞數量
            max_notes_per_platform: 每個平臺最大爬取內容數量
            login_type: 登錄方式
        
        Returns:
            爬取結果統計
        """
        if not target_date:
            target_date = date.today()
        
        if not platforms:
            platforms = self.supported_platforms
        
        print(f"🚀 開始執行 {target_date} 的深度情感爬取任務")
        print(f"目標平臺: {platforms}")
        
        # 1. 獲取關鍵詞摘要
        summary = self.keyword_manager.get_crawling_summary(target_date)
        print(f"📊 關鍵詞摘要: {summary}")
        
        if not summary['has_data']:
            print("⚠️ 沒有找到話題數據，無法進行爬取")
            return {"success": False, "error": "沒有話題數據"}
        
        # 2. 獲取關鍵詞（不分配，所有平臺使用相同關鍵詞）
        print(f"\n📝 獲取關鍵詞...")
        keywords = self.keyword_manager.get_latest_keywords(target_date, max_keywords_per_platform)
        
        if not keywords:
            print("⚠️ 沒有找到關鍵詞，無法進行爬取")
            return {"success": False, "error": "沒有關鍵詞"}
        
        print(f"   獲取到 {len(keywords)} 個關鍵詞")
        print(f"   將在 {len(platforms)} 個平臺上爬取每個關鍵詞")
        print(f"   總爬取任務: {len(keywords)} × {len(platforms)} = {len(keywords) * len(platforms)}")
        
        # 3. 執行全平臺關鍵詞爬取
        print(f"\n🔄 開始全平臺關鍵詞爬取...")
        crawl_results = self.platform_crawler.run_multi_platform_crawl_by_keywords(
            keywords, platforms, login_type, max_notes_per_platform
        )
        
        # 4. 生成最終報告
        final_report = {
            "date": target_date.isoformat(),
            "summary": summary,
            "crawl_results": crawl_results,
            "success": crawl_results["successful_tasks"] > 0
        }
        
        print(f"\n✅ 深度情感爬取任務完成!")
        print(f"   日期: {target_date}")
        print(f"   成功任務: {crawl_results['successful_tasks']}/{crawl_results['total_tasks']}")
        print(f"   總關鍵詞: {crawl_results['total_keywords']} 個")
        print(f"   總平臺: {crawl_results['total_platforms']} 個")
        print(f"   總內容: {crawl_results['total_notes']} 條")
        
        return final_report
    
    def run_platform_crawling(self, platform: str, target_date: date = None,
                             max_keywords: int = 50, max_notes: int = 50,
                             login_type: str = "qrcode") -> Dict:
        """
        執行單個平臺的爬取任務
        
        Args:
            platform: 平臺名稱
            target_date: 目標日期
            max_keywords: 最大關鍵詞數量
            max_notes: 最大爬取內容數量
            login_type: 登錄方式
        
        Returns:
            爬取結果
        """
        if platform not in self.supported_platforms:
            raise ValueError(f"不支持的平臺: {platform}")
        
        if not target_date:
            target_date = date.today()
        
        print(f"🎯 開始執行 {platform} 平臺的爬取任務 ({target_date})")
        
        # 獲取關鍵詞
        keywords = self.keyword_manager.get_keywords_for_platform(
            platform, target_date, max_keywords
        )
        
        if not keywords:
            print(f"⚠️ 沒有找到 {platform} 平臺的關鍵詞")
            return {"success": False, "error": "沒有關鍵詞"}
        
        print(f"📝 準備爬取 {len(keywords)} 個關鍵詞")
        
        # 執行爬取
        result = self.platform_crawler.run_crawler(
            platform, keywords, login_type, max_notes
        )
        
        return result
    
    def list_available_topics(self, days: int = 7):
        """列出最近可用的話題"""
        print(f"📋 最近 {days} 天的話題數據:")
        
        recent_topics = self.keyword_manager.db_manager.get_recent_topics(days)
        
        if not recent_topics:
            print("   暫無話題數據")
            return
        
        for topic in recent_topics:
            extract_date = topic['extract_date']
            keywords_count = len(topic.get('keywords', []))
            summary_preview = topic.get('summary', '')[:100] + "..." if len(topic.get('summary', '')) > 100 else topic.get('summary', '')
            
            print(f"   📅 {extract_date}: {keywords_count} 個關鍵詞")
            print(f"      摘要: {summary_preview}")
            print()
    
    def show_platform_guide(self):
        """顯示平臺使用指南"""
        print("🔧 平臺爬取指南:")
        print()
        
        platform_info = {
            'xhs': '小紅書 - 美妝、生活、時尚內容爲主',
            'dy': '抖音 - 短視頻、娛樂、生活內容',
            'ks': '快手 - 生活、娛樂、農村題材內容',
            'bili': 'B站 - 科技、學習、遊戲、動漫內容',
            'wb': '微博 - 熱點新聞、明星、社會話題',
            'tieba': '百度貼吧 - 興趣討論、遊戲、學習',
            'zhihu': '知乎 - 知識問答、深度討論'
        }
        
        for platform, desc in platform_info.items():
            print(f"   {platform}: {desc}")
        
        print()
        print("💡 使用建議:")
        print("   1. 首次使用需要掃碼登錄各平臺")
        print("   2. 建議先測試單個平臺，確認登錄正常")
        print("   3. 爬取數量不宜過大，避免被限制")
        print("   4. 可以使用 --test 模式進行小規模測試")
    
    def close(self):
        """關閉資源"""
        if self.keyword_manager:
            self.keyword_manager.close()

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="DeepSentimentCrawling - 基於話題的深度情感爬取")
    
    # 基本參數
    parser.add_argument("--date", type=str, help="目標日期 (YYYY-MM-DD)，默認爲今天")
    parser.add_argument("--platform", type=str, choices=['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu'], 
                       help="指定單個平臺進行爬取")
    parser.add_argument("--platforms", type=str, nargs='+', 
                       choices=['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu'],
                       help="指定多個平臺進行爬取")
    
    # 爬取參數
    parser.add_argument("--max-keywords", type=int, default=50, 
                       help="每個平臺最大關鍵詞數量 (默認: 50)")
    parser.add_argument("--max-notes", type=int, default=50,
                       help="每個平臺最大爬取內容數量 (默認: 50)")
    parser.add_argument("--login-type", type=str, choices=['qrcode', 'phone', 'cookie'], 
                       default='qrcode', help="登錄方式 (默認: qrcode)")
    
    # 功能參數
    parser.add_argument("--list-topics", action="store_true", help="列出最近的話題數據")
    parser.add_argument("--days", type=int, default=7, help="查看最近幾天的話題 (默認: 7)")
    parser.add_argument("--guide", action="store_true", help="顯示平臺使用指南")
    parser.add_argument("--test", action="store_true", help="測試模式 (少量數據)")
    
    args = parser.parse_args()
    
    # 解析日期
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("❌ 日期格式錯誤，請使用 YYYY-MM-DD 格式")
            return
    
    # 創建爬取實例
    crawler = DeepSentimentCrawling()
    
    try:
        # 顯示指南
        if args.guide:
            crawler.show_platform_guide()
            return
        
        # 列出話題
        if args.list_topics:
            crawler.list_available_topics(args.days)
            return
        
        # 測試模式調整參數
        if args.test:
            args.max_keywords = min(args.max_keywords, 10)
            args.max_notes = min(args.max_notes, 10)
            print("測試模式：限制關鍵詞和內容數量")
        
        # 單平臺爬取
        if args.platform:
            result = crawler.run_platform_crawling(
                args.platform, target_date, args.max_keywords, 
                args.max_notes, args.login_type
            )
            
            if result['success']:
                print(f"\n{args.platform} 爬取成功！")
            else:
                print(f"\n{args.platform} 爬取失敗: {result.get('error', '未知錯誤')}")
            
            return
        
        # 多平臺爬取
        platforms = args.platforms if args.platforms else None
        result = crawler.run_daily_crawling(
            target_date, platforms, args.max_keywords, 
            args.max_notes, args.login_type
        )
        
        if result['success']:
            print(f"\n多平臺爬取任務完成！")
        else:
            print(f"\n多平臺爬取失敗: {result.get('error', '未知錯誤')}")
    
    except KeyboardInterrupt:
        print("\n用戶中斷操作")
    except Exception as e:
        print(f"\n執行出錯: {e}")
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
