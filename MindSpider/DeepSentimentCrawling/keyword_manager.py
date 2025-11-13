#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling模塊 - 關鍵詞管理器
從BroadTopicExtraction模塊獲取關鍵詞並分配給不同平臺進行爬取
"""

import sys
import json
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List, Dict, Optional
import random
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("無法導入config.py配置文件")

from config import settings
from loguru import logger

class KeywordManager:
    """關鍵詞管理器"""
    
    def __init__(self):
        """初始化關鍵詞管理器"""
        self.engine: Engine = None
        self.connect()
    
    def connect(self):
        """連接數據庫"""
        try:
            dialect = (settings.DB_DIALECT or "mysql").lower()
            if dialect in ("postgresql", "postgres"):
                url = f"postgresql+psycopg://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            else:
                url = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?charset={settings.DB_CHARSET}"
            self.engine = create_engine(url, future=True)
            logger.info(f"關鍵詞管理器成功連接到數據庫: {settings.DB_NAME}")
        except ModuleNotFoundError as e:
            missing: str = str(e)
            if "psycopg" in missing:
                logger.error("數據庫連接失敗: 未安裝PostgreSQL驅動 psycopg。請安裝: psycopg[binary]。參考指令：uv pip install psycopg[binary]")
            elif "pymysql" in missing:
                logger.error("數據庫連接失敗: 未安裝MySQL驅動 pymysql。請安裝: pymysql。參考指令：uv pip install pymysql")
            else:
                logger.error(f"數據庫連接失敗(缺少驅動): {e}")
            raise
        except Exception as e:
            logger.exception(f"關鍵詞管理器數據庫連接失敗: {e}")
            raise
    
    def get_latest_keywords(self, target_date: date = None, max_keywords: int = 100) -> List[str]:
        """
        獲取最新的關鍵詞列表
        
        Args:
            target_date: 目標日期，默認爲今天
            max_keywords: 最大關鍵詞數量
        
        Returns:
            關鍵詞列表
        """
        if not target_date:
            target_date = date.today()
        
        logger.info(f"正在獲取 {target_date} 的關鍵詞...")
        
        # 首先嚐試獲取指定日期的關鍵詞
        topics_data = self.get_daily_topics(target_date)
        
        if topics_data and topics_data.get('keywords'):
            keywords = topics_data['keywords']
            logger.info(f"成功獲取 {target_date} 的 {len(keywords)} 個關鍵詞")
            
            # 如果關鍵詞太多，隨機選擇指定數量
            if len(keywords) > max_keywords:
                keywords = random.sample(keywords, max_keywords)
                logger.info(f"隨機選擇了 {max_keywords} 個關鍵詞")
            
            return keywords
        
        # 如果沒有當天的關鍵詞，嘗試獲取最近幾天的
        logger.info(f"{target_date} 沒有關鍵詞數據，嘗試獲取最近的關鍵詞...")
        recent_topics = self.get_recent_topics(days=7)
        
        if recent_topics:
            # 合併最近幾天的關鍵詞
            all_keywords = []
            for topic in recent_topics:
                if topic.get('keywords'):
                    all_keywords.extend(topic['keywords'])
            
            # 去重並限制數量
            unique_keywords = list(set(all_keywords))
            if len(unique_keywords) > max_keywords:
                unique_keywords = random.sample(unique_keywords, max_keywords)
            
            logger.info(f"從最近7天的數據中獲取到 {len(unique_keywords)} 個關鍵詞")
            return unique_keywords
        
        # 如果都沒有，返回默認關鍵詞
        logger.info("沒有找到任何關鍵詞數據，使用默認關鍵詞")
        return self._get_default_keywords()
    
    def get_daily_topics(self, extract_date: date = None) -> Optional[Dict]:
        """
        獲取每日話題分析
        
        Args:
            extract_date: 提取日期，默認爲今天
        
        Returns:
            話題分析數據，如果不存在返回None
        """
        if not extract_date:
            extract_date = date.today()
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM daily_topics WHERE extract_date = :d"),
                    {"d": extract_date},
                ).mappings().first()
            
            if result:
                # 轉爲可變dict再賦值
                result = dict(result)
                result['keywords'] = json.loads(result['keywords']) if result.get('keywords') else []
                return result
            else:
                return None
                
        except Exception as e:
            logger.exception(f"獲取話題分析失敗: {e}")
            return None
    
    def get_recent_topics(self, days: int = 7) -> List[Dict]:
        """
        獲取最近幾天的話題分析
        
        Args:
            days: 天數
        
        Returns:
            話題分析列表
        """
        try:
            start_date = date.today() - timedelta(days=days)
            with self.engine.connect() as conn:
                results = conn.execute(
                    text(
                        """
                        SELECT * FROM daily_topics 
                        WHERE extract_date >= :start_date
                        ORDER BY extract_date DESC
                        """
                    ),
                    {"start_date": start_date},
                ).mappings().all()
            
            # 轉爲可變dict列表再處理
            results = [dict(r) for r in results]
            for result in results:
                result['keywords'] = json.loads(result['keywords']) if result.get('keywords') else []
            
            return results
            
        except Exception as e:
            logger.exception(f"獲取最近話題分析失敗: {e}")
            return []
    
    def _get_default_keywords(self) -> List[str]:
        """獲取默認關鍵詞列表"""
        return [
            "科技", "人工智能", "AI", "編程", "互聯網",
            "創業", "投資", "理財", "股市", "經濟",
            "教育", "學習", "考試", "大學", "就業",
            "健康", "養生", "運動", "美食", "旅遊",
            "時尚", "美妝", "購物", "生活", "家居",
            "電影", "音樂", "遊戲", "娛樂", "明星",
            "新聞", "熱點", "社會", "政策", "環保"
        ]
    
    def get_all_keywords_for_platforms(self, platforms: List[str], target_date: date = None, 
                                      max_keywords: int = 100) -> List[str]:
        """
        爲所有平臺獲取相同的關鍵詞列表
        
        Args:
            platforms: 平臺列表
            target_date: 目標日期
            max_keywords: 最大關鍵詞數量
        
        Returns:
            關鍵詞列表（所有平臺共用）
        """
        keywords = self.get_latest_keywords(target_date, max_keywords)
        
        if keywords:
            logger.info(f"爲 {len(platforms)} 個平臺準備了相同的 {len(keywords)} 個關鍵詞")
            logger.info(f"每個關鍵詞將在所有平臺上進行爬取")
        
        return keywords
    
    def get_keywords_for_platform(self, platform: str, target_date: date = None, 
                                max_keywords: int = 50) -> List[str]:
        """
        爲特定平臺獲取關鍵詞（現在所有平臺使用相同關鍵詞）
        
        Args:
            platform: 平臺名稱
            target_date: 目標日期
            max_keywords: 最大關鍵詞數量
        
        Returns:
            關鍵詞列表（與其他平臺相同）
        """
        keywords = self.get_latest_keywords(target_date, max_keywords)
        
        logger.info(f"爲平臺 {platform} 準備了 {len(keywords)} 個關鍵詞（與其他平臺相同）")
        return keywords
    
    def _filter_keywords_by_platform(self, keywords: List[str], platform: str) -> List[str]:
        """
        根據平臺特性過濾關鍵詞
        
        Args:
            keywords: 原始關鍵詞列表
            platform: 平臺名稱
        
        Returns:
            過濾後的關鍵詞列表
        """
        # 平臺特性關鍵詞映射（可以根據需要調整）
        platform_preferences = {
            'xhs': ['美妝', '時尚', '生活', '美食', '旅遊', '購物', '健康', '養生'],
            'dy': ['娛樂', '音樂', '舞蹈', '搞笑', '美食', '生活', '科技', '教育'],
            'ks': ['生活', '搞笑', '農村', '美食', '手工', '音樂', '娛樂'],
            'bili': ['科技', '遊戲', '動漫', '學習', '編程', '數碼', '科普'],
            'wb': ['熱點', '新聞', '娛樂', '明星', '社會', '時事', '科技'],
            'tieba': ['遊戲', '動漫', '學習', '生活', '興趣', '討論'],
            'zhihu': ['知識', '學習', '科技', '職場', '投資', '教育', '思考']
        }
        
        # 如果平臺有特定偏好，優先選擇相關關鍵詞
        preferred_keywords = platform_preferences.get(platform, [])
        
        if preferred_keywords:
            # 先選擇平臺偏好的關鍵詞
            filtered = []
            remaining = []
            
            for keyword in keywords:
                if any(pref in keyword for pref in preferred_keywords):
                    filtered.append(keyword)
                else:
                    remaining.append(keyword)
            
            # 如果偏好關鍵詞不夠，補充其他關鍵詞
            if len(filtered) < len(keywords) // 2:
                filtered.extend(remaining[:len(keywords) - len(filtered)])
            
            return filtered
        
        # 如果沒有特定偏好，返回原關鍵詞
        return keywords
    
    def get_crawling_summary(self, target_date: date = None) -> Dict:
        """
        獲取爬取任務摘要
        
        Args:
            target_date: 目標日期
        
        Returns:
            爬取摘要信息
        """
        if not target_date:
            target_date = date.today()
        
        topics_data = self.get_daily_topics(target_date)
        
        if topics_data:
            return {
                'date': target_date,
                'keywords_count': len(topics_data.get('keywords', [])),
                'summary': topics_data.get('summary', ''),
                'has_data': True
            }
        else:
            return {
                'date': target_date,
                'keywords_count': 0,
                'summary': '暫無數據',
                'has_data': False
            }
    
    def close(self):
        """關閉數據庫連接"""
        if self.engine:
            self.engine.dispose()
            logger.info("關鍵詞管理器數據庫連接已關閉")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    # 測試關鍵詞管理器
    with KeywordManager() as km:
        # 測試獲取關鍵詞
        keywords = km.get_latest_keywords(max_keywords=20)
        logger.info(f"獲取到的關鍵詞: {keywords}")
        
        # 測試平臺分配
        platforms = ['xhs', 'dy', 'bili']
        distribution = km.distribute_keywords_by_platform(keywords, platforms)
        for platform, kws in distribution.items():
            logger.info(f"{platform}: {kws}")
        
        # 測試爬取摘要
        summary = km.get_crawling_summary()
        logger.info(f"爬取摘要: {summary}")
        
        logger.info("關鍵詞管理器測試完成！")
