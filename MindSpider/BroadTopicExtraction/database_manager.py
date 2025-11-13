#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction模塊 - 數據庫管理器
只負責新聞數據和話題分析的存儲和查詢
"""

import sys
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from loguru import logger

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("無法導入config.py配置文件")

from config import settings


class DatabaseManager:
    """數據庫管理器"""

    def __init__(self):
        """初始化數據庫管理器"""
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
            logger.info(f"成功連接到數據庫: {settings.DB_NAME}")
        except ModuleNotFoundError as e:
            missing: str = str(e)
            if "psycopg" in missing:
                logger.error(
                    "數據庫連接失敗: 未安裝PostgreSQL驅動 psycopg。請安裝: psycopg[binary]。參考指令：uv pip install psycopg[binary]")
            elif "pymysql" in missing:
                logger.error("數據庫連接失敗: 未安裝MySQL驅動 pymysql。請安裝: pymysql。參考指令：uv pip install pymysql")
            else:
                logger.error(f"數據庫連接失敗(缺少驅動): {e}")
            raise
        except Exception as e:
            logger.exception(f"數據庫連接失敗: {e}")
            raise

    def close(self):
        """關閉數據庫連接"""
        if self.engine:
            self.engine.dispose()
            logger.info("數據庫連接已關閉")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 新聞數據操作 ====================

    def save_daily_news(self, news_data: List[Dict], crawl_date: date = None) -> int:
        """
        保存每日新聞數據，如果當天已有數據則覆蓋

        Args:
            news_data: 新聞數據列表
            crawl_date: 爬取日期，默認爲今天

        Returns:
            保存的新聞數量
        """
        if not crawl_date:
            crawl_date = date.today()

        current_timestamp = int(datetime.now().timestamp())

        try:
            saved_count = 0
            # 先獨立事務執行刪除，防止後續插入失敗導致無法清理
            with self.engine.begin() as conn:
                deleted = conn.execute(text("DELETE FROM daily_news WHERE crawl_date = :d"), {"d": crawl_date}).rowcount
                if deleted and deleted > 0:
                    logger.info(f"覆蓋模式：刪除了當天已有的 {deleted} 條新聞記錄")

            # 逐條插入，單條失敗不影響後續（每條獨立事務）
            for news_item in news_data:
                try:
                    # news_item.get('id') 已經是完整的 news_id（格式：source_item_id）
                    # 爲了支持同一條新聞在不同日期出現，將 crawl_date 加入到 news_id 中
                    base_news_id = news_item.get(
                        'id') or f"{news_item.get('source', 'unknown')}_rank_{news_item.get('rank', 0)}"
                    # 將日期格式化爲字符串並加入到 news_id 中，確保全局唯一性
                    news_id = f"{base_news_id}_{crawl_date.strftime('%Y%m%d')}"

                    title_val = (news_item.get("title", "") or "")
                    if len(title_val) > 500:
                        title_val = title_val[:500]
                    with self.engine.begin() as conn:
                        conn.execute(
                            text(
                                """
                                INSERT INTO daily_news (
                                    news_id, source_platform, title, url, crawl_date,
                                    rank_position, add_ts, last_modify_ts
                                ) VALUES (:news_id, :source_platform, :title, :url, :crawl_date, :rank_position, :add_ts, :last_modify_ts)
                                """
                            ),
                            {
                                "news_id": news_id,
                                "source_platform": news_item.get("source", "unknown"),
                                "title": title_val,
                                "url": news_item.get("url", ""),
                                "crawl_date": crawl_date,
                                "rank_position": news_item.get("rank", None),
                                "add_ts": current_timestamp,
                                "last_modify_ts": current_timestamp,
                            },
                        )
                    saved_count += 1
                except Exception as e:
                    logger.exception(f"保存單條新聞失敗: {e}")
                    continue
            logger.info(f"成功保存 {saved_count} 條新聞記錄")
            return saved_count
        except Exception as e:
            logger.exception(f"保存新聞數據失敗: {e}")
            return 0

    def get_daily_news(self, crawl_date: date = None) -> List[Dict]:
        """
        獲取每日新聞數據

        Args:
            crawl_date: 爬取日期，默認爲今天

        Returns:
            新聞列表
        """
        if not crawl_date:
            crawl_date = date.today()

        query = (
            "SELECT * FROM daily_news WHERE crawl_date = :d ORDER BY rank_position ASC"
        )
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"d": crawl_date})
            rows = result.mappings().all()
        return rows

    # ==================== 話題數據操作 ====================

    def save_daily_topics(self, keywords: List[str], summary: str, extract_date: date = None) -> bool:
        """
        保存每日話題分析

        Args:
            keywords: 話題關鍵詞列表
            summary: 新聞分析總結
            extract_date: 提取日期，默認爲今天

        Returns:
            是否保存成功
        """
        if not extract_date:
            extract_date = date.today()

        current_timestamp = int(datetime.now().timestamp())

        try:
            keywords_json = json.dumps(keywords, ensure_ascii=False)
            # 爲了支持外鍵引用，topic_id 需要全局唯一，所以將日期加入到 topic_id 中
            topic_id = f"summary_{extract_date.strftime('%Y%m%d')}"

            with self.engine.begin() as conn:
                check = conn.execute(
                    text("SELECT id FROM daily_topics WHERE extract_date = :d AND topic_id = :tid"),
                    {"d": extract_date, "tid": topic_id},
                ).first()
                if check:
                    conn.execute(
                        text(
                            "UPDATE daily_topics SET keywords = :k, topic_description = :s, add_ts = :ts, last_modify_ts = :lmt, topic_name = :tn WHERE extract_date = :d AND topic_id = :tid"
                        ),
                        {"k": keywords_json, "s": summary, "ts": current_timestamp, "lmt": current_timestamp,
                         "d": extract_date, "tid": topic_id, "tn": "每日新聞分析"},
                    )
                    logger.info(f"更新了 {extract_date} 的話題分析")
                else:
                    conn.execute(
                        text(
                            "INSERT INTO daily_topics (extract_date, topic_id, topic_name, keywords, topic_description, add_ts, last_modify_ts) VALUES (:d, :tid, :tn, :k, :s, :ts, :lmt)"
                        ),
                        {"d": extract_date, "tid": topic_id, "tn": "每日新聞分析", "k": keywords_json, "s": summary,
                         "ts": current_timestamp, "lmt": current_timestamp},
                    )
                    logger.info(f"保存了 {extract_date} 的話題分析")
            return True
        except Exception as e:
            logger.exception(f"保存話題分析失敗: {e}")
            return False

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
                result = conn.execute(text("SELECT * FROM daily_topics WHERE extract_date = :d"),
                                      {"d": extract_date}).mappings().first()
                if result:
                    result = dict(result)  # 轉爲可變dict以支持item賦值
                    result["keywords"] = json.loads(result["keywords"]) if result.get("keywords") else []
                    return result
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
                for r in results:
                    r["keywords"] = json.loads(r["keywords"]) if r.get("keywords") else []
                return results
        except Exception as e:
            logger.exception(f"獲取最近話題分析失敗: {e}")
            return []

    # ==================== 統計查詢 ====================

    def get_summary_stats(self, days: int = 7) -> Dict:
        """獲取統計摘要"""
        try:
            start_date = date.today() - timedelta(days=days)
            with self.engine.connect() as conn:
                news_stats = conn.execute(
                    text(
                        """
                        SELECT crawl_date, COUNT(*) as news_count, COUNT(DISTINCT source_platform) as platforms_count
                        FROM daily_news 
                        WHERE crawl_date >= :start_date
                        GROUP BY crawl_date
                        ORDER BY crawl_date DESC
                        """
                    ),
                    {"start_date": start_date},
                ).all()
                topics_stats = conn.execute(
                    text(
                        """
                        SELECT extract_date, keywords, CHAR_LENGTH(topic_description) as summary_length
                        FROM daily_topics 
                        WHERE extract_date >= :start_date
                        ORDER BY extract_date DESC
                        """
                    ),
                    {"start_date": start_date},
                ).all()
                return {"news_stats": news_stats, "topics_stats": topics_stats}
        except Exception as e:
            logger.exception(f"獲取統計摘要失敗: {e}")
            return {"news_stats": [], "topics_stats": []}


if __name__ == "__main__":
    # 測試數據庫管理器
    with DatabaseManager() as db:
        # 測試獲取新聞
        news = db.get_daily_news()
        logger.info(f"今日新聞數量: {len(news)}")

        # 測試獲取話題
        topics = db.get_daily_topics()
        if topics:
            logger.info(f"今日話題關鍵詞: {topics['keywords']}")
        else:
            logger.info("今日暫無話題分析")

        logger.info("簡化數據庫管理器測試完成！")
